"""
traffic_env_v2.py

Corrected scalable multi-agent SUMO traffic-signal environment.

Revision goals
--------------
1. Preserve the original six-feature observation representation.
2. Preserve automatic traffic-light and neighbour discovery.
3. Preserve the original reward formulation.
4. Preserve 5-second agent decision intervals.
5. Preserve 3-second yellow transitions.
6. Correct the control semantics:

   Action 0 = HOLD the current green phase.
   Action 1 = request transition to the opposite green,
              subject to a minimum-green constraint.

Unlike the original environment, SUMO is prevented from automatically
ending a green phase after the static 42-second duration. The MARL
controller therefore determines when a green phase ends.

The original network XML files are NOT modified.
"""

import os
import sys

import gymnasium as gym
import numpy as np
import sumolib
import traci

from gymnasium import spaces


# ============================================================
# SUMO SETUP
# ============================================================

if "SUMO_HOME" in os.environ:
    tools = os.path.join(
        os.environ["SUMO_HOME"],
        "tools",
    )

    if tools not in sys.path:
        sys.path.append(tools)


# ============================================================
# EXPERIMENT CONSTANTS
# ============================================================

MIN_GREEN = 10

YELLOW_TIME = 3

# Large remaining phase duration used so that SUMO does not
# automatically terminate a green phase.
#
# This does not modify the .net.xml file. It only changes the
# remaining duration of the current phase during this simulation.
CONTROLLED_GREEN_DURATION = 100000.0


# Reward coefficients retained from the original experiment.

ALPHA = 0.3
BETA = 0.2
DELTA = 0.1
LAMBDA = 0.5


# ============================================================
# TOPOLOGY DISCOVERY
# ============================================================

def discover_tls_and_neighbours(net_file):
    """
    Discover all traffic-light-controlled junctions and their
    one-hop traffic-light neighbours.

    No intersection IDs are hard-coded.
    """

    net = sumolib.net.readNet(net_file)

    tls_nodes = [
        node
        for node in net.getNodes()
        if node.getType() == "traffic_light"
    ]

    tls_ids = sorted(
        node.getID()
        for node in tls_nodes
    )

    tls_id_set = set(tls_ids)

    neighbours = {
        tls_id: []
        for tls_id in tls_ids
    }

    for node in tls_nodes:

        this_id = node.getID()

        neighbour_ids = set()

        for edge in node.getOutgoing():

            to_node = edge.getToNode()

            if (
                to_node.getID() in tls_id_set
                and to_node.getID() != this_id
            ):
                neighbour_ids.add(
                    to_node.getID()
                )

        for edge in node.getIncoming():

            from_node = edge.getFromNode()

            if (
                from_node.getID() in tls_id_set
                and from_node.getID() != this_id
            ):
                neighbour_ids.add(
                    from_node.getID()
                )

        neighbours[this_id] = sorted(
            neighbour_ids
        )

    return tls_ids, neighbours


# ============================================================
# PHASE CLASSIFICATION
# ============================================================

def classify_signal_state(state):
    """
    Classify a SUMO signal state.

    Returns:
        GREEN
        YELLOW
        ALL_RED
        MIXED
        OTHER
    """

    has_green = any(
        char in state
        for char in ("G", "g")
    )

    has_yellow = any(
        char in state
        for char in ("Y", "y")
    )

    if has_green and has_yellow:
        return "MIXED"

    if has_green:
        return "GREEN"

    if has_yellow:
        return "YELLOW"

    if state and all(
        char in ("r", "R")
        for char in state
    ):
        return "ALL_RED"

    return "OTHER"


# ============================================================
# ENVIRONMENT
# ============================================================

class TrafficSignalEnvV2(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        net_file,
        route_file,
        use_gui=False,
        num_seconds=1800,
        decision_interval=5,
        seed=None,
        tripinfo_output=None,
    ):

        super().__init__()

        self.net_file = str(net_file)
        self.route_file = str(route_file)

        self.use_gui = use_gui

        self.num_seconds = int(
            num_seconds
        )

        self.decision_interval = int(
            decision_interval
        )

        self.seed_value = seed

        self.tripinfo_output = (
            str(tripinfo_output)
            if tripinfo_output is not None
            else None
        )

        self.sumo_binary = (
            "sumo-gui"
            if use_gui
            else "sumo"
        )

        self.sim_started = False

        # ----------------------------------------------------
        # Automatic topology discovery
        # ----------------------------------------------------

        (
            self.tls_ids,
            self.neighbours,
        ) = discover_tls_and_neighbours(
            self.net_file
        )

        self.num_agents = len(
            self.tls_ids
        )

        # ----------------------------------------------------
        # Observation/action spaces
        # ----------------------------------------------------

        self.n_features = 6

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(
            2
        )

        # ----------------------------------------------------
        # Runtime structures
        # ----------------------------------------------------

        self.controlled_lanes = {}

        self.prev_waiting = {}

        self.prev_vehicle_ids = {}

        # Actual elapsed GREEN time.
        #
        # Unlike the original implementation, this is not merely
        # "time since action 1 was requested".
        self.green_elapsed = {}

        # Current controlled green phase.
        self.current_green_phase = {}

        # Phase mappings discovered from the actual SUMO
        # traffic-light programme.
        self.green_phases = {}

        self.yellow_after_green = {}

        self.next_green = {}

        # Used during the 3-second yellow transition.
        self.yellow_remaining = {}

        self.pending_next_green = {}


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

        if self.sim_started:

            traci.close()

            self.sim_started = False

        sumo_cmd = [
            self.sumo_binary,

            "-n",
            self.net_file,

            "-r",
            self.route_file,

            "--no-warnings",
            "true",

            "--no-step-log",
            "true",

            "--time-to-teleport",
            "300",
        ]

        # Gym reset seed takes priority if explicitly supplied.

        actual_seed = (
            seed
            if seed is not None
            else self.seed_value
        )

        if actual_seed is not None:

            sumo_cmd += [
                "--seed",
                str(actual_seed),
            ]

        if self.tripinfo_output is not None:

            sumo_cmd += [
                "--tripinfo-output",
                self.tripinfo_output,
            ]

        traci.start(
            sumo_cmd
        )

        self.sim_started = True

        # ----------------------------------------------------
        # Discover and validate the traffic-light programmes
        # ----------------------------------------------------

        self._discover_phase_structure()

        # ----------------------------------------------------
        # Initialise agent data
        # ----------------------------------------------------

        for tls in self.tls_ids:

            lanes = (
                traci.trafficlight
                .getControlledLanes(tls)
            )

            # Remove duplicate lane IDs while preserving order.

            self.controlled_lanes[tls] = list(
                dict.fromkeys(lanes)
            )

            self.prev_waiting[tls] = 0.0

            self.prev_vehicle_ids[tls] = set()

            self.green_elapsed[tls] = 0

            self.yellow_remaining[tls] = 0

            self.pending_next_green[tls] = None

            # Begin every controller at the first legal green
            # phase in the configured programme.

            initial_green = (
                self.green_phases[tls][0]
            )

            self.current_green_phase[tls] = (
                initial_green
            )

            traci.trafficlight.setPhase(
                tls,
                initial_green,
            )

            # Prevent SUMO from automatically leaving this green.

            traci.trafficlight.setPhaseDuration(
                tls,
                CONTROLLED_GREEN_DURATION,
            )

        observations = {
            tls: self._get_observation(tls)
            for tls in self.tls_ids
        }

        info = {
            "num_agents": self.num_agents,
            "tls_ids": list(self.tls_ids),
        }

        return observations, info


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        actions,
    ):

        switch_cost = {
            tls: 0
            for tls in self.tls_ids
        }

        switching_this_step = {
            tls: False
            for tls in self.tls_ids
        }

        # ----------------------------------------------------
        # Interpret agent actions
        # ----------------------------------------------------

        for tls in self.tls_ids:

            action = int(
                actions.get(
                    tls,
                    0,
                )
            )

            can_switch = (
                self.green_elapsed[tls]
                >= MIN_GREEN
            )

            if (
                action == 1
                and can_switch
            ):

                self._begin_transition(
                    tls
                )

                switch_cost[tls] = 1

                switching_this_step[tls] = True

            else:

                # Action 0 means TRUE HOLD.
                #
                # The green phase has already been assigned a
                # very long remaining duration, so SUMO cannot
                # leave it automatically.

                current_green = (
                    self.current_green_phase[tls]
                )

                traci.trafficlight.setPhaseDuration(
                    tls,
                    CONTROLLED_GREEN_DURATION,
                )

        # ----------------------------------------------------
        # Advance SUMO one second at a time.
        #
        # This allows the 3-second yellow transition to be
        # represented exactly even though the agent decision
        # interval is 5 seconds.
        # ----------------------------------------------------

        for _ in range(
            self.decision_interval
        ):

            traci.simulationStep()

            for tls in self.tls_ids:

                if self.yellow_remaining[tls] > 0:

                    self.yellow_remaining[tls] -= 1

                    if (
                        self.yellow_remaining[tls]
                        == 0
                    ):

                        self._complete_transition(
                            tls
                        )

                else:

                    # The agent is currently serving a green.
                    self.green_elapsed[tls] += 1

        # ----------------------------------------------------
        # Observations and rewards
        # ----------------------------------------------------

        observations = {}

        rewards = {}

        for tls in self.tls_ids:

            observations[tls] = (
                self._get_observation(
                    tls
                )
            )

            rewards[tls] = (
                self._get_reward(
                    tls,
                    switch_cost[tls],
                )
            )

        simulation_time = (
            traci.simulation.getTime()
        )

        terminated = (
            simulation_time
            >= self.num_seconds
        )

        truncated = False

        info = {
            "simulation_time": simulation_time,
            "switches": switch_cost,
        }

        return (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )


    # ========================================================
    # PHASE STRUCTURE
    # ========================================================

    def _discover_phase_structure(
        self,
    ):
        """
        Validate and map the actual phase structure for every TLS.

        Expected programme:

            green
              ->
            yellow
              ->
            green
              ->
            yellow

        No intersection ID is hard-coded.
        """

        for tls in self.tls_ids:

            program_id = (
                traci.trafficlight
                .getProgram(tls)
            )

            logics = (
                traci.trafficlight
                .getAllProgramLogics(tls)
            )

            if not logics:

                raise RuntimeError(
                    f"No traffic-light programme "
                    f"found for TLS {tls}."
                )

            logic = None

            for candidate in logics:

                candidate_id = getattr(
                    candidate,
                    "programID",
                    None,
                )

                if candidate_id == program_id:

                    logic = candidate

                    break

            if logic is None:
                logic = logics[0]

            phases = list(
                logic.phases
            )

            classifications = [
                classify_signal_state(
                    phase.state
                )
                for phase in phases
            ]

            expected = [
                "GREEN",
                "YELLOW",
                "GREEN",
                "YELLOW",
            ]

            if classifications != expected:

                raise RuntimeError(
                    "\nUnsupported traffic-light structure.\n"
                    f"TLS: {tls}\n"
                    f"Programme: {program_id}\n"
                    f"Observed: {classifications}\n"
                    f"Expected: {expected}"
                )

            durations = [
                float(
                    phase.duration
                )
                for phase in phases
            ]

            # Validate the yellow duration used by the experiment.

            if (
                abs(
                    durations[1]
                    - YELLOW_TIME
                )
                > 1e-9
                or abs(
                    durations[3]
                    - YELLOW_TIME
                )
                > 1e-9
            ):

                raise RuntimeError(
                    f"TLS {tls} does not use "
                    f"{YELLOW_TIME}s yellow phases. "
                    f"Durations = {durations}"
                )

            green_indices = [
                0,
                2,
            ]

            self.green_phases[tls] = (
                green_indices
            )

            self.yellow_after_green[tls] = {
                0: 1,
                2: 3,
            }

            self.next_green[tls] = {
                0: 2,
                2: 0,
            }


    # ========================================================
    # CONTROL TRANSITION
    # ========================================================

    def _begin_transition(
        self,
        tls,
    ):
        """
        Begin:

            current green
                ->
            corresponding yellow
        """

        current_green = (
            self.current_green_phase[tls]
        )

        yellow_phase = (
            self.yellow_after_green[tls][
                current_green
            ]
        )

        target_green = (
            self.next_green[tls][
                current_green
            ]
        )

        traci.trafficlight.setPhase(
            tls,
            yellow_phase,
        )

        traci.trafficlight.setPhaseDuration(
            tls,
            YELLOW_TIME,
        )

        self.yellow_remaining[tls] = (
            YELLOW_TIME
        )

        self.pending_next_green[tls] = (
            target_green
        )

        # Do not start counting the new green until yellow ends.

        self.green_elapsed[tls] = 0


    def _complete_transition(
        self,
        tls,
    ):
        """
        Finish:

            yellow
              ->
            target green
        """

        target_green = (
            self.pending_next_green[tls]
        )

        if target_green is None:

            raise RuntimeError(
                f"TLS {tls} has no pending target green."
            )

        traci.trafficlight.setPhase(
            tls,
            target_green,
        )

        traci.trafficlight.setPhaseDuration(
            tls,
            CONTROLLED_GREEN_DURATION,
        )

        self.current_green_phase[tls] = (
            target_green
        )

        self.pending_next_green[tls] = None

        self.green_elapsed[tls] = 0


    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_observation(
        self,
        tls,
    ):

        lanes = (
            self.controlled_lanes[tls]
        )

        queue = sum(
            traci.lane
            .getLastStepHaltingNumber(lane)
            for lane in lanes
        )

        waiting = sum(
            traci.lane
            .getWaitingTime(lane)
            for lane in lanes
        )

        if lanes:

            occupancy = float(
                np.mean(
                    [
                        traci.lane
                        .getLastStepOccupancy(
                            lane
                        )
                        for lane in lanes
                    ]
                )
            )

        else:

            occupancy = 0.0

        phase = (
            traci.trafficlight
            .getPhase(tls)
        )

        logics = (
            traci.trafficlight
            .getAllProgramLogics(tls)
        )

        n_phases = len(
            logics[0].phases
        )

        phase_norm = (
            phase
            / max(
                n_phases - 1,
                1,
            )
        )

        time_in_phase_norm = min(
            self.green_elapsed[tls]
            / 60.0,
            1.0,
        )

        neighbour_queue = sum(

            sum(
                traci.lane
                .getLastStepHaltingNumber(
                    lane
                )
                for lane in self.controlled_lanes.get(
                    neighbour,
                    [],
                )
            )

            for neighbour
            in self.neighbours[tls]
        )

        number_of_neighbours = max(
            len(
                self.neighbours[tls]
            ),
            1,
        )

        neighbour_queue_norm = min(
            (
                neighbour_queue
                / number_of_neighbours
            )
            / 10.0,
            1.0,
        )

        observation = np.array(
            [
                min(
                    queue / 20.0,
                    1.0,
                ),

                min(
                    waiting / 200.0,
                    1.0,
                ),

                min(
                    occupancy,
                    1.0,
                ),

                min(
                    phase_norm,
                    1.0,
                ),

                time_in_phase_norm,

                neighbour_queue_norm,
            ],
            dtype=np.float32,
        )

        return observation


    # ========================================================
    # REWARD
    # ========================================================

    def _get_reward(
        self,
        tls,
        switched,
    ):

        lanes = (
            self.controlled_lanes[tls]
        )

        waiting = sum(
            traci.lane
            .getWaitingTime(lane)
            for lane in lanes
        )

        queue = sum(
            traci.lane
            .getLastStepHaltingNumber(lane)
            for lane in lanes
        )

        current_vehicle_ids = set()

        for lane in lanes:

            current_vehicle_ids.update(
                traci.lane
                .getLastStepVehicleIDs(
                    lane
                )
            )

        throughput = len(
            self.prev_vehicle_ids[tls]
            - current_vehicle_ids
        )

        self.prev_vehicle_ids[tls] = (
            current_vehicle_ids
        )

        delta_waiting = (
            waiting
            - self.prev_waiting[tls]
        )

        self.prev_waiting[tls] = (
            waiting
        )

        reward = (
            -ALPHA * delta_waiting
            -BETA * queue
            +DELTA * throughput
            -LAMBDA * switched
        )

        return float(
            reward
        )


    # ========================================================
    # CLOSE
    # ========================================================

    def close(
        self,
    ):

        if self.sim_started:

            traci.close()

            self.sim_started = False


# Compatibility alias.
#
# Later training scripts may import:
#
# from traffic_env_v2 import TrafficSignalEnv
#
# without needing another class name.

TrafficSignalEnv = TrafficSignalEnvV2
