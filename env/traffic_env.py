"""
traffic_env.py (Phase 2: generalized for any grid size)

Multi-agent traffic signal control environment for a SUMO grid network
of ANY size. Traffic-light intersections and their neighbours are
auto-discovered from the network file at reset time -- nothing is
hardcoded to a specific set of intersection IDs, so the same code
supports 2x2 (4 agents), 3x3 (9 agents), 4x4 (16 agents) and 5x5
(25 agents) without modification.
"""

import os
import sys
import numpy as np
import traci
import sumolib
import gymnasium as gym
from gymnasium import spaces

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

MIN_GREEN = 10
YELLOW_TIME = 3

ALPHA = 0.3
BETA = 0.2
DELTA = 0.1
LAMBDA = 0.5


def discover_tls_and_neighbours(net_file):
    """
    Reads the .net.xml file with sumolib and returns:
      - tls_ids: sorted list of all traffic-light-controlled junction IDs
      - neighbours: dict {tls_id: [neighbour_tls_ids]}, found by checking
        which OTHER traffic-light junctions are directly connected by a
        single edge (i.e. one hop away) to each junction.
    This replaces the old hardcoded TLS_IDS list and neighbours dict.
    """
    net = sumolib.net.readNet(net_file)
    tls_nodes = [n for n in net.getNodes() if n.getType() == "traffic_light"]
    tls_ids = sorted([n.getID() for n in tls_nodes])
    tls_id_set = set(tls_ids)

    neighbours = {tid: [] for tid in tls_ids}
    for node in tls_nodes:
        this_id = node.getID()
        neighbour_ids = set()
        for edge in node.getOutgoing():
            to_node = edge.getToNode()
            if to_node.getID() in tls_id_set and to_node.getID() != this_id:
                neighbour_ids.add(to_node.getID())
        for edge in node.getIncoming():
            from_node = edge.getFromNode()
            if from_node.getID() in tls_id_set and from_node.getID() != this_id:
                neighbour_ids.add(from_node.getID())
        neighbours[this_id] = sorted(neighbour_ids)

    return tls_ids, neighbours


class TrafficSignalEnv(gym.Env):
    def __init__(self, net_file, route_file, use_gui=False, num_seconds=1800,
                 decision_interval=5, seed=None, tripinfo_output=None):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.decision_interval = decision_interval
        self.seed_value = seed
        self.tripinfo_output = tripinfo_output

        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.sim_started = False

        self.tls_ids, self.neighbours = discover_tls_and_neighbours(net_file)
        self.num_agents = len(self.tls_ids)

        self.time_since_switch = {tls: 0 for tls in self.tls_ids}
        self.prev_waiting = {tls: 0.0 for tls in self.tls_ids}
        self.prev_vehicle_ids = {tls: set() for tls in self.tls_ids}

        self.n_features = 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.controlled_lanes = {}

    def reset(self, *, seed=None, options=None):
        if self.sim_started:
            traci.close()
            self.sim_started = False

        sumo_cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "300",
        ]
        if self.seed_value is not None:
            sumo_cmd += ["--seed", str(self.seed_value)]
        if self.tripinfo_output is not None:
            sumo_cmd += ["--tripinfo-output", self.tripinfo_output]

        traci.start(sumo_cmd)
        self.sim_started = True

        for tls in self.tls_ids:
            self.controlled_lanes[tls] = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls)))
            self.time_since_switch[tls] = 0
            self.prev_waiting[tls] = 0.0
            self.prev_vehicle_ids[tls] = set()

        obs = {tls: self._get_observation(tls) for tls in self.tls_ids}
        info = {}
        return obs, info

    def step(self, actions):
        switch_cost = {tls: 0 for tls in self.tls_ids}

        for tls in self.tls_ids:
            act = actions.get(tls, 0)
            can_switch = self.time_since_switch[tls] >= MIN_GREEN
            if act == 1 and can_switch:
                self._advance_phase(tls)
                self.time_since_switch[tls] = 0
                switch_cost[tls] = 1
            else:
                self.time_since_switch[tls] += self.decision_interval

        for _ in range(self.decision_interval):
            traci.simulationStep()

        obs, rewards, info = {}, {}, {}
        for tls in self.tls_ids:
            obs[tls] = self._get_observation(tls)
            rewards[tls] = self._get_reward(tls, switch_cost[tls])

        terminated = traci.simulation.getTime() >= self.num_seconds
        truncated = False

        return obs, rewards, terminated, truncated, info

    def close(self):
        if self.sim_started:
            traci.close()
            self.sim_started = False

    def _advance_phase(self, tls):
        program = traci.trafficlight.getAllProgramLogics(tls)[0]
        n_phases = len(program.phases)
        current = traci.trafficlight.getPhase(tls)
        traci.trafficlight.setPhase(tls, (current + 1) % n_phases)

    def _get_observation(self, tls):
        lanes = self.controlled_lanes[tls]
        queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        waiting = sum(traci.lane.getWaitingTime(l) for l in lanes)
        occupancy = np.mean([traci.lane.getLastStepOccupancy(l) for l in lanes]) if lanes else 0.0
        phase = traci.trafficlight.getPhase(tls)
        n_phases = len(traci.trafficlight.getAllProgramLogics(tls)[0].phases)
        phase_norm = phase / max(n_phases - 1, 1)
        time_in_phase_norm = min(self.time_since_switch[tls] / 60.0, 1.0)

        neighbour_queue = sum(
            sum(traci.lane.getLastStepHaltingNumber(l) for l in self.controlled_lanes.get(n, []))
            for n in self.neighbours[tls]
        )
        n_neighbours = max(len(self.neighbours[tls]), 1)
        neighbour_queue_norm = min((neighbour_queue / n_neighbours) / 10.0, 1.0)

        obs = np.array([
            min(queue / 20.0, 1.0),
            min(waiting / 200.0, 1.0),
            min(occupancy, 1.0),
            phase_norm,
            time_in_phase_norm,
            neighbour_queue_norm,
        ], dtype=np.float32)
        return obs

    def _get_reward(self, tls, switched):
        lanes = self.controlled_lanes[tls]
        waiting = sum(traci.lane.getWaitingTime(l) for l in lanes)
        queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

        current_vehicle_ids = set()
        for l in lanes:
            current_vehicle_ids.update(traci.lane.getLastStepVehicleIDs(l))
        throughput = len(self.prev_vehicle_ids[tls] - current_vehicle_ids)
        self.prev_vehicle_ids[tls] = current_vehicle_ids

        delta_waiting = waiting - self.prev_waiting[tls]
        self.prev_waiting[tls] = waiting

        reward = (
            -ALPHA * delta_waiting
            - BETA * queue
            + DELTA * throughput
            - LAMBDA * switched
        )
        return reward
