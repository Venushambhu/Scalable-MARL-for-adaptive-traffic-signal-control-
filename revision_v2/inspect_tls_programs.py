import xml.etree.ElementTree as ET

from common import NETWORK_FILES, GRID_AGENT_COUNTS


def classify_phase(state):
    has_green = any(c in state for c in ("G", "g"))
    has_yellow = any(c in state for c in ("Y", "y"))

    if has_green and has_yellow:
        return "MIXED"

    if has_green:
        return "GREEN"

    if has_yellow:
        return "YELLOW"

    if state and all(c in ("r", "R") for c in state):
        return "ALL-RED"

    return "OTHER"


def inspect_network(grid, net_file):

    print("\n" + "=" * 80)
    print(
        f"GRID {grid} "
        f"(expected agents: {GRID_AGENT_COUNTS[grid]})"
    )
    print("=" * 80)

    tree = ET.parse(net_file)
    root = tree.getroot()

    tl_logics = root.findall("tlLogic")

    unique_ids = sorted(
        set(tl.attrib["id"] for tl in tl_logics)
    )

    print(f"Network file: {net_file}")
    print(f"Traffic-light programs found: {len(tl_logics)}")
    print(f"Unique TLS IDs: {len(unique_ids)}")

    if len(unique_ids) == GRID_AGENT_COUNTS[grid]:
        print("TLS count matches expected agent count.")
    else:
        print(
            f"WARNING: expected {GRID_AGENT_COUNTS[grid]} "
            f"but found {len(unique_ids)}"
        )

    for tl in tl_logics:

        tls_id = tl.attrib.get("id")
        program_id = tl.attrib.get("programID", "unknown")
        logic_type = tl.attrib.get("type", "unknown")
        offset = tl.attrib.get("offset", "0")

        phases = tl.findall("phase")

        print("\n" + "-" * 80)
        print(f"TLS ID: {tls_id}")
        print(f"Program ID: {program_id}")
        print(f"Type: {logic_type}")
        print(f"Offset: {offset}")
        print(f"Phase count: {len(phases)}")

        total_cycle = 0.0

        for index, phase in enumerate(phases):

            duration = float(
                phase.attrib.get("duration", 0)
            )

            state = phase.attrib.get("state", "")

            min_dur = phase.attrib.get(
                "minDur",
                "-"
            )

            max_dur = phase.attrib.get(
                "maxDur",
                "-"
            )

            total_cycle += duration

            print(
                f"Phase {index:>2}: "
                f"{classify_phase(state):<8} "
                f"duration={duration:>5.1f}s "
                f"minDur={min_dur} "
                f"maxDur={max_dur}"
            )

            print(f"          state={state}")

        print(
            f"Nominal cycle duration: "
            f"{total_cycle:.1f} seconds"
        )


def main():

    print("\nTHESIS REVISION V2")
    print("SUMO TRAFFIC-LIGHT PROGRAM AUDIT")

    for grid, net_file in NETWORK_FILES.items():
        inspect_network(grid, net_file)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("No network files were modified.")
    print("=" * 80)


if __name__ == "__main__":
    main()

