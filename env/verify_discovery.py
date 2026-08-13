import sys
sys.path.append(".")
from traffic_env import discover_tls_and_neighbours

tls_ids, neighbours = discover_tls_and_neighbours("../network/grid2x2.net.xml")
print("Discovered TLS IDs:", tls_ids)
print("Expected:           ['B1', 'B2', 'C1', 'C2']")
print()
for tls in tls_ids:
    print(f"  {tls} neighbours: {neighbours[tls]}")
print()
print("Match expected 4 agents, 2 neighbours each:", 
      tls_ids == ["B1", "B2", "C1", "C2"] and all(len(neighbours[t]) == 2 for t in tls_ids))
