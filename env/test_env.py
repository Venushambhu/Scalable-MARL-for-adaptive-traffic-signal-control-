import sys
sys.path.append("..")
from traffic_env import TrafficSignalEnv

env = TrafficSignalEnv(
    net_file="../network/grid2x2.net.xml",
    route_file="../routes/low_demand.rou.xml",
    num_seconds=120,
    seed=1
)

obs, info = env.reset()
print("Initial obs for B1:", obs["B1"])

for step in range(10):
    actions = {tls: 0 for tls in env.tls_ids}  # do-nothing policy for this test
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f"step {step}: rewards = {rewards}")

env.close()
print("Environment smoke test successful")