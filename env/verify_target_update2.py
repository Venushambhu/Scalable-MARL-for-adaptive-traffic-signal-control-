import sys
sys.path.append(".")
from stable_baselines3 import DQN
from train_dqn_marl import _DummySingleAgentEnv

trained = DQN.load("../models_verify/dqn_B1_low.zip")
trained_checksum = sum(p.sum().item() for p in trained.q_net_target.parameters())

fresh = DQN("MlpPolicy", _DummySingleAgentEnv(), seed=1, verbose=0)
fresh_checksum = sum(p.sum().item() for p in fresh.q_net_target.parameters())

print(f"Trained model's target-net checksum: {trained_checksum:.6f}")
print(f"Fresh (never-trained) target-net checksum: {fresh_checksum:.6f}")
print(f"Different (fix confirmed working): {abs(trained_checksum - fresh_checksum) > 1e-6}")