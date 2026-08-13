import sys
sys.path.append(".")
from train_dqn_marl import train
from stable_baselines3 import DQN
import torch

# Quick 600-step run just to exercise a couple of target updates (interval=250)
train("low", total_steps=1000, seed=1, save_dir="../models_verify")

model = DQN.load("../models_verify/dqn_B1_low.zip")
checksum = sum(p.sum().item() for p in model.q_net_target.parameters())
print(f"Final target network checksum: {checksum:.6f}")
print("(if this were still broken, the target net would equal its random init forever)")