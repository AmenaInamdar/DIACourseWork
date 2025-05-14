import torch
import numpy as np
import csv
import os
from sumo_env import SUMOEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = "dqn_model.pth"
LOG_PATH = "evaluation_log.csv"
EPISODES = 1

# Initialize
env = SUMOEnv(gui=True)
state_dim = 3
action_dim = 5
agent = DQNAgent(state_dim, action_dim)
agent.epsilon = 0.0  # Evaluation mode

# Load trained model
agent.model.load_state_dict(torch.load(MODEL_PATH))
agent.model.eval()

all_rewards = []

# Prepare CSV logging
with open(LOG_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Step", "Speed", "Lane", "Distance", "Action", "Reward"])

    # Evaluation loop
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            speed = round(state[0] * 30, 2)
            lane = int(state[1] * 2)
            dist = round(state[2] * 100, 2)

            writer.writerow([episode + 1, step + 1, speed, lane, dist, action, round(reward, 2)])

            state = next_state
            total_reward += reward
            step += 1

        all_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {step}")

env.close()

# Plot reward graph
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPISODES + 1), all_rewards, marker='o')
plt.title("DQN Evaluation - Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_result.png")
plt.show()
