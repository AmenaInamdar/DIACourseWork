from sumo_env import SUMOEnv
from dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import torch

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Initialize environment and agent
env = SUMOEnv(gui=False)
state_dim = 3
action_dim = env.action_space
agent = DQNAgent(state_dim, action_dim)

EPISODES = 1000
rewards = []

# Setup live plot
plt.ion()
fig, ax = plt.subplots()
line_total, = ax.plot([], [], label="Total Reward", color='blue')
line_avg, = ax.plot([], [], label="Moving Average (10)", color='orange', linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_title("DQN Training - Live Rewards")
ax.legend()
ax.grid(True)

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
        step_count += 1

    rewards.append(total_reward)

    # Update plot
    line_total.set_data(range(len(rewards)), rewards)
    if len(rewards) >= 10:
        smoothed = moving_average(rewards)
        line_avg.set_data(range(9, len(smoothed)+9), smoothed)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

    print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Steps: {step_count}")

env.close()

# Final plot and model saving
plt.ioff()
plt.savefig("training_rewards.png")
plt.show()
torch.save(agent.model.state_dict(), "dqn_model.pth")
