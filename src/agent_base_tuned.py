import gymnasium as gym
from state_discretizer import StateDiscretizer
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random

# Function to calculate the moving average using deque
def moving_average(data, window_size=100):
    moving_avg = []
    window = deque(maxlen=window_size)
    for d in data:
        window.append(d)
        moving_avg.append(np.mean(window))
    return moving_avg

class LunarLanderAgent:
    def __init__(self):
        self.env = gym.make('LunarLander-v3')
        self.state_discretizer = StateDiscretizer(self.env)
        self.num_actions = self.env.action_space.n
        self.num_features = self.state_discretizer.iht_size

        # Initialize weights for each action
        self.weights = np.zeros((self.num_actions, self.num_features))

        # Hyperparameters
        self.alpha = 0.5 / self.state_discretizer.num_tilings  # Adjusted learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 # Adjusted decay rate

        # Set random seeds for reproducibility
        # np.random.seed(42)
        # random.seed(42)

    def get_active_features(self, state):
        active_tiles = self.state_discretizer.discretize(state)
        return active_tiles

    def q_values(self, state):
        active_features = self.get_active_features(state)
        q_vals = np.sum(self.weights[:, active_features], axis=1)
        return q_vals

    def select_action(self, state):
        # Epsilon-greedy policy for training
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore: random action
        else:
            q_vals = self.q_values(state)
            return np.argmax(q_vals)  # Exploit: best action

    def train(self, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        return rewards

    def update(self, state, action, reward, next_state, done):
        active_features = self.get_active_features(state)
        q_current = np.sum(self.weights[action, active_features])

        if done:
            td_target = reward
        else:
            q_next = self.q_values(next_state)
            td_target = reward + self.gamma * np.max(q_next)

        td_error = td_target - q_current
        self.weights[action, active_features] += self.alpha * td_error

    def test(self, num_episodes=100):
        total_rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                q_vals = self.q_values(state)
                action = np.argmax(q_vals)  # Always exploit
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}: Total Reward: {total_reward}")

        avg_test_reward = np.mean(total_rewards)
        print(f"Average Test Reward over {num_episodes} episodes: {avg_test_reward}")
        return avg_test_reward

    def save_agent(self, file_name):
        np.save(file_name, self.weights)

    def load_agent(self, file_name):
        self.weights = np.load(file_name)

if __name__ == '__main__':
    agent = LunarLanderAgent()
    agent_model_file = 'model_tuned_q_learning.npy'

    num_training_episodes = 15000*2  # Increased training episodes
    print("Training the agent...")
    rewards = agent.train(num_training_episodes)
    print("Training completed.")

    print("Testing the agent...")
    agent.test(100)  # Test over 100 episodes for reliability

    agent.save_agent(agent_model_file)
    print("Model saved.")

    smoothed_rewards = moving_average(rewards, window_size=100)
    plt.plot(smoothed_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Last 100 Episodes)')
    plt.title('Training Progress (Smoothed)')
    plt.show()
