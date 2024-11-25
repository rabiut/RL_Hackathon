import gymnasium as gym
from state_discretizer import StateDiscretizer
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate the moving average
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

class LunarLanderAgent:
    def __init__(self):
        self.env = gym.make('LunarLander-v3')
        self.state_discretizer = StateDiscretizer(self.env)
        self.q_table = np.zeros((self.state_discretizer.iht_size, self.env.action_space.n))
        print(f"Q-table shape: {self.q_table.shape}")

        # Hyperparameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state):
        # Epsilon-greedy policy for training
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore: random action
        
        state_idx = sum(self.state_discretizer.discretize(state)) % self.q_table.shape[0]
        action = np.argmax(self.q_table[state_idx])  # Exploit: best action

        if not self.env.action_space.contains(action):
            print(f"Invalid action {action} detected, defaulting to random action.")
            action = self.env.action_space.sample()

        return action

    def train(self, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")

        return rewards

    def update(self, state, action, reward, next_state, done):
        state_idx = sum(self.state_discretizer.discretize(state)) % self.q_table.shape[0]
        next_state_idx = sum(self.state_discretizer.discretize(next_state)) % self.q_table.shape[0]
        best_next_action = np.argmax(self.q_table[next_state_idx])
        target = reward + self.gamma * self.q_table[next_state_idx, best_next_action] * (1 - done)
        self.q_table[state_idx, action] += self.alpha * (target - self.q_table[state_idx, action])

    def test(self, num_episodes=100):
        total_rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                state_idx = sum(self.state_discretizer.discretize(state)) % self.q_table.shape[0]
                action = np.argmax(self.q_table[state_idx])  # Always exploit

                if not self.env.action_space.contains(action):
                    print(f"Invalid action {action} during testing. Defaulting to random action.")
                    action = self.env.action_space.sample()

                state, reward, done, _, _ = self.env.step(action)
                total_reward += reward

            total_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}: Total Reward: {total_reward}")

        print(f"Average Test Reward: {np.mean(total_rewards)}")

    def save_agent(self, file_name):
        np.save(file_name, self.q_table)

    def load_agent(self, file_name):
        self.q_table = np.load(file_name)

if __name__ == '__main__':
    agent = LunarLanderAgent()
    agent_model_file = 'model.pkl'

    num_training_episodes = 1000
    print("Training the agent...")
    rewards = agent.train(num_training_episodes)
    print("Training completed.")

    print("Testing the agent...")
    agent.test(10)

    agent.save_agent(agent_model_file)
    print("Model saved.")

    smoothed_rewards = moving_average(rewards, window_size=100)
    plt.plot(smoothed_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Last 100 Episodes)')
    plt.title('Training Progress (Smoothed)')
    plt.show()
