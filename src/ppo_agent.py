import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt
import os

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc_policy = nn.Linear(hidden_size, action_size)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value

class LunarLanderAgent:
    def __init__(self):
        self.env = gym.make('LunarLander-v3')
        self.state_size = self.env.observation_space.shape[0]  # Should be 8
        self.action_size = self.env.action_space.n  # Should be 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize policy network and optimizer
        self.policy = ActorCritic(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.lam = 0.95  # GAE lambda
        self.epochs = 10  # Number of epochs per update
        self.batch_size = 64  # Batch size for updates

        # For plotting
        self.training_rewards = []
        self.average_rewards = []
        self.testing_rewards = []
        # Initialize best test average
        self.best_test_average = float('-inf')  # Negative infinity

    def select_action(self, state):
        """Selects an action using the current policy."""
        state = torch.FloatTensor(state).to(self.device)
        policy_logits, _ = self.policy(state)
        action_probs = torch.softmax(policy_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    # def select_action(self, state):
    #     state = torch.FloatTensor(state).to(self.device)
    #     policy_logits, _ = self.policy(state)
    #     action_probs = torch.softmax(policy_logits, dim=-1)
    #     dist = Categorical(action_probs)
    #     action = dist.sample()
    #     log_prob = dist.log_prob(action)
    #     return action.item(), log_prob  # `.item()` ensures native type


    def select_action_deterministic(self, state):
        """Selects an action deterministically (used during testing)."""
        state = torch.FloatTensor(state).to(self.device)
        policy_logits, _ = self.policy(state)
        action_probs = torch.softmax(policy_logits, dim=-1)
        action = torch.argmax(action_probs).item()
        return action

    def train(self, max_episodes):
        scores_window = deque(maxlen=100)  # For tracking average over last 100 episodes
        consecutive_testing_successes = 0  # To track consecutive successful testing phases

        for i_episode in range(1, max_episodes + 1):
            state, _ = self.env.reset()
            done = False
            score = 0
            trajectory = []

            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                trajectory.append((state, action, reward, log_prob, done))
                state = next_state
                score += reward

            scores_window.append(score)
            self.training_rewards.append(score)
            average_score = np.mean(scores_window)
            self.average_rewards.append(average_score)

            # After collecting the trajectory, update the policy
            self.update(trajectory)

            print(f'\rEpisode {i_episode}\tAverage Score: {average_score:.2f}', end="")

            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {average_score:.2f}')
                # After every 100 episodes, run testing for 100 episodes
                testing_average = self.test(num_episodes=100)
                # Check for new best test average
                if testing_average > self.best_test_average:
                    self.best_test_average = testing_average
                    self.save_agent('lunar_lander_ppo_best.pth')
                    print(f'New best test average: {testing_average:.2f}. Model saved.')
                if testing_average >= 285.0:
                    consecutive_testing_successes += 1
                    print(f'Testing Success {consecutive_testing_successes}/2')
                    if consecutive_testing_successes >= 2:
                        print(f'\nTraining completed in {i_episode} episodes!')
                        break
                else:
                    consecutive_testing_successes = 0

        # Plot the training progress
        self.plot_training_progress()

    def update(self, trajectory):
        """Performs the PPO update."""
        # Convert lists to NumPy arrays before creating tensors
        states = torch.FloatTensor(np.array([t[0] for t in trajectory])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in trajectory])).to(self.device)
        rewards = [t[2] for t in trajectory]
        log_probs_old = torch.stack([t[3] for t in trajectory]).detach()
        dones = [t[4] for t in trajectory]

        # Compute returns and advantages using GAE-Lambda
        returns, advantages = self.compute_gae(rewards, dones, states)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages

        # Optimize policy for multiple epochs
        for _ in range(self.epochs):
            # Compute new log_probs and values
            policy_logits, values = self.policy(states)
            action_probs = torch.softmax(policy_logits, dim=-1)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Ratio for clipped surrogate objective
            ratios = torch.exp(log_probs - log_probs_old)

            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_gae(self, rewards, dones, states):
        """Computes Generalized Advantage Estimation (GAE)."""
        _, values = self.policy(states)
        values = values.squeeze().detach().cpu().numpy()
        values = np.append(values, 0)  # Append 0 for the value of the terminal state

        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages[step] = gae
        returns = advantages + values[:-1]
        return returns, advantages

    def test(self, num_episodes):
        """Evaluate the agent's performance over a number of episodes."""
        total_rewards = []
        for i in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.select_action_deterministic(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
            total_rewards.append(total_reward)
            print(f'\rTest Episode {i + 1}/{num_episodes}\tReward: {total_reward:.2f}', end="")

        average_reward = np.mean(total_rewards)
        print(f'\nAverage Test Reward over {num_episodes} episodes: {average_reward:.2f}')
        self.testing_rewards.append(average_reward)
        return average_reward

    def save_agent(self, file_name):
        """Save the trained model."""
        torch.save(self.policy.state_dict(), file_name)

    def load_agent(self, file_name):
        """Load a trained model."""
        if os.path.isfile(file_name):
            self.policy.load_state_dict(torch.load(file_name))
            print(f'Model loaded from {file_name}')
        else:
            print(f'File {file_name} does not exist.')

    def plot_training_progress(self):
        """Plots the training rewards and average rewards."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_rewards, label='Episode Reward')
        plt.plot(self.average_rewards, label='Average Reward (Over Last 100 Episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_progress.png')
        plt.show()

if __name__ == '__main__':
    agent = LunarLanderAgent()
    agent_model_file = 'lunar_lander_ppo.pth'
    num_training_episodes = 5000000  # Run for a long time as per your requirement
    agent.train(num_training_episodes)
    agent.save_agent(agent_model_file)

    # Conduct a final test over 100 episodes and print the average reward
    final_test_average = agent.test(num_episodes=100)
    print(f'\nFinal Test Average Reward over 100 episodes: {final_test_average:.2f}')
