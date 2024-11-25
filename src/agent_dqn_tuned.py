# agent_template.py
import numpy as np
import gymnasium as gym
# from state_discretizer import StateDiscretizer
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import random
import matplotlib.pyplot as plt

class QNN(nn.Module):# 8 and 4 represent the number of dimensions in the state space and number of actions, respectively
    def __init__(self, hidden_layer_size):
        super(QNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 4)
        )

    def forward(self, x):
        return self.layers(x)

class Replay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
       
    def push(self, batch):
        self.memory.append(batch)
        if len(self.memory) > self.capacity:
            del self.memory[0]    

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class LunarLanderAgent:
    def __init__(self):
        self.env = gym.vector.SyncVectorEnv([lambda: gym.make("LunarLander-v3") for _ in range(4)])
        
        self.batch_size = 64
        self.alpha = 0.001 # learning rate
        self.gamma = 0.99 # discount factor
        self.tau = 0.1 # for soft update, prevents drastic changes and improves stability  
        self.epsilon = 1.0 # initial exploration rate
        self.epsilon_min = 0.01 # minimum exploration rate
        self.epsilon_decay = 0.1 #not used rn
        self.hidden_layer_size=64

        self.q_local = QNN(self.hidden_layer_size)
        self.q_target = QNN(self.hidden_layer_size)

        self.mse = nn.MSELoss()
        self.optim = optim.Adam(self.q_local.parameters(), lr=self.alpha)
        self.replay = Replay(50000)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.q_local(state).argmax(dim=1)
        else:
            return torch.randint(0, 4, (state.shape[0],))
        
    def train(self, num_episodes):
        scores = []
        best_average = -float('inf')  # Track the best average score

        for i_episode in range(num_episodes):
            self.epsilon = max(self.epsilon_min, 1.0 - (1.0 - self.epsilon_min) * (i_episode / 200))
            score = self.run_episode()
            scores.append(score)

            self.soft_update()

            print(f"Episode {i_episode}, Score: {scores[-1]:.2f}, Epsilon: {self.epsilon:.2f}")

            # Check rolling average for the last 100 episodes
            if len(scores) >= 100:
                rolling_avg = np.mean(scores[-100:])
                if rolling_avg > best_average:  # If new best average is achieved
                    best_average = rolling_avg
                    self.save_agent(f'best_model.pth')  # Autosave the best model
                    print(f"New best average score: {best_average:.2f}")
        print(f" batch_size = {self.batch_size}, \
            alpha = {self.alpha}, \
            gamma = {self.gamma}, \
            tau = {self.tau}, \
            epsilon_min = {self.epsilon_min}, \
            epsilon_decay = {self.epsilon_decay}, \
            hidden_layer_size={self.hidden_layer_size}")
        return scores

    def soft_update(self):
        for target_param, local_param in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update(self, state, action, reward, next_state, done):

        for i in range(self.env.num_envs):
            self.replay.push((
                state[i].unsqueeze(0),
                torch.tensor([[action[i]]], dtype=torch.long),
                reward[i].unsqueeze(0),
                next_state[i].unsqueeze(0),
                done[i].unsqueeze(0)
            ))

        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        batch = self.Transition(*zip(*batch))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)

        Q_expected = self.q_local(states).gather(1, actions)

        Q_targets_next = self.q_target(next_states).max(1)[0].detach()
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        loss = self.mse(Q_expected, Q_targets.unsqueeze(1))
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def run_episode(self):
        states,_  = self.env.reset()
        states = torch.tensor(states, dtype=torch.float32)
        done = np.zeros(self.env.num_envs, dtype=bool)
        total_rewards = np.zeros(self.env.num_envs)

        while not np.all(done):
            actions = self.select_action(states).numpy()

            next_states, rewards, terminated, truncated, _ = self.env.step(actions)

            dones = np.logical_or(terminated, truncated)

            next_states = torch.tensor(next_states, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            self.update(states, actions, rewards, next_states, dones)

            states = next_states
            total_rewards += rewards.numpy() * (~done)  # Accumulate rewards for active envs
            done = np.logical_or(done, dones.numpy())
        
        return np.mean(total_rewards)

    def test(self, num_episodes = 100):
        total_rewards = []

        self.epsilon = 0.0  # Disable exploration during testing

        for _ in range(num_episodes):
            states, _ = self.env.reset()
            states = torch.tensor(states, dtype=torch.float32)
            done = np.zeros(self.env.num_envs, dtype=bool)
            episode_rewards = np.zeros(self.env.num_envs)

            while not np.all(done):
                with torch.no_grad():
                    actions = self.q_local(states).argmax(dim=1).numpy()

                next_states, rewards, terminated, truncated, _ = self.env.step(actions)
                dones = np.logical_or(terminated, truncated)

                next_states = torch.tensor(next_states, dtype=torch.float32)
                episode_rewards += rewards * (~done)
                states = next_states
                done = np.logical_or(done, dones)

            total_rewards.append(np.mean(episode_rewards))

        average_reward = np.mean(total_rewards)
        print(f"100-Average test reward: {average_reward}")
        return average_reward

    def save_agent(self, file_name):
        torch.save({
            'q_local_state_dict': self.q_local.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'epsilon': self.epsilon
        }, file_name)
    def load_agent(self, file_name):
        checkpoint = torch.load(file_name)
        self.q_local.load_state_dict(checkpoint['q_local_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {file_name}.")

    def plot(self, scores):
        print(f'Reward Total:{sum(scores)}, Reward Avg.: {np.mean(scores)}')
        plt.figure(figsize=(10, 6))
        plt.plot(scores, label='Episode Reward')
        moving_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
        plt.plot(moving_avg, label=f'100-Episode Moving Avg.', color='red')
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    agent = LunarLanderAgent()
    agent_model_file = 'model.pkl'  # Set the model file name

    num_training_episodes = 1000  # Define the number of training episodes
    print("Training...")
    rewards = agent.train(num_training_episodes)
    print("Training done.")

    # Save the trained model
    agent.save_agent(agent_model_file)
    print("Model saved!")

    print("Testing...")
    agent.test()
    print("Testing done.")

    agent.plot(rewards)