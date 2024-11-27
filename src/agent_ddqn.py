import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DQNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Define the network architecture
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Hyperparameters
        self.buffer_size = int(1e5)  # Replay buffer size
        self.batch_size = 64         # Minibatch size
        self.gamma = 0.99            # Discount factor
        self.tau = 1e-3              # For soft update of target parameters
        self.lr = 5e-4               # Learning rate
        self.update_every = 4        # How often to update the network

        # Q-Network
        self.qnetwork_local = DQNNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DQNNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Epsilon for epsilon-greedy action selection
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
        
        # Decrease epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def act(self, state, eps=0.0):
        # Epsilon-greedy action selection
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # ------------------- Double DQN Update ------------------- #
        with torch.no_grad():
            next_actions = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = nn.SmoothL1Loss()(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = (state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

def train_dqn(agent, env, n_episodes=2000, max_t=1000):
    scores = []
    scores_window = deque(maxlen=100)  # Last 100 scores
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, agent.epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)
        scores.append(score)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {agent.epsilon:.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {agent.epsilon:.2f}')
    torch.save(agent.qnetwork_local.state_dict(), 'DQN_LunarLander_Final.pth')
    print("\nFinal model saved as 'DQN_LunarLander_Final.pth'.")
    return scores

def test_dqn(agent, env, n_episodes=100):
    total_rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, eps=0.0)  # Always exploit
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Total Reward: {total_reward}")
    avg_test_reward = np.mean(total_rewards)
    print(f"Average Test Reward over {n_episodes} episodes: {avg_test_reward}")

if __name__ == '__main__':
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    seed = 0
    agent = DQNAgent(state_size, action_size, seed)

    print("Training the DQN agent...")
    scores = train_dqn(agent, env, n_episodes=2000)
    print("Training completed.")

    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('Episode #')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.show()

    print("Testing the DQN agent...")
    agent.qnetwork_local.load_state_dict(torch.load('DQN_LunarLander_Final.pth'))
    test_dqn(agent, env, n_episodes=100)
