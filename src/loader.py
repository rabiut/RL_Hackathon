import numpy as np
from agent_dqn_tuned import LunarLanderAgent
import gymnasium as gym
import torch
class LocalEvaluation:
    def __init__(self, agent, num_episodes=100):
        self.agent = agent
        self.num_episodes = num_episodes
        self.env = gym.make('LunarLander-v3')
        self.state = None
        self.total_reward = []
        self.episodes_completed = 0


    def start_evaluation(self):
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]  # Extract the state from the tuple
        self.total_reward = []
        self.episodes_completed = 0

        self.run_episode()

    def run_episode(self):
        while self.episodes_completed < self.num_episodes:
            state_tensor = torch.FloatTensor(self.state).unsqueeze(0)  # Convert to tensor and add batch dimension
            action = self.agent.select_action(state_tensor)
            next_state, reward, done, _, _ = self.env.step(action)  # Use .item() to get the scalar value
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Extract the state from the tuple
            self.state = next_state
            self.total_reward.append(reward)


            # Print progress message
            print(f"Episode: {self.episodes_completed + 1}/{self.num_episodes}, Reward: {reward:.2f}", end='\r')

            if done:
                self.episodes_completed += 1
                print(f"\nEpisode {self.episodes_completed} completed. Total Reward: {sum(self.total_reward):.2f}, Avg. Reward: {sum(self.total_reward)/100:.2f}")

                if self.episodes_completed < self.num_episodes:
                    self.state = self.env.reset()
                    if isinstance(self.state, tuple):
                        self.state = self.state[0]  # Extract the state from the tuple

        print(f"\nEvaluation completed.")
        self.env.close()
if __name__ == '__main__':
    # Initialize the agent
    agent = LunarLanderAgent()

    # Load the trained model
    agent_model_file = 'Nov27_275reward_model_dqn_tuned.pkl'
    print("Loading Agent...")
    agent.load_agent(agent_model_file)

    # Run local evaluation
    print("Starting local evaluation...")
    evaluator = LocalEvaluation(agent)
    evaluator.start_evaluation()