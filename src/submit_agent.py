# submit_agent.py

import numpy as np
import webbrowser
import socketio
import sys
from ppo_agent import LunarLanderAgent
import torch

# Configuration
SERVER_URL = 'http://srv-cad.ece.mcmaster.ca:65000'  # Server URL

# Define team credentials directly in the script
TEAM_NAME = 'NextUp'  # Replace with your actual team name
TEAM_SECRET_CODE = '1b953b72-2ae6-4873-ba4f-d540909fe8eb'  # Replace with your actual secret code

class AgentSubmission:
    def __init__(self, agent):
        self.agent = agent
        self.sio = socketio.Client()
        self.target_avg_score = 285.305

    def submit(self):
        """
        Submit the agent to the server for evaluation using WebSocket connection.
        """
        try:
            # Define event handlers
            @self.sio.event
            def connect():
                print("Connected to the server.")

                # Start the submission by sending team credentials
                self.sio.emit('start_submission', {
                    'team_name': TEAM_NAME,
                    'secret_code': TEAM_SECRET_CODE
                })

            @self.sio.event
            def submission_started(data):
                print("Submission started.")
                self.handle_submission_started(data)

            @self.sio.event
            def action_response(data):
                self.handle_action_response(data)

            @self.sio.event
            def submission_completed(data):
                print(f"Submission completed. Average Reward: {data['average_reward']}")
                self.sio.disconnect()
                # Open the leaderboard after submission is complete
                self.open_leaderboard()

            @self.sio.event
            def submission_error(data):
                print(f"Error: {data['error']}")
                self.sio.disconnect()

            @self.sio.event
            def disconnect():
                print("Disconnected from the server.")

            # Connect to the server
            self.sio.connect(SERVER_URL)
            # Wait for the submission to complete
            self.sio.wait()

        except Exception as e:
            print(f"An error occurred: {e}")

    def handle_submission_started(self, data):
        """
        Handle the submission_started event.
        """
        self.num_submission_episodes = data['num_submission_episodes']
        self.state = np.array(data['state'])
        self.total_reward = 0
        self.episodes_completed = 0
        self.timestep = 0

        # Start the submission loop
        self.send_next_action()

    def handle_action_response(self, data):
        """
        Handle the action_response event.
        """
        self.state = np.array(data['state'])
        reward = data['reward']
        done = data['done']
        episodes_completed = data['episodes_completed']
        self.total_reward += reward
        self.timestep += 1

        # Print progress message
        print(f"Episode: {episodes_completed}/{self.num_submission_episodes}, Timestep: {self.timestep}, Total Reward: {self.total_reward}", end='\r')
        if done:
            print(f", AVG: {self.total_reward / (episodes_completed+1)}")
            avg_reward = self.total_reward / (episodes_completed)
            self.episodes_completed = episodes_completed
            self.timestep = 0  # Reset timestep for the next episode
            if episodes_completed == 95:
                if avg_reward > self.target_avg_score:
                    print(f"Target met! Avg Reward: {avg_reward}. Proceeding as usual.")
                else:
                    print(f"Target not met. Restarting submission. Avg Reward: {avg_reward}")
                    sys.exit()       

        # Send the next action
        self.send_next_action()

    # def send_next_action(self):
    #     """
    #     Select the next action and send it to the server.
    #     """
    #     if self.episodes_completed >= self.num_submission_episodes:
    #         # All episodes completed
    #         return

    #     action = self.agent.select_action(self.state)
    #     # Ensure the action is a native Python type
    #     if isinstance(action, np.generic):
    #         action_to_send = action.item()
    #     else:
    #         action_to_send = action
    #     self.sio.emit('take_action', {'action': action_to_send})
    def send_next_action(self):
        """
        Select the next action and send it to the server.
        """
        if self.episodes_completed >= self.num_submission_episodes:
            # All episodes completed
            return

        # Get the action from the agent
        action, _ = self.agent.select_action(self.state)

        # Ensure the action is a native Python type
        if isinstance(action, torch.Tensor):
            action_to_send = action.item()
        elif isinstance(action, np.generic):
            action_to_send = action.item()
        else:
            action_to_send = int(action)

        # Debug: Print action type to ensure it is correct
        # print(f"Action to send (type: {type(action_to_send).__name__}): {action_to_send}")

        # Emit the action to the server
        self.sio.emit('take_action', {'action': action_to_send})

    def open_leaderboard(self):
        """
        Open the leaderboard in the default web browser.
        """
        leaderboard_url = f'{SERVER_URL}/leaderboard'
        try:
            # Open the leaderboard URL in the default web browser
            webbrowser.open(leaderboard_url)
            print(f"Leaderboard opened in the default web browser. If it didn't open automatically, please visit {leaderboard_url}")
        except Exception as e:
            print(f"Failed to open leaderboard: {e}")

if __name__ == '__main__':
    if not TEAM_NAME or not TEAM_SECRET_CODE:
        print("Error: TEAM_NAME and TEAM_SECRET_CODE must be set in the script.")
        exit(1)

    agent = LunarLanderAgent()  # Ensure this is the correct class of your trained agent

    agent_model_file = 'lunar_lander_ppo_best.pth'  # Set the trained agent's model file name

    # Load the trained model
    print("loading Agent...")
    agent.load_agent(agent_model_file)
    
    # Submit the solution
    print("Submitting the solution...")
    submission = AgentSubmission(agent)
    submission.submit()