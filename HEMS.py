import pandas as pd
import numpy as np
import env
import dqn
from tqdm import tqdm
import matplotlib.pyplot as plt


# HEMS = Home Energy Management System
# This class manages the training and testing of the RL agent
class HEMS:

    def __init__(self, battery=20, max_en=1.5, eff=0.9, price_coefs=[2,1], data_path='data/rtp.csv'):

        # basic system parameters
        self.battery = battery
        self.max_en = max_en
        self.eff = eff
        self.price_coefs = price_coefs

        # load electricity price dataset
        self.df = pd.read_csv(data_path)

        # RL agent will be created during training
        self.agent = None

        # epsilon controls exploration
        self.epsilon = 1

        print("Smart Home Energy System initialized")
        print("Battery capacity:", battery, "kWh")


    # ---------------------------------------------------
    # TRAINING FUNCTION
    # ---------------------------------------------------
    def train(self, episodes=100, epsilon_decay=0.98, steps=500):

        # create environment
        environment = env.Env(
            self.df,
            self.battery,
            self.max_en,
            self.eff,
            self.price_coefs,
            2,
            steps
        )

        environment.reset(0)

        # create DQN agent
        state_size = environment.next_observation_normalized().shape[0]

        self.agent = dqn.DQN(state_size, 4)

        print("Training started...")

        for episode in tqdm(range(episodes)):

            # get initial system state
            state = environment.next_observation_normalized()

            for step in range(steps):

                # choose action using epsilon-greedy strategy
                action = self.agent.choose_action(state, self.epsilon)

                obs, reward, done, data = environment.step(action)

                next_state = environment.next_observation_normalized()

                # store experience for replay learning
                self.agent.store_transition(state, action, reward, next_state)

                # start training once memory has enough samples
                if self.agent.memory_counter > 2000:
                    self.agent.learn()

                state = next_state

                if done:
                    break

            # gradually reduce exploration
            self.epsilon *= epsilon_decay

        print("Training finished")


    # ---------------------------------------------------
    # TEST THE TRAINED MODEL
    # ---------------------------------------------------
    def test(self, steps=500):

        environment = env.Env(
            self.df,
            self.battery,
            self.max_en,
            self.eff,
            self.price_coefs,
            2,
            steps
        )

        environment.reset(0)

        state = environment.next_observation_normalized()

        rewards = []
        battery_levels = []
        prices = []
        energy_flow_data = []

        total_reward = 0

        for step in range(steps):

            # choose action using trained policy
            action = self.agent.choose_action(state, self.epsilon)

            obs, reward, done, flow_data = environment.step(action)

            next_state = environment.next_observation_normalized()

            rewards.append(reward)

            # store some system variables for visualization
            battery_levels.append(obs[1,-1])
            prices.append(obs[5,-1])
            energy_flow_data.append(flow_data)

            total_reward += reward

            state = next_state

            if done:
                break

        print("Total reward from test run:", total_reward)

        # visualize results
        self.plot_results(rewards, battery_levels, prices, energy_flow_data)


    # ---------------------------------------------------
    # PLOT SYSTEM BEHAVIOR
    # ---------------------------------------------------
    def plot_results(self, rewards, battery_levels, prices, energy_flow):

        time_steps = range(len(rewards))

        plt.figure(figsize=(12,8))

        # cumulative reward (cost)
        plt.subplot(3,1,1)
        plt.plot(time_steps, np.cumsum(rewards))
        plt.title("Cumulative Energy Cost")
        plt.ylabel("Cost")

        # battery charge level
        plt.subplot(3,1,2)
        plt.plot(time_steps, battery_levels)
        plt.title("Battery Level Over Time")
        plt.ylabel("Energy (kWh)")

        # electricity market price
        plt.subplot(3,1,3)
        plt.plot(time_steps, prices)
        plt.title("Electricity Market Price")
        plt.ylabel("Price")

        plt.tight_layout()
        plt.show()

        # also visualize energy flow
        self.plot_energy_flow(energy_flow)


    # ---------------------------------------------------
    # ENERGY FLOW VISUALIZATION
    # ---------------------------------------------------
    def plot_energy_flow(self, energy_flow):

        energy_flow = np.array(energy_flow).T

        flow_labels = [
            "Battery → Grid",
            "Grid → Battery",
            "Grid → Home",
            "Grid → Storage",
            "Battery → Home"
        ]

        plt.figure(figsize=(10,6))

        for i in range(min(len(energy_flow), len(flow_labels))):
            plt.plot(energy_flow[i], label=flow_labels[i])

        plt.title("Energy Flow Inside the Smart Home")
        plt.xlabel("Time Step")
        plt.ylabel("Energy (kWh)")
        plt.legend()

        plt.show()
