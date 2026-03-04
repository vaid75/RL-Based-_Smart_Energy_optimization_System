import pandas as pd
import numpy as np
import env
import dqn
from tqdm import tqdm
import matplotlib.pyplot as plt

# HEMS = Home Energy Management System
class HEMS:

    def __init__(self, battery=20, max_en=1.5, eff=0.9, price_coefs=[2,1], data_path='data/rtp.csv'):

        self.battery = battery
        self.max_en = max_en
        self.eff = eff
        self.price_coefs = price_coefs
        self.df = pd.read_csv(data_path)
        self.agent = None
        self.epsilon = 1

        print("HEMS Initialized")
        print("Battery:", battery, "kWh")

    # ---------------------------
    # TRAIN MODEL
    # ---------------------------
    def train(self, episodes=100, epsilon_decay=0.98, steps=500):

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

        self.agent = dqn.DQN(
            environment.next_observation_normalized().shape[0],
            4
        )

        print("Training started...")

        for ep in tqdm(range(episodes)):

            state = environment.next_observation_normalized()

            for step in range(steps):

                action = self.agent.choose_action(state, self.epsilon)

                obs, reward, done, data = environment.step(action)

                next_state = environment.next_observation_normalized()

                self.agent.store_transition(state, action, reward, next_state)

                if self.agent.memory_counter > 2000:
                    self.agent.learn()

                state = next_state

                if done:
                    break

            self.epsilon *= epsilon_decay

        print("Training finished")

    # ---------------------------
    # TEST MODEL + VISUALIZATION
    # ---------------------------
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
        energy_flow = []

        total_reward = 0

        for step in range(steps):

            action = self.agent.choose_action(state, self.epsilon)

            obs, reward, done, data = environment.step(action)

            next_state = environment.next_observation_normalized()

            rewards.append(reward)
            battery_levels.append(obs[1,-1])
            prices.append(obs[5,-1])
            energy_flow.append(data)

            total_reward += reward

            state = next_state

            if done:
                break

        print("Total reward:", total_reward)

        self.plot_results(rewards, battery_levels, prices, energy_flow)


    # ---------------------------
    # PLOT RESULTS
    # ---------------------------
    def plot_results(self, rewards, battery, prices, energy_flow):

        steps = range(len(rewards))

        plt.figure(figsize=(12,8))

        # reward graph
        plt.subplot(3,1,1)
        plt.plot(steps, np.cumsum(rewards))
        plt.title("Cumulative Energy Cost")
        plt.ylabel("Cost")

        # battery graph
        plt.subplot(3,1,2)
        plt.plot(steps, battery)
        plt.title("Battery Level")
        plt.ylabel("kWh")

        # market price
        plt.subplot(3,1,3)
        plt.plot(steps, prices)
        plt.title("Electricity Market Price")
        plt.ylabel("Price")

        plt.tight_layout()
        plt.show()

        self.plot_energy_flow(energy_flow)


    # ---------------------------
    # ENERGY FLOW VISUALIZATION
    # ---------------------------
    def plot_energy_flow(self, energy_flow):

        energy_flow = np.array(energy_flow).T

        labels = [
            "Battery Out",
            "Battery In",
            "Grid to Home",
            "Grid to Battery",
            "Battery to Home"
        ]

        plt.figure(figsize=(10,6))

        for i in range(min(len(energy_flow), len(labels))):
            plt.plot(energy_flow[i], label=labels[i])

        plt.title("Energy Flow in Smart Home System")
        plt.xlabel("Time Step")
        plt.ylabel("Energy (kWh)")
        plt.legend()

        plt.show()
    
