import pandas as pd
import numpy as np
import env
import dqn
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class HEMS:

    def __init__(self, battery=20, max_en=1.5, eff=0.9,
                 price_coefs=None, data_path='data/rtp.csv', load=False, path=None):

        self.memory_capacity = 2000
        self.agent = None

        self.battery = battery
        self.max_en = max_en
        self.eff = eff
        self.price_coefs = price_coefs if price_coefs is not None else [1.0, 1.0]
        self.df = pd.read_csv(data_path)

        # Epsilon starts at 1.0 (full exploration) and decays toward 0.1
        self.epsilon = 1.0

        if load and path:
            dummy_env = env.Env(self.df, self.battery, self.max_en, self.eff, self.price_coefs, n_steps=10)
            self.agent = dqn.DQN(dummy_env.n_features, dummy_env.n_actions)
            try:
                self.agent.load_model(path)
                print(f"Successfully loaded pre-trained model from {path}")
                self.epsilon = 0.1 # Model is trained, so we minimize random exploration
            except Exception as e:
                print(f"Could not load model from {path}. Starting fresh. Error: {e}")

        import threading
        self.training_lock = threading.Lock()

        print("\nHEMS Initialized")
        print(f"Battery capacity : {battery} kWh")
        print(f"Max energy/step  : {max_en} kWh")

    # ------------------------------------------------------------------
    def train(self, n_episodes=100, epsilon_decay=0.97, steps=500):
        with self.training_lock:
            environment = env.Env(
                self.df,
                self.battery,
                self.max_en,
                self.eff,
                self.price_coefs,
                n_steps=steps,
            )

            if self.agent is None:
                self.agent = dqn.DQN(
                    environment.n_features,
                    environment.n_actions
                )

            # If user asks for very few episodes, we must decay epsilon faster!
            # We want epsilon to reach 0.1 by the end of training.
            if n_episodes < 50:
                epsilon_decay = (0.1)**(1.0 / max(1, n_episodes * 0.8))

            print(f"\nTraining started — {n_episodes} episodes, {steps} steps each")
            print(f"Starting epsilon : {self.epsilon:.2f}")

            epsilon = self.epsilon

            episode_rewards  = []
            episode_savings  = []

            pbar = tqdm(range(n_episodes))
            for episode in pbar:

                state, _ = environment.reset(seed=episode)
                ep_reward   = 0.0
                ep_loss     = 0.0
                ep_savings  = 0.0   # track raw cost savings (unscaled)
                ep_cost     = 0.0
                learn_steps = 0
                action_counts = {0: 0, 1: 0, 2: 0}

                for step in range(steps):

                    action = self.agent.choose_action(state, epsilon)
                    action_counts[action] += 1

                    next_state, reward, terminated, truncated, info = environment.step(action)

                    self.agent.store_transition(state, action, reward, next_state)
                    ep_reward  += reward
                    ep_savings += info.get('cost_savings', 0.0)
                    ep_cost    += info.get('cost', 0.0)

                    if self.agent.memory_counter > dqn.BATCH_SIZE and step % 4 == 0:
                        loss = self.agent.learn()
                        if loss is not None and loss != 0.0:
                            ep_loss    += loss
                            learn_steps += 1

                    state = next_state

                    if terminated or truncated:
                        break

                # Decay epsilon
                epsilon = max(0.1, epsilon * epsilon_decay)

                ep_baseline = ep_cost + ep_savings
                ep_savings_pct = (ep_savings / ep_baseline * 100) if ep_baseline > 0 else 0.0

                episode_rewards.append(ep_reward)
                episode_savings.append(ep_savings_pct)

                avg_loss = ep_loss / learn_steps if learn_steps > 0 else 0.0
                acts = f"{action_counts[0]}/{action_counts[1]}/{action_counts[2]}"
                pbar.set_postfix({
                    'Rwd':     f"{ep_reward:.3f}",
                    'Sav%':    f"{ep_savings_pct:.2f}",
                    'Cost':    f"{ep_cost:.3f}",
                    'Loss':    f"{avg_loss:.5f}",
                    'Eps':     f"{epsilon:.3f}",
                    'C/D/I':   acts,
                })

            self.epsilon = epsilon
            self.agent.save_model("dqn_model.pth")

            print("\nTraining finished")
            print(f"Final epsilon          : {epsilon:.3f}")
            print(f"Final avg reward/ep    : {np.mean(episode_rewards[-10:]):.4f}")
            print(f"Final avg savings/step : {np.mean(episode_savings[-10:]):.6f}")

            return episode_rewards, episode_savings

    # ------------------------------------------------------------------
    def test(self, steps=500):

        environment = env.Env(
            self.df,
            self.battery,
            self.max_en,
            self.eff,
            self.price_coefs,
            n_steps=steps,
            test=True,
        )

        if self.agent is None:
            self.agent = dqn.DQN(
                environment.n_features,
                environment.n_actions
            )

        import os
        if os.path.exists("dqn_model.pth"):
            self.agent.load_model("dqn_model.pth")
            print("Loaded trained model for testing.")
        else:
            print("No saved model found! Using random weights.")

        state, _ = environment.reset(seed=0)

        rewards        = []
        battery_levels = []
        prices         = []
        total_cost     = 0.0
        total_baseline = 0.0
        total_savings  = 0.0
        total_solar_charge = 0.0
        total_sold_energy  = 0.0

        print(f"\n{'Step':>5} | {'Action':>10} | {'Price':>8} | {'Demand':>7} | "
              f"{'Batt':>6} | {'Dischg':>7} | {'ChgDrw':>7} | {'GridUse':>8} | "
              f"{'BaseCost':>9} | {'ActCost':>8} | {'Savings':>8} | {'Reward':>8}")
        print("-" * 120)

        for step in range(steps):

            action = self.agent.choose_action(state, epsilon=0.0)

            next_state, reward, terminated, truncated, info = environment.step(action)

            action_name = {0: "Charge", 1: "Discharge", 2: "Idle"}[action]

            price       = info['price']
            demand      = info.get('demand', 0.0)
            battery     = info['battery_level']
            discharge   = info.get('discharge', 0.0)
            charge_draw = info.get('charge_draw', 0.0)
            grid_usage  = info.get('grid_usage', 0.0)
            base_cost   = info.get('base_cost', 0.0)
            act_cost    = info.get('cost', 0.0)
            savings     = info.get('cost_savings', 0.0)

            print(f"{step:>5} | {action_name:>10} | {price:>8.5f} | {demand:>7.4f} | "
                  f"{battery:>6.2f} | {discharge:>7.4f} | {charge_draw:>7.4f} | {grid_usage:>8.4f} | "
                  f"{base_cost:>9.5f} | {act_cost:>8.5f} | {savings:>8.5f} | {reward:>8.5f}")

            rewards.append(reward)
            battery_levels.append(battery)
            prices.append(price)
            total_cost     += act_cost
            total_baseline += base_cost
            total_savings  += savings
            total_solar_charge += info.get('solar_charge', 0.0)
            total_sold_energy  += info.get('sold_energy', 0.0)

            state = next_state
            if terminated or truncated:
                break

        print("-" * 120)
        
        # Add residual battery value to accurately reflect economic position
        # Value inventory at the maximum price since the agent could sell it later
        max_price = 0.15
        residual_value = battery_levels[-1] * max_price
        total_cost -= residual_value
        total_savings += residual_value

        print(f"\nResidual battery val: {residual_value:.4f}")
        print(f"Total baseline cost : {total_baseline:.4f}")
        print(f"Total actual cost   : {total_cost:.4f}")
        print(f"Total savings       : {total_savings:.4f}")
        pct = (total_savings / total_baseline * 100) if total_baseline > 0 else 0.0
        print(f"Savings %           : {pct:.2f}%")

        return {
            "cost":          round(total_cost, 4),
            "baseline_cost": round(total_baseline, 4),
            "savings":       round(total_savings, 4),
            "battery":       int((battery_levels[-1] / self.battery) * 100),
            "solar_charge":  round(total_solar_charge, 4),
            "sold_energy":   round(total_sold_energy, 4),
            "rewards":       rewards,
            "battery_levels": battery_levels,
            "prices":        prices,
        }

    # ------------------------------------------------------------------
    def save_graph(self, rewards, battery, prices):
        steps = range(len(rewards))
        plt.figure(figsize=(10, 6))
        plt.plot(steps, rewards,  label="Reward (savings)")
        plt.plot(steps, battery,  label="Battery Level (kWh)")
        plt.plot(steps, prices,   label="Price (SMP)")
        plt.legend()
        plt.title("Energy Performance")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.savefig("static/output.png")
        plt.close()