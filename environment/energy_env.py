import numpy as np

class EnergyEnvironment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.total_steps = len(data)

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        row = self.data[self.current_step]
        return np.array([row[0], row[1], row[2]])

    def step(self, action):
        time, load, price = self.data[self.current_step]

        adjusted_load = load * (0.8 + 0.1 * action)
        cost = adjusted_load * price
        reward = -cost

        self.current_step += 1
        done = self.current_step >= self.total_steps

        next_state = self._get_state() if not done else None

        return next_state, reward, done
