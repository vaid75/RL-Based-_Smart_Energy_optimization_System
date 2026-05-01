import numpy as np

# Global price normalisation ceiling (slightly above observed max of 0.145)
PRICE_MAX = 0.15

# Opportunity-cost threshold: penalise idle when price is above this
# Set to median SMP (~0.062) so idle is penalised half the time
PRICE_OPP_THRESHOLD = 0.062


class Env:

    def __init__(self, df, full_battery_capacity=20, max_energy=1.5,
                 eff=0.9, price_coefs=None, n_days=2,
                 n_steps=500, low=0, high=30000, test=False):

        self.df = df
        self.full_battery_capacity = full_battery_capacity
        self.max_energy = max_energy
        self.eff = eff
        self.price_coefs = price_coefs if price_coefs is not None else [1.0, 1.0]

        self.n_steps = n_steps
        self.test = test

        self.current_step = 0
        self.history = []

        self.n_actions = 3   # 0=charge, 1=discharge, 2=idle
        self.n_features = 6

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if not hasattr(self, 'rng'):
            self.rng = np.random.default_rng(seed)
        elif seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.history = []

        if self.test:
            # Start at a position that has price variety (skip the flat first segment)
            # Use seed to pick a deterministic but non-zero position
            rng_test = np.random.default_rng(seed if seed is not None else 42)
            max_pos = max(1, len(self.df) - self.n_steps - 1)
            self.pos = int(rng_test.integers(max_pos // 4, max_pos))
        else:
            self.pos = self.rng.integers(0, max(1, len(self.df) - self.n_steps - 1))

        # Compute episode price stats for relative normalisation
        episode_prices = self.df['SMP'].iloc[self.pos: self.pos + self.n_steps].values
        self.ep_price_min = float(episode_prices.min())
        self.ep_price_max = float(max(episode_prices.max(), self.ep_price_min + 1e-6))
        self.ep_price_mid = (self.ep_price_min + self.ep_price_max) / 2.0

        # Charge threshold: lower 33% of episode price range
        self.price_low  = self.ep_price_min + 0.33 * (self.ep_price_max - self.ep_price_min)
        # Discharge threshold: upper 33% of episode price range
        self.price_high = self.ep_price_min + 0.67 * (self.ep_price_max - self.ep_price_min)

        # Start battery half-full so the agent can discover positive rewards from discharging immediately
        self.battery = self.full_battery_capacity / 2.0

        obs, info = self._get_obs_and_info()
        return obs, info

    # ------------------------------------------------------------------
    def _get_obs_and_info(self):
        if self.pos + self.current_step >= len(self.df):
            row = self.df.iloc[-1]
        else:
            row = self.df.iloc[self.pos + self.current_step]

        smp = row['SMP']

        # Feature 0: battery state of charge (0-1)
        battery_norm = self.battery / self.full_battery_capacity

        # Feature 1: absolute price normalised by global max
        price_norm = smp / PRICE_MAX

        # Feature 2: RELATIVE price in episode (0=cheapest, 1=most expensive)
        # This lets the agent learn "current price is high/low for THIS episode"
        price_rel = (smp - self.ep_price_min) / (self.ep_price_max - self.ep_price_min)

        gen  = row.get('Energy_Generation', 0)
        cons = row['Energy_Consumption']
        ev   = row.get('EV_Consumption', 0)
        net_load = max(0.0, cons + ev - gen)
        net_load_norm = min(net_load / 10.0, 1.0)   # demand normalised

        time_norm = row['Time_of_Day'] / 24.0

        obs = np.array([
            battery_norm,
            price_norm,
            price_rel,          # relative price (was hardcoded 0.0)
            net_load_norm,
            time_norm,
            0.0,
        ], dtype=np.float32)

        demand = cons + ev - gen
        baseline_cost = demand * smp

        unnorm_obs = [
            [self.current_step],
            [self.battery],
            [gen],
            [cons],
            [ev],
            [smp],
            [0],
            [row['Time_of_Day']],
        ]

        info = {
            'battery_level': self.battery,
            'price':         smp,
            'unnormalized_obs': unnorm_obs,
            'base_cost':     baseline_cost,
        }
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action):
        row = self.df.iloc[self.pos + self.current_step]

        cons  = row['Energy_Consumption']
        ev    = row.get('EV_Consumption', 0)
        gen   = row.get('Energy_Generation', 0)
        price = row['SMP']

        demand = cons + ev - gen
        baseline_cost = demand * price

        # ------------------------------------------------------------------
        # Battery physics
        # ------------------------------------------------------------------
        penalty     = 0.0
        charge_draw = 0.0
        discharge   = 0.0

        if action == 0:  # CHARGE from solar only
            room   = self.full_battery_capacity - self.battery
            # Limit charging to available solar generation
            charge = min(self.max_energy, room, gen)
            if charge <= 0:
                penalty = 0.1
            else:
                self.battery  += charge
                charge_draw    = charge

        elif action == 1:  # DISCHARGE
            avail    = self.battery
            discharge = min(self.max_energy, avail)
            if discharge <= 0:
                penalty = 0.1
            else:
                self.battery -= discharge

        # Clamp
        self.battery = max(0.0, min(self.battery, self.full_battery_capacity))

        # Grid usage calculation
        # Net load before battery is (cons + ev - gen)
        # Charging adds to the load (taking from gen), discharging reduces the load
        net_load = cons + ev - gen
        grid_usage = net_load + charge_draw - discharge

        actual_cost  = grid_usage * price
        cost_savings = baseline_cost - actual_cost

        # ------------------------------------------------------------------
        # Reward
        #
        # DISCHARGE: positive (grid usage reduced)
        # CHARGE: negative (grid usage increased — investing for future)
        # IDLE: 0 normally, but penalised if battery has charge at high price
        #       (opportunity cost — forces agent to discharge rather than hoard)
        # ------------------------------------------------------------------
        reward = cost_savings * 100.0  # Magnify the real financial impact to speed up learning
        reward -= penalty

        # Opportunity cost for idle when battery is charged and price is high
        if action == 2 and self.battery > 0 and price > self.price_high:
            # Scales with how much battery we're sitting on and how high the price is
            opp = (self.battery / self.full_battery_capacity) * (price - self.price_high) * 10.0
            reward -= opp

        # REWARD SHAPING FOR ULTRA-FAST LEARNING
        if action == 0 and price < self.price_low and charge_draw > 0:
            reward += 10.0
        elif action == 1 and price > self.price_high and discharge > 0:
            reward += 10.0
        elif action == 0 and price > self.price_high:
            reward -= 10.0
        elif action == 1 and price < self.price_low:
            reward -= 10.0

        # ------------------------------------------------------------------
        # History and step advancement
        # ------------------------------------------------------------------
        _, info_pre = self._get_obs_and_info()
        info_pre['cost']         = actual_cost
        info_pre['demand']       = demand
        info_pre['discharge']    = discharge
        info_pre['charge_draw']  = charge_draw
        info_pre['grid_usage']   = grid_usage
        info_pre['cost_savings'] = cost_savings
        info_pre['solar_charge'] = charge_draw  # Since charging only comes from solar
        info_pre['sold_energy']  = max(0.0, -grid_usage)

        unnorm_obs = info_pre['unnormalized_obs']
        self.history.append([r[-1] if isinstance(r, list) else r for r in unnorm_obs])

        self.current_step += 1
        terminated = self.current_step >= self.n_steps
        truncated  = False

        obs, final_info = self._get_obs_and_info()
        final_info['cost']         = actual_cost
        final_info['demand']       = demand
        final_info['discharge']    = discharge
        final_info['charge_draw']  = charge_draw
        final_info['grid_usage']   = grid_usage
        final_info['cost_savings'] = cost_savings
        final_info['solar_charge'] = charge_draw
        final_info['sold_energy']  = max(0.0, -grid_usage)

        return obs, reward, terminated, truncated, final_info