import numpy as np
import random
from typing import List, Tuple

class LBFEnv:
    def __init__(self, grid_size=4, n_agents=2, n_foods=2, agent_levels=[1, 2],
                 food_levels=[1, 2], max_steps=40, seed=None):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_foods = n_foods
        self.agent_levels = agent_levels
        self.food_levels = food_levels
        self.max_steps = max_steps
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()

    def reset(self):
        self.step_count = 0
        positions = set()
        self.agent_pos = []

        # Place agents
        for _ in range(self.n_agents):
            while True:
                pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                if pos not in positions:
                    positions.add(pos)
                    self.agent_pos.append(pos)
                    break

        # Place food
        self.food_pos = []
        self.food_exists = [True] * self.n_foods
        for i in range(self.n_foods):
            while True:
                pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                if pos not in positions and all(pos != self.food_pos[j] for j in range(i)):
                    positions.add(pos)
                    self.food_pos.append(pos)
                    break
        return self._get_obs()

    def _get_obs(self):
        """Enhanced observation space with relative positions"""
        obs_list = []
        for i in range(self.n_agents):
            obs = []

            # Own position (normalized)
            own_x, own_y = self.agent_pos[i]
            obs.extend([own_x / (self.grid_size - 1), own_y / (self.grid_size - 1)])

            # Active food positions (relative to agent) and levels
            active_foods = []
            for j in range(self.n_foods):
                if self.food_exists[j]:
                    fx, fy = self.food_pos[j]
                    rel_x = (fx - own_x) / (self.grid_size - 1)
                    rel_y = (fy - own_y) / (self.grid_size - 1)
                    dist = (abs(fx - own_x) + abs(fy - own_y)) / (2 * (self.grid_size - 1))
                    active_foods.append([
                        rel_x, rel_y, dist,
                        self.food_levels[j] / max(self.food_levels)
                    ])

            # Pad to consistent size
            while len(active_foods) < self.n_foods:
                active_foods.append([-2.0, -2.0, 1.0, 0.0])

            for food_info in active_foods:
                obs.extend(food_info)

            # Other agent positions (relative) and levels
            for j in range(self.n_agents):
                if i != j:
                    ax, ay = self.agent_pos[j]
                    rel_x = (ax - own_x) / (self.grid_size - 1)
                    rel_y = (ay - own_y) / (self.grid_size - 1)
                    obs.extend([rel_x, rel_y, self.agent_levels[j] / max(self.agent_levels)])

            obs_list.append(obs)
        return obs_list

    def step(self, actions):
        self.step_count += 1
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        done = (self.step_count >= self.max_steps)

        # Move agents
        dir_map = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        new_positions = []

        for a_idx, act in enumerate(actions):
            if act in dir_map:
                dx, dy = dir_map[act]
                nx = np.clip(self.agent_pos[a_idx][0] + dx, 0, self.grid_size - 1)
                ny = np.clip(self.agent_pos[a_idx][1] + dy, 0, self.grid_size - 1)
                new_positions.append((nx, ny))
            else:
                new_positions.append(self.agent_pos[a_idx])

        self.agent_pos = new_positions

        # Enhanced reward structure
        any_food_collected = False
        for f_idx in range(self.n_foods):
            if not self.food_exists[f_idx]:
                continue

            f_pos = self.food_pos[f_idx]
            f_level = self.food_levels[f_idx]

            # Calculate distances for all agents
            distances = [abs(self.agent_pos[j][0] - f_pos[0]) + abs(self.agent_pos[j][1] - f_pos[1])
                         for j in range(self.n_agents)]

            # Proximity rewards
            for j in range(self.n_agents):
                if distances[j] == 0:
                    rewards[j] += 2.0
                elif distances[j] <= 2:
                    rewards[j] += 1.0 / (distances[j] + 1)

            # Find agents at food position
            agents_at_food = [j for j in range(self.n_agents) if distances[j] == 0]

            if agents_at_food:
                total_level = sum(self.agent_levels[j] for j in agents_at_food)

                if total_level >= f_level:
                    collection_reward = 13.0 + f_level * 3.0 #from 11
                    cooperation_bonus = 1.5 if len(agents_at_food) > 1 else 0.0 #from 1.0

                    for j in range(self.n_agents):
                        rewards[j] += collection_reward + cooperation_bonus

                    self.food_exists[f_idx] = False
                    any_food_collected = True

        # Completion bonus
        if all(not exists for exists in self.food_exists):
            completion_bonus = 11.0 * (1.0 - self.step_count / self.max_steps) #from 10
            for j in range(self.n_agents):
                rewards[j] += completion_bonus
            done = True

        # Time penalty and uncollected food penalty
        if not any_food_collected:
            time_penalty = 0.04 * (self.step_count / self.max_steps)#from 0.02
            for j in range(self.n_agents):
                rewards[j] -= time_penalty
        if done and any(self.food_exists):
            for j in range(self.n_agents):
                rewards[j] -= 0.5 * sum(self.food_exists)

        obs = self._get_obs()
        return obs, rewards.tolist(), done, [{}] * self.n_agents