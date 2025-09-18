import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


@dataclass
class HyperParams:
    """Hyperparameter configuration"""
    alpha: float = 0.28457111354474063
    gamma: float = 0.99
    epsilon: float = 0.7849863661813384
    epsilon_decay: float = 0.998  # Adjusted from 0.9963570200350645
    epsilon_min: float = 0.059031733973677815
    alpha_decay: float = 0.98  # Adjusted from 0.9478492911803126
    alpha_min: float = 0.05  # Adjusted from 0.01
    reward_scaling: float = 1.4656855889679763
    exploration_bonus: float = 0.2  # Adjusted from 0.4488916678936584

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


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
                    # Relative position
                    rel_x = (fx - own_x) / (self.grid_size - 1)
                    rel_y = (fy - own_y) / (self.grid_size - 1)
                    # Manhattan distance (normalized)
                    dist = (abs(fx - own_x) + abs(fy - own_y)) / (2 * (self.grid_size - 1))

                    active_foods.append([
                        rel_x, rel_y, dist,
                        self.food_levels[j] / max(self.food_levels)
                    ])

            # Pad to consistent size
            while len(active_foods) < self.n_foods:
                active_foods.append([-2.0, -2.0, 1.0, 0.0])  # Inactive food marker

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

            # Proximity rewards (inverse distance)
            for j in range(self.n_agents):
                if distances[j] == 0:
                    rewards[j] += 2.0  # At food location
                elif distances[j] <= 2:
                    rewards[j] += 1.0 / (distances[j] + 1)  # Close to food

            # Find agents at food position
            agents_at_food = [j for j in range(self.n_agents) if distances[j] == 0]

            if agents_at_food:
                total_level = sum(self.agent_levels[j] for j in agents_at_food)

                # Collection logic
                if total_level >= f_level:
                    collection_reward = 10.0 + f_level * 3.0  # Adjusted from 8.0
                    cooperation_bonus = 1.0 if len(agents_at_food) > 1 else 0.0

                    for j in range(self.n_agents):
                        rewards[j] += collection_reward + cooperation_bonus

                    self.food_exists[f_idx] = False
                    any_food_collected = True

        # Completion bonus
        if all(not exists for exists in self.food_exists):
            completion_bonus = 10.0 * (1.0 - self.step_count / self.max_steps)
            for j in range(self.n_agents):
                rewards[j] += completion_bonus
            done = True

        # Time penalty and uncollected food penalty
        if not any_food_collected:
            time_penalty = 0.02 * (self.step_count / self.max_steps)
            for j in range(self.n_agents):
                rewards[j] -= time_penalty
        if done and any(self.food_exists):
            for j in range(self.n_agents):
                rewards[j] -= 0.5 * sum(self.food_exists)

        obs = self._get_obs()
        return obs, rewards.tolist(), done, [{}] * self.n_agents


class AdvancedIQLAgent:
    def __init__(self, obs_dim, n_actions=5, hyperparams: HyperParams = None):
        """Advanced IQL agent with sophisticated learning mechanisms"""

        self.hyperparams = hyperparams or HyperParams()
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        # Initialize parameters from hyperparams
        self.initial_alpha = self.hyperparams.alpha
        self.alpha = self.hyperparams.alpha
        self.gamma = self.hyperparams.gamma
        self.epsilon = self.hyperparams.epsilon
        self.epsilon_decay = self.hyperparams.epsilon_decay
        self.epsilon_min = self.hyperparams.epsilon_min
        self.alpha_decay = self.hyperparams.alpha_decay
        self.alpha_min = self.hyperparams.alpha_min
        self.reward_scaling = self.hyperparams.reward_scaling
        self.exploration_bonus = self.hyperparams.exploration_bonus

        # Q-table with optimistic initialization
        self.Q = defaultdict(lambda: np.ones(n_actions) * 0.5)

        # Advanced tracking
        self.update_count = 0
        self.state_visits = defaultdict(int)
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        self.recent_rewards = deque(maxlen=100)
        self.recent_td_errors = deque(maxlen=100)

        # Adaptive mechanisms
        self.performance_window = deque(maxlen=50)
        self.last_performance_check = 0

    def discretize_obs(self, obs, precision=2):
        """Enhanced discretization with adaptive precision"""
        discretized = []
        for val in obs:
            if val < -1.5:  # Inactive marker
                discretized.append(-999)
            else:
                # Use higher precision for critical values
                discretized.append(round(val, precision))
        return tuple(discretized)

    def act(self, obs, training=True):
        state = self.discretize_obs(obs)
        self.state_visits[state] += 1

        if not training:
            # Exploitation mode
            q_values = self.Q[state]
            return int(np.argmax(q_values + np.random.normal(0, 1e-8, self.n_actions)))

        # Advanced exploration strategy
        visit_bonus = min(self.exploration_bonus, 10.0 / (1 + self.state_visits[state]))
        effective_epsilon = min(0.95, self.epsilon + visit_bonus)

        # UCB-like exploration for less visited state-action pairs
        if random.random() < effective_epsilon:
            # Favor less explored actions
            action_counts = [self.state_action_counts[state][a] for a in range(self.n_actions)]
            min_count = min(action_counts) if action_counts else 0
            least_tried = [a for a in range(self.n_actions) if action_counts[a] <= min_count + 1]
            return random.choice(least_tried)

        # Exploitation with noise
        q_values = self.Q[state]
        return int(np.argmax(q_values + np.random.normal(0, 1e-6, self.n_actions)))

    def update(self, state, action, reward, next_obs, done):
        state = self.discretize_obs(state)
        next_state = self.discretize_obs(next_obs)

        # Scale reward
        scaled_reward = reward * self.reward_scaling
        self.recent_rewards.append(scaled_reward)

        # Track state-action visits
        self.state_action_counts[state][action] += 1

        # Q-learning update with enhancements
        if done:
            td_target = scaled_reward
        else:
            td_target = scaled_reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - self.Q[state][action]
        self.recent_td_errors.append(abs(td_error))

        # Adaptive learning rate
        adaptive_alpha = self.alpha

        # Learn faster from significant rewards/errors
        if abs(scaled_reward) > 5:
            adaptive_alpha *= 1.3
        if abs(td_error) > 2:
            adaptive_alpha *= 1.2

        # Reduce learning for frequently visited state-actions
        visit_count = self.state_action_counts[state][action]
        if visit_count > 10:
            adaptive_alpha *= (10.0 / visit_count) ** 0.3

        # Update Q-value
        self.Q[state][action] += adaptive_alpha * td_error
        self.update_count += 1

        # Parameter decay
        if self.update_count % 500 == 0:
            self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def adapt_parameters(self, episode_return):
        """Adaptive parameter adjustment based on performance"""
        self.performance_window.append(episode_return)

        if len(self.performance_window) >= 25 and self.update_count - self.last_performance_check > 1000:
            recent_performance = np.mean(list(self.performance_window)[-10:])
            older_performance = np.mean(list(self.performance_window)[-25:-10])

            # If performance is stagnating, increase exploration
            if recent_performance <= older_performance + 1.0:
                self.epsilon = min(0.8, self.epsilon * 1.1)
                self.alpha = min(0.3, self.alpha * 1.05)

            self.last_performance_check = self.update_count

    def get_stats(self):
        """Comprehensive agent statistics"""
        if not self.Q:
            return {'n_states': 0, 'avg_q': 0, 'max_q': 0, 'min_q': 0, 'updates': 0}

        all_q_values = []
        for state_q in self.Q.values():
            all_q_values.extend(state_q)

        return {
            'n_states': len(self.Q),
            'n_state_actions': sum(len(sa) for sa in self.state_action_counts.values()),
            'avg_q': np.mean(all_q_values),
            'max_q': np.max(all_q_values),
            'min_q': np.min(all_q_values),
            'updates': self.update_count,
            'avg_td_error': np.mean(self.recent_td_errors) if self.recent_td_errors else 0,
            'epsilon': self.epsilon,
            'alpha': self.alpha
        }


def evaluate_agent(env_params, agent_policy_func, n_episodes=100, gamma=0.99, seed=42):
    """Robust evaluation function"""
    np.random.seed(seed)
    random.seed(seed)

    env = LBFEnv(**env_params)
    returns = []
    foods_collected = []
    episode_lengths = []
    success_episodes = 0

    for ep in range(n_episodes):
        obs = env.reset()
        G = 0.0
        done = False
        t = 0
        initial_foods = sum(env.food_exists)

        while not done and t < env.max_steps:
            actions = agent_policy_func(obs)
            obs, rewards, done, _ = env.step(actions)
            G += (gamma ** t) * sum(rewards)
            t += 1

        final_foods = sum(env.food_exists)
        foods_collected_ep = initial_foods - final_foods
        foods_collected.append(foods_collected_ep)
        returns.append(G)
        episode_lengths.append(t)

        if foods_collected_ep == initial_foods:  # All food collected
            success_episodes += 1

    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_foods': np.mean(foods_collected),
        'std_foods': np.std(foods_collected),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_episodes / n_episodes,
        'median_return': np.median(returns)
    }


def train_advanced_iql(env, hyperparams: HyperParams, episodes=5000,  # Adjusted from 3000
                       eval_interval=200,  # Adjusted from 100
                       verbose=False, seed=None):
    """Enhanced training with comprehensive tracking"""

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    obs_dim = len(env._get_obs()[0])
    agents = [AdvancedIQLAgent(obs_dim, hyperparams=hyperparams) for _ in range(env.n_agents)]

    # Tracking
    history = {'returns': [], 'foods': [], 'success_rates': [], 'episodes': []}
    recent_returns = deque(maxlen=100)

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        episode_return = 0

        while not done:
            actions = [agents[i].act(obs[i], training=True) for i in range(env.n_agents)]
            next_obs, rewards, done, _ = env.step(actions)

            # Update agents
            for i in range(env.n_agents):
                agents[i].update(obs[i], actions[i], rewards[i], next_obs[i], done)

            episode_return += sum(rewards)
            obs = next_obs

        recent_returns.append(episode_return)

        # Adaptive parameter adjustment
        for agent in agents:
            agent.adapt_parameters(episode_return)

        # Evaluation
        if ep % eval_interval == 0:
            def policy_func(o):
                return [agents[i].act(o[i], training=False) for i in range(len(agents))]

            env_params = {
                'grid_size': env.grid_size,
                'n_agents': env.n_agents,
                'n_foods': env.n_foods,
                'agent_levels': env.agent_levels,
                'food_levels': env.food_levels,
                'max_steps': env.max_steps
            }

            stats = evaluate_agent(env_params, policy_func, n_episodes=50, seed=42 + ep)

            history['returns'].append(stats['mean_return'])
            history['foods'].append(stats['mean_foods'])
            history['success_rates'].append(stats['success_rate'])
            history['episodes'].append(ep)

            if verbose:
                agent_stats = agents[0].get_stats()
                train_return = np.mean(recent_returns) if recent_returns else 0

                print(f"Ep {ep:4d}: Train={train_return:6.1f}, "
                      f"Eval={stats['mean_return']:6.2f}±{stats['std_return']:5.2f}, "
                      f"Foods={stats['mean_foods']:.2f}, "
                      f"Success={stats['success_rate']:.2f}, "
                      f"ε={agent_stats['epsilon']:.3f}, "
                      f"α={agent_stats['alpha']:.3f}, "
                      f"States={agent_stats['n_states']}")

    return agents, history


class HyperparameterOptimizer:
    def __init__(self, env_params):
        self.env_params = env_params
        self.results = []

    def random_search(self, n_trials=50, training_episodes=500, seed=42):
        """Advanced random search with smart sampling"""

        print(f"Starting random hyperparameter search with {n_trials} trials...")
        start_time = time.time()

        best_score = -np.inf
        best_params = None

        # Define parameter distributions
        param_ranges = {
            'alpha': (0.05, 0.4),
            'gamma': [0.95, 0.97, 0.99, 0.995],
            'epsilon': (0.7, 0.95),
            'epsilon_decay': (0.99, 0.999),
            'epsilon_min': (0.01, 0.08),
            'alpha_decay': (0.92, 0.98),
            'reward_scaling': (0.8, 1.5),
            'exploration_bonus': (0.1, 0.5)
        }

        for trial in range(n_trials):
            # Sample hyperparameters
            hyperparams = HyperParams(
                alpha=np.random.uniform(*param_ranges['alpha']),
                gamma=np.random.choice(param_ranges['gamma']),
                epsilon=np.random.uniform(*param_ranges['epsilon']),
                epsilon_decay=np.random.uniform(*param_ranges['epsilon_decay']),
                epsilon_min=np.random.uniform(*param_ranges['epsilon_min']),
                alpha_decay=np.random.uniform(*param_ranges['alpha_decay']),
                reward_scaling=np.random.uniform(*param_ranges['reward_scaling']),
                exploration_bonus=np.random.uniform(*param_ranges['exploration_bonus'])
            )

            print(f"Trial {trial + 1}/{n_trials}: Testing hyperparams...")

            # Train and evaluate
            env = LBFEnv(**self.env_params, seed=seed)
            agents, history = train_advanced_iql(
                env, hyperparams, episodes=training_episodes,
                eval_interval=training_episodes // 2, verbose=False, seed=seed + trial
            )

            # Final evaluation
            def policy_func(o):
                return [agents[i].act(o[i], training=False) for i in range(len(agents))]

            final_stats = evaluate_agent(self.env_params, policy_func, n_episodes=100, seed=seed)

            # Composite score (return + success_rate + foods)
            score = (final_stats['mean_return'] +
                     final_stats['success_rate'] * 20 +
                     final_stats['mean_foods'] * 10)

            result = {
                'trial': trial + 1,
                'hyperparams': hyperparams.to_dict(),
                'score': score,
                'stats': final_stats
            }

            self.results.append(result)

            if score > best_score:
                best_score = score
                best_params = hyperparams

            print(f"  Score: {score:.2f}, Return: {final_stats['mean_return']:.2f}, "
                  f"Success: {final_stats['success_rate']:.2f}")

        elapsed = time.time() - start_time
        print(f"\nOptimization completed in {elapsed:.1f}s")
        print(f"Best score: {best_score:.2f}")
        print(f"Best hyperparams: {best_params.to_dict()}")

        return best_params, self.results

    def grid_search(self, param_grid=None, training_episodes=300):
        """Focused grid search on key parameters"""

        if param_grid is None:
            param_grid = {
                'alpha': [0.1, 0.2, 0.3],
                'gamma': [0.95, 0.99],
                'epsilon': [0.8, 0.9],
                'epsilon_decay': [0.995, 0.998]
            }

        print(f"Starting grid search...")
        total_combinations = np.prod([len(values) for values in param_grid.values()])
        print(f"Total combinations: {total_combinations}")

        best_score = -np.inf
        best_params = None
        trial = 0

        for alpha in param_grid['alpha']:
            for gamma in param_grid['gamma']:
                for epsilon in param_grid['epsilon']:
                    for epsilon_decay in param_grid['epsilon_decay']:
                        trial += 1

                        hyperparams = HyperParams(
                            alpha=alpha,
                            gamma=gamma,
                            epsilon=epsilon,
                            epsilon_decay=epsilon_decay
                        )

                        print(f"Trial {trial}/{total_combinations}: "
                              f"α={alpha}, γ={gamma}, ε={epsilon}, ε_decay={epsilon_decay}")

                        # Train and evaluate
                        env = LBFEnv(**self.env_params, seed=42)
                        agents, history = train_advanced_iql(
                            env, hyperparams, episodes=training_episodes,
                            eval_interval=training_episodes, verbose=False, seed=42 + trial
                        )

                        # Final evaluation
                        def policy_func(o):
                            return [agents[i].act(o[i], training=False) for i in range(len(agents))]

                        final_stats = evaluate_agent(self.env_params, policy_func,
                                                     n_episodes=100, seed=42)

                        score = (final_stats['mean_return'] +
                                 final_stats['success_rate'] * 15 +
                                 final_stats['mean_foods'] * 8)

                        result = {
                            'trial': trial,
                            'hyperparams': hyperparams.to_dict(),
                            'score': score,
                            'stats': final_stats
                        }

                        self.results.append(result)

                        if score > best_score:
                            best_score = score
                            best_params = hyperparams

                        print(f"  Score: {score:.2f}, Return: {final_stats['mean_return']:.2f}")

        print(f"\nBest score: {best_score:.2f}")
        print(f"Best hyperparams: {best_params.to_dict()}")

        return best_params, self.results


def plot_optimization_results(results, title="Hyperparameter Optimization Results"):
    """Visualize optimization results"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Extract data
    scores = [r['score'] for r in results]
    returns = [r['stats']['mean_return'] for r in results]
    success_rates = [r['stats']['success_rate'] for r in results]
    foods = [r['stats']['mean_foods'] for r in results]

    # Score progression
    axes[0, 0].plot(scores, 'b-', alpha=0.7)
    axes[0, 0].scatter(range(len(scores)), scores, alpha=0.6)
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Composite Score')
    axes[0, 0].set_title('Score Progression')
    axes[0, 0].grid(True, alpha=0.3)

    # Returns distribution
    axes[0, 1].hist(returns, bins=20, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Mean Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Returns Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Success rate vs Return
    axes[1, 0].scatter(returns, success_rates, alpha=0.6, c=scores, cmap='viridis')
    axes[1, 0].set_xlabel('Mean Return')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title('Success Rate vs Return')
    axes[1, 0].grid(True, alpha=0.3)

    # Parameter correlation (alpha vs score)
    alphas = [r['hyperparams']['alpha'] for r in results]
    axes[1, 1].scatter(alphas, scores, alpha=0.6, c='red')
    axes[1, 1].set_xlabel('Alpha (Learning Rate)')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Alpha vs Score')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_training_curves(history, title="Training Progress"):
    """Plot training curves"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    episodes = history['episodes']

    # Returns
    axes[0].plot(episodes, history['returns'], 'b-', linewidth=2, label='Mean Return')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Return')
    axes[0].set_title('Learning Curve: Returns')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Food collection
    axes[1].plot(episodes, history['foods'], 'g-', linewidth=2, label='Foods Collected')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Foods Collected')
    axes[1].set_title('Learning Curve: Food Collection')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Success rate
    axes[2].plot(episodes, history['success_rates'], 'r-', linewidth=2, label='Success Rate')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_title('Learning Curve: Success Rate')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    print("ADVANCED IQL WITH HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    # Environment configuration
    env_params = {
        'grid_size': 4,
        'n_agents': 2,
        'n_foods': 2,
        'agent_levels': [1, 2],
        'food_levels': [1, 2],
        'max_steps': 40
    }

    env = LBFEnv(**env_params, seed=42)
    print(f"Environment: {env.grid_size}x{env.grid_size} grid")
    print(f"Agents: {env.n_agents} (levels {env.agent_levels})")
    print(f"Foods: {env.n_foods} (levels {env.food_levels})")
    print(f"Max steps: {env.max_steps}")

    # Baseline evaluation
    print("\nBASELINE EVALUATION")
    print("-" * 40)


    def random_policy(obs):
        return [np.random.randint(0, 5) for _ in range(len(obs))]


    baseline_stats = evaluate_agent(env_params, random_policy, n_episodes=200, seed=42)
    print(f"Random baseline - Return: {baseline_stats['mean_return']:.2f}, "
          f"Foods: {baseline_stats['mean_foods']:.2f}, "
          f"Success: {baseline_stats['success_rate']:.2f}")

    # Hyperparameter optimization
    print(f"\nHYPERPARAMETER OPTIMIZATION")
    print("-" * 40)

    optimizer = HyperparameterOptimizer(env_params)

    # Choose optimization method
    use_random_search = True  # Set to False for grid search

    if use_random_search:
        best_params, results = optimizer.random_search(
            n_trials=30, training_episodes=800, seed=42
        )
        method_name = "Random Search"
    else:
        best_params, results = optimizer.grid_search(training_episodes=600)
        method_name = "Grid Search"

    # Display top results
    print(f"\nTOP 5 RESULTS FROM {method_name.upper()}:")
    print("-" * 60)
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]

    for i, result in enumerate(sorted_results):
        print(f"Rank {i + 1}: Score={result['score']:.2f}")
        print(f"  Return: {result['stats']['mean_return']:.2f}±{result['stats']['std_return']:.2f}")
        print(f"  Foods: {result['stats']['mean_foods']:.2f}, Success: {result['stats']['success_rate']:.2f}")
        print(f"  Hyperparams: {result['hyperparams']}")
        print()

    # Training with best hyperparameters
    print(f"TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("-" * 40)
    print(f"Using best hyperparams: {best_params.to_dict()}")

    env = LBFEnv(**env_params, seed=42)
    best_agents, best_history = train_advanced_iql(
        env, best_params, episodes=5000, eval_interval=200, verbose=True, seed=42
    )

    # Final comprehensive evaluation
    print(f"\nFINAL EVALUATION")
    print("-" * 40)


    def best_policy(obs):
        return [best_agents[i].act(obs[i], training=False) for i in range(len(obs))]


    final_stats = evaluate_agent(env_params, best_policy, n_episodes=500, seed=42)

    # Results comparison
    print(f"\nRESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Random':<12} {'Optimized IQL':<15} {'Improvement':<15}")
    print("-" * 62)

    metrics = ['mean_return', 'mean_foods', 'success_rate']
    labels = ['Average Return', 'Foods Collected', 'Success Rate']

    for metric, label in zip(metrics, labels):
        baseline_val = baseline_stats[metric]
        final_val = final_stats[metric]
        improvement = final_val - baseline_val

        print(f"{label:<20} {baseline_val:<12.3f} {final_val:<15.3f} {improvement:+.3f}")

    # Agent analysis
    print(f"\nAGENT ANALYSIS")
    print("-" * 40)
    for i, agent in enumerate(best_agents):
        stats = agent.get_stats()
        print(f"Agent {i}:")
        print(f"  States explored: {stats['n_states']}")
        print(f"  State-actions: {stats['n_state_actions']}")
        print(f"  Updates: {stats['updates']}")
        print(f"  Q-value range: [{stats['min_q']:.2f}, {stats['max_q']:.2f}]")
        print(f"  Final ε: {stats['epsilon']:.3f}, α: {stats['alpha']:.3f}")
        print(f"  Avg TD error: {stats['avg_td_error']:.3f}")

    # Success evaluation
    improvement_threshold = 10.0
    success = (final_stats['mean_return'] - baseline_stats['mean_return']) > improvement_threshold

    print(f"\n{'🎉 OPTIMIZATION SUCCESS!' if success else '⚠️ NEEDS MORE WORK'}")

    if success:
        print(f"Optimized IQL significantly outperformed random policy!")
        print(f"Return improvement: {final_stats['mean_return'] - baseline_stats['mean_return']:.2f}")
    else:
        print(f"Try different hyperparameter ranges or longer training.")

    # Visualization
    print(f"\nGENERATING VISUALIZATIONS...")

    # Plot optimization results
    plot_optimization_results(results, f"Hyperparameter {method_name} Results")

    # Plot training curves for best run
    plot_training_curves(best_history, "Training with Optimized Hyperparameters")

    # Save results
    results_summary = {
        'optimization_method': method_name,
        'best_hyperparams': best_params.to_dict(),
        'baseline_stats': baseline_stats,
        'final_stats': final_stats,
        'improvement': {
            'return': final_stats['mean_return'] - baseline_stats['mean_return'],
            'foods': final_stats['mean_foods'] - baseline_stats['mean_foods'],
            'success_rate': final_stats['success_rate'] - baseline_stats['success_rate']
        },
        'all_results': results
    }

    # Optional: Save to file
    try:
        with open('iql_optimization_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"Results saved to 'iql_optimization_results.json'")
    except Exception as e:
        print(f"Could not save results file: {e}")

    print(f"\nOPTIMIZATION COMPLETE!")
    print(f"Best configuration found and validated.")
    print(f"Use the best hyperparameters for future training runs.")