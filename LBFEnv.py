import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict


class LBFEnv:
    def __init__(self, grid_size=4, n_agents=2, n_foods=2, agent_levels=[1, 2], food_levels=[1, 2], max_steps=50):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_foods = n_foods
        self.agent_levels = agent_levels
        self.food_levels = food_levels
        self.max_steps = max_steps
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
        """Improved normalized observation space - only active food"""
        obs_list = []
        for i in range(self.n_agents):
            obs = []

            # Own position (normalized)
            obs.extend([self.agent_pos[i][0] / (self.grid_size - 1),
                        self.agent_pos[i][1] / (self.grid_size - 1)])

            # Only active food positions and levels
            active_foods = []
            for j in range(self.n_foods):
                if self.food_exists[j]:
                    active_foods.append([
                        self.food_pos[j][0] / (self.grid_size - 1),
                        self.food_pos[j][1] / (self.grid_size - 1),
                        self.food_levels[j] / max(self.food_levels)
                    ])

            # Pad to consistent size (always n_foods entries)
            while len(active_foods) < self.n_foods:
                active_foods.append([-1.0, -1.0, 0.0])  # Inactive food marker

            for food_info in active_foods:
                obs.extend(food_info)

            # Other agent positions
            for j in range(self.n_agents):
                if i != j:
                    obs.extend([self.agent_pos[j][0] / (self.grid_size - 1),
                                self.agent_pos[j][1] / (self.grid_size - 1)])

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

        # Check for food collection with improved rewards
        any_food_collected = False
        for f_idx in range(self.n_foods):
            if not self.food_exists[f_idx]:
                continue

            f_pos = self.food_pos[f_idx]
            f_level = self.food_levels[f_idx]

            # Find agents at food position
            agents_at_food = [j for j in range(self.n_agents) if self.agent_pos[j] == f_pos]

            if agents_at_food:
                total_level = sum(self.agent_levels[j] for j in agents_at_food)

                # Small reward for reaching food
                for j in agents_at_food:
                    rewards[j] += 0.5

                # Balanced collection reward
                if total_level >= f_level:
                    print(f"Collected food at {f_pos} with levels {total_level} >= {f_level}")

                    # Balanced reward structure
                    collection_reward = 5.0 + f_level * 2.0
                    for j in range(self.n_agents):
                        rewards[j] += collection_reward

                    self.food_exists[f_idx] = False
                    any_food_collected = True

        # Check if all food collected (no bonus - keep rewards balanced)
        if all(not exists for exists in self.food_exists):
            done = True

        # Minimal time penalty
        if not any_food_collected:
            for j in range(self.n_agents):
                rewards[j] -= 0.01

        obs = self._get_obs()
        return obs, rewards.tolist(), done, [{}] * self.n_agents


class ImprovedIQLAgent:
    def __init__(self, obs_dim, n_actions=5, alpha=0.2, gamma=0.99, epsilon=0.8, epsilon_decay=0.99, epsilon_min=0.05):
        """Improved IQL agent with better hyperparameters"""

        self.initial_alpha = alpha
        self.alpha = alpha  # Higher learning rate
        self.gamma = gamma  # Higher discount factor
        self.epsilon = epsilon  # Start with high exploration
        self.epsilon_decay = epsilon_decay  # Faster decay
        self.epsilon_min = epsilon_min  # Lower minimum
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        # Q-table with zero initialization
        self.Q = defaultdict(lambda: np.zeros(n_actions))

        self.update_count = 0
        self.state_visits = defaultdict(int)

    def discretize_obs(self, obs):
        """Better discretization with rounding"""
        discretized = []
        for val in obs:
            if val < -0.5:  # Inactive food marker
                discretized.append(-1)
            else:
                # Round to 1 decimal place for normalized values
                discretized.append(round(val, 1))

        return tuple(discretized)

    def act(self, obs, training=True):
        state = self.discretize_obs(obs)
        self.state_visits[state] += 1

        # Epsilon-greedy with exploration bonus
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        # Choose best action with small random tie-breaking
        q_values = self.Q[state] + np.random.normal(0, 1e-6, self.n_actions)
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_obs, done):
        state = self.discretize_obs(state)
        next_state = self.discretize_obs(next_obs)

        # Q-learning update
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        self.update_count += 1

        # Decay learning rate and epsilon
        self.alpha = max(0.01, self.alpha * 0.9999)  # Slow decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_stats(self):
        if not self.Q:
            return {'n_states': 0, 'avg_q': 0, 'max_q': 0, 'min_q': 0, 'updates': 0}

        all_q_values = []
        for state_q in self.Q.values():
            all_q_values.extend(state_q)

        return {
            'n_states': len(self.Q),
            'avg_q': np.mean(all_q_values),
            'max_q': np.max(all_q_values),
            'min_q': np.min(all_q_values),
            'updates': self.update_count
        }


def evaluate(env, get_actions_func, n_episodes=100, gamma=0.99, verbose=False):
    """Consistent evaluation function"""
    returns = []
    foods_collected = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs = env.reset()
        G = 0.0
        done = False
        t = 0
        initial_foods = sum(env.food_exists)

        while not done and t < env.max_steps:
            actions = get_actions_func(obs)
            obs, rewards, done, _ = env.step(actions)
            G += (gamma ** t) * sum(rewards)
            t += 1

        final_foods = sum(env.food_exists)
        foods_collected.append(initial_foods - final_foods)
        returns.append(G)
        episode_lengths.append(t)

    stats = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_foods': np.mean(foods_collected),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean([f > 0 for f in foods_collected])
    }

    if verbose:
        print(f"Returns: {stats['mean_return']:.3f} ± {stats['std_return']:.3f}")
        print(f"Foods: {stats['mean_foods']:.2f}, Success rate: {stats['success_rate']:.2f}")
        print(f"Avg episode length: {stats['mean_length']:.1f}")

    return stats


def train_improved_iql(env, episodes=3000, eval_interval=100, verbose=True):
    """Enhanced training with consistent evaluation"""

    obs_dim = len(env._get_obs()[0])
    agents = [ImprovedIQLAgent(obs_dim) for _ in range(env.n_agents)]

    history = []
    food_history = []
    eval_episodes = []

    if verbose:
        print("Training Improved IQL...")
        print(f"Environment: {env.grid_size}x{env.grid_size}, {env.n_agents} agents, {env.n_foods} foods")
        print(f"Observation dimension: {obs_dim}")

    # Track training progress
    recent_returns = []
    best_return = -float('inf')
    patience = 15  # Increased patience
    no_improvement = 0

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
        if len(recent_returns) > 100:
            recent_returns.pop(0)

        # Evaluation with consistent seeding
        if ep % eval_interval == 0:
            # Use consistent seed for evaluation
            eval_seed = 42 + (ep // eval_interval)
            np.random.seed(eval_seed)
            random.seed(eval_seed)

            stats = evaluate(env, lambda o: [agents[i].act(o[i], training=False) for i in range(len(agents))],
                             n_episodes=50)

            # Reset seeds for training
            np.random.seed()
            random.seed()

            history.append(stats['mean_return'])
            food_history.append(stats['mean_foods'])
            eval_episodes.append(ep)

            if stats['mean_return'] > best_return:
                best_return = stats['mean_return']
                no_improvement = 0
            else:
                no_improvement += 1

            if verbose:
                avg_epsilon = np.mean([agent.epsilon for agent in agents])
                avg_alpha = np.mean([agent.alpha for agent in agents])
                q_stats = agents[0].get_q_stats()
                train_return = np.mean(recent_returns) if recent_returns else 0

                print(
                    f"Ep {ep:4d}: Train={train_return:6.1f}, Eval={stats['mean_return']:6.3f}±{stats['std_return']:5.3f}, "
                    f"Foods={stats['mean_foods']:.2f}, Success={stats['success_rate']:.2f}, "
                    f"ε={avg_epsilon:.3f}, α={avg_alpha:.3f}, States={q_stats['n_states']}")

            # Early stopping
            if no_improvement >= patience and ep > 1500:
                if verbose:
                    print(f"Early stopping at episode {ep}")
                break

    return agents, history, food_history, eval_episodes


def plot_results(history, food_history, eval_episodes, baseline_stats):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Returns plot
    ax1.plot(eval_episodes, history, 'b-', linewidth=2, label='IQL Return')
    ax1.axhline(y=baseline_stats['mean_return'], color='r', linestyle='--',
                label=f'Random Baseline ({baseline_stats["mean_return"]:.1f})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Return')
    ax1.set_title('Learning Curve: Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Food collection plot
    ax2.plot(eval_episodes, food_history, 'g-', linewidth=2, label='IQL Foods')
    ax2.axhline(y=baseline_stats['mean_foods'], color='r', linestyle='--',
                label=f'Random Baseline ({baseline_stats["mean_foods"]:.2f})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Foods Collected')
    ax2.set_title('Learning Curve: Food Collection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    print("IMPROVED IQL WITH BALANCED REWARDS")
    print("=" * 60)

    # Environment setup
    env = LBFEnv(
        grid_size=4,
        n_agents=2,
        n_foods=2,
        agent_levels=[1, 2],
        food_levels=[1, 2],
        max_steps=50
    )

    print(f"Environment: {env.grid_size}x{env.grid_size} grid")
    print(f"Agents: {env.n_agents} (levels {env.agent_levels})")
    print(f"Foods: {env.n_foods} (levels {env.food_levels})")
    print(f"Max steps: {env.max_steps}")

    # Baseline evaluation
    print("\nBASELINE EVALUATION")
    print("-" * 40)
    np.random.seed(42)
    random.seed(42)


    def random_policy(obs):
        return [np.random.randint(0, 5) for _ in range(len(obs))]


    baseline_stats = evaluate(env, random_policy, n_episodes=300, verbose=True)

    # Training
    print(f"\nTRAINING IMPROVED IQL")
    print("-" * 40)

    agents, history, food_history, eval_episodes = train_improved_iql(
        env, episodes=3000, eval_interval=100, verbose=True
    )

    # Final evaluation with more episodes
    print(f"\nFINAL EVALUATION")
    print("-" * 40)
    np.random.seed(42)
    random.seed(42)


    def iql_policy(obs):
        return [agents[i].act(obs[i], training=False) for i in range(len(obs))]


    final_stats = evaluate(env, iql_policy, n_episodes=300, verbose=True)

    # Results summary
    print(f"\nRESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Random':<12} {'IQL':<12} {'Improvement':<15}")
    print("-" * 60)

    return_improvement = final_stats['mean_return'] - baseline_stats['mean_return']
    food_improvement = final_stats['mean_foods'] - baseline_stats['mean_foods']

    print(f"{'Average Return':<20} {baseline_stats['mean_return']:<12.3f} "
          f"{final_stats['mean_return']:<12.3f} {return_improvement:+.3f}")
    print(f"{'Foods Collected':<20} {baseline_stats['mean_foods']:<12.2f} "
          f"{final_stats['mean_foods']:<12.2f} {food_improvement:+.2f}")
    print(f"{'Success Rate':<20} {baseline_stats['success_rate']:<12.2f} "
          f"{final_stats['success_rate']:<12.2f} "
          f"{final_stats['success_rate'] - baseline_stats['success_rate']:+.2f}")

    # Agent analysis
    print(f"\nAGENT ANALYSIS")
    print("-" * 40)
    for i, agent in enumerate(agents):
        stats = agent.get_q_stats()
        print(f"Agent {i}: {stats['n_states']} states, {stats['updates']} updates")
        print(f"          Q-range: [{stats['min_q']:.2f}, {stats['max_q']:.2f}], avg: {stats['avg_q']:.2f}")

    success = return_improvement > 5.0 and food_improvement > 0.1
    print(f"\n{'SUCCESS!' if success else 'STILL NEEDS WORK'}")

    if success:
        print("IQL successfully learned to outperform random policy!")
    else:
        print("Try even simpler environment or check reward structure")
        print("Debug: Check if Q-values are reasonable and agents explore properly")

    # Plot results
    if history:
        plot_results(history, food_history, eval_episodes, baseline_stats)

    # Debug information
    print(f"\nDEBUG INFO:")
    print(f"Final epsilon: {agents[0].epsilon:.3f}")
    print(f"Final alpha: {agents[0].alpha:.3f}")
    print(f"States visited: {len(agents[0].state_visits)}")

    # Sample some Q-values for debugging
    if agents[0].Q:
        sample_states = list(agents[0].Q.keys())[:5]
        print("Sample Q-values:")
        for state in sample_states:
            print(f"  State {state}: {agents[0].Q[state]}")