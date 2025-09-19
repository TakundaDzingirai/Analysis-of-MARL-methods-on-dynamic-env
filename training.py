import numpy as np
import random
from collections import deque
from environment import LBFEnv
from agent import AdvancedIQLAgent
from hyperparams import HyperParams
from maddpg import MADDPG


def train_maddpg(env, hyperparams, episodes=1000, eval_interval=200, verbose=True, seed=42):
    np.random.seed(seed)
    state_dim = len(env._get_obs()[0])
    action_dim = 5  # Stay, up, down, left, right
    maddpg = MADDPG(state_dim, action_dim, env.n_agents, hyperparams)
    history = {
        'episode': [],
        'mean_return': [],
        'mean_foods': [],
        'success_rate': []
    }

    for episode in range(episodes):
        state = env.reset()
        episode_rewards = np.zeros(env.n_agents)
        done = False
        while not done:
            actions = maddpg.act(state, training=True)
            next_state, rewards, done, _ = env.step(actions)
            maddpg.add_experience(state, actions, rewards, next_state, [done] * env.n_agents)
            maddpg.update(batch_size=64)
            state = next_state
            episode_rewards += np.array(rewards)

        if (episode + 1) % eval_interval == 0:
            stats = evaluate_maddpg(env_params=env.__dict__, maddpg=maddpg, n_episodes=100, seed=seed)
            history['episode'].append(episode + 1)
            history['mean_return'].append(stats['mean_return'])
            history['mean_foods'].append(stats['mean_foods'])
            history['success_rate'].append(stats['success_rate'])
            if verbose:
                print(f"Episode {episode + 1}/{episodes}: Return={stats['mean_return']:.2f}, "
                      f"Foods={stats['mean_foods']:.2f}, Success={stats['success_rate']:.2f}")

    return maddpg, history


def evaluate_maddpg(env_params, maddpg, n_episodes=100, seed=42):
    env = LBFEnv(**env_params, seed=seed)
    episode_returns = []
    episode_foods = []
    successes = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = np.zeros(env.n_agents)
        foods_collected = 0
        done = False
        while not done:
            actions = maddpg.act(state, training=False)
            state, rewards, done, _ = env.step(actions)
            episode_reward += np.array(rewards)
            foods_collected += sum(
                1 for i, exists in enumerate(env.food_exists) if not exists and i < len(env.food_exists))
        episode_returns.append(np.mean(episode_reward))
        episode_foods.append(foods_collected / env.n_foods)
        successes.append(1.0 if foods_collected == env.n_foods else 0.0)

    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_foods': np.mean(episode_foods),
        'success_rate': np.mean(successes)
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

        if foods_collected_ep == initial_foods:
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

def train_advanced_iql(env, hyperparams: HyperParams, episodes=5000, eval_interval=200, verbose=False, seed=None):
    """Enhanced training with comprehensive tracking"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    obs_dim = len(env._get_obs()[0])
    agents = [AdvancedIQLAgent(obs_dim, hyperparams=hyperparams) for _ in range(env.n_agents)]

    history = {'returns': [], 'foods': [], 'success_rates': [], 'episodes': []}
    recent_returns = deque(maxlen=100)

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        episode_return = 0

        while not done:
            actions = [agents[i].act(obs[i], training=True) for i in range(env.n_agents)]
            next_obs, rewards, done, _ = env.step(actions)

            for i in range(env.n_agents):
                agents[i].update(obs[i], actions[i], rewards[i], next_obs[i], done)

            episode_return += sum(rewards)
            obs = next_obs

        recent_returns.append(episode_return)

        for agent in agents:
            agent.adapt_parameters(episode_return)

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