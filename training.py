import numpy as np
import random
from collections import deque
from environment import LBFEnv
from agent import AdvancedIQLAgent
from maddpg import MADDPG
from hyperparams import HyperParams

def train_iql(env, hyperparams, episodes=5000, eval_interval=200, verbose=True, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    obs_dim = len(env._get_obs()[0])
    agents = [AdvancedIQLAgent(obs_dim, hyperparams=hyperparams) for _ in range(env.n_agents)]
    history = {
        'episode': [],
        'mean_return': [],
        'mean_foods': [],
        'success_rate': [],
        'avg_td_error': []
    }
    recent_returns = deque(maxlen=100)

    for episode in range(episodes):
        obs = env.reset()
        episode_rewards = np.zeros(env.n_agents)
        done = False
        while not done:
            actions = [agents[i].act(obs[i], training=True) for i in range(env.n_agents)]
            next_obs, rewards, done, _ = env.step(actions)
            for i in range(env.n_agents):
                agents[i].update(obs[i], actions[i], rewards[i], next_obs[i], done)
            obs = next_obs
            episode_rewards += np.array(rewards)
        recent_returns.append(np.sum(episode_rewards))

        if (episode + 1) % eval_interval == 0:
            stats = evaluate_agent(env_params=env.__dict__, agent_policy_func=lambda o: [agents[i].act(o[i], training=False) for i in range(len(agents))], n_episodes=100, seed=seed)
            history['episode'].append(episode + 1)
            history['mean_return'].append(stats['mean_return'])
            history['mean_foods'].append(stats['mean_foods'])
            history['success_rate'].append(stats['success_rate'])
            history['avg_td_error'].append(np.mean([agent.get_stats()['avg_td_error'] for agent in agents]))
            if verbose:
                print(f"Episode {episode + 1}/{episodes}: Return={stats['mean_return']:.2f}, Foods={stats['mean_foods']:.2f}, Success={stats['success_rate']:.2f}, TD Error={history['avg_td_error'][-1]:.2f}")

    return agents, history

def train_maddpg(env, hyperparams, episodes=1000, eval_interval=200, verbose=True, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    state_dim = len(env._get_obs()[0])
    action_dim = 5  # Stay, up, down, left, right
    maddpg = MADDPG(state_dim, action_dim, env.n_agents, hyperparams)
    history = {
        'episode': [],
        'mean_return': [],
        'mean_foods': [],
        'success_rate': [],
        'avg_critic_loss': []
    }

    for episode in range(episodes):
        state = env.reset()
        episode_rewards = np.zeros(env.n_agents)
        done = False
        while not done:
            actions = maddpg.act(state, training=True)
            next_state, rewards, done, _ = env.step(actions)
            maddpg.add_experience(state, actions, rewards, next_state, [done] * env.n_agents)
            critic_loss = maddpg.update(batch_size=64)
            state = next_state
            episode_rewards += np.array(rewards)

        if (episode + 1) % eval_interval == 0:
            stats = evaluate_agent(env_params=env.__dict__, agent_policy_func=lambda o: maddpg.act(o, training=False), n_episodes=100, seed=seed)
            history['episode'].append(episode + 1)
            history['mean_return'].append(stats['mean_return'])
            history['mean_foods'].append(stats['mean_foods'])
            history['success_rate'].append(stats['success_rate'])
            history['avg_critic_loss'].append(np.mean([s['avg_critic_loss'] for s in maddpg.get_stats()]))
            if verbose:
                print(f"Episode {episode + 1}/{episodes}: Return={stats['mean_return']:.2f}, Foods={stats['mean_foods']:.2f}, Success={stats['success_rate']:.2f}, Critic Loss={history['avg_critic_loss'][-1]:.2f}")

    return maddpg, history

def evaluate_agent(env_params, agent_policy_func, n_episodes=100, gamma=0.99, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    env = LBFEnv(**env_params, seed=seed)
    returns = []
    foods_collected = []
    successes = []

    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = np.zeros(env.n_agents)
        initial_foods = sum(env.food_exists)
        done = False
        while not done:
            actions = agent_policy_func(obs)
            obs, rewards, done, _ = env.step(actions)
            episode_reward += np.array(rewards)
        final_foods = sum(env.food_exists)
        foods = initial_foods - final_foods
        returns.append(np.mean(episode_reward))
        foods_collected.append(foods / env.n_foods)
        successes.append(1.0 if foods == env.n_foods else 0.0)

    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_foods': np.mean(foods_collected),
        'success_rate': np.mean(successes)
    }