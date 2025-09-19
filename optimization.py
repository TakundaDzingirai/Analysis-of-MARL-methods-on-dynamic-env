import numpy as np
import time
import json
from environment import LBFEnv
from training import train_iql, train_maddpg, evaluate_agent
from hyperparams import HyperParams

class HyperparameterOptimizer:
    def __init__(self, env_params):
        self.env_params = env_params
        self.results = []

    def random_search(self, model_type='iql', n_trials=30, training_episodes=800, seed=42, early_stopping_trials=15, min_trials=20):
        """Random search with early stopping for IQL or MADDPG"""
        print(f"Starting random hyperparameter search for {model_type.upper()} with up to {n_trials} trials, "
              f"minimum {min_trials} trials, and early stopping after {early_stopping_trials} trials without improvement...")
        start_time = time.time()

        best_score = -np.inf
        best_params = None
        no_improvement_count = 0
        trial = 0

        if model_type == 'iql':
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
        else:  # maddpg
            param_ranges = {
                'lr_actor': (1e-4, 1e-3),
                'lr_critic': (1e-3, 2e-3),
                'gamma': [0.95, 0.99],
                'tau': (0.005, 0.02),
                'reward_scaling': (0.8, 1.5)
            }

        while trial < n_trials and (trial < min_trials or no_improvement_count < early_stopping_trials):
            trial += 1
            hyperparams = HyperParams(**{
                key: np.random.uniform(*param_ranges[key]) if isinstance(param_ranges[key], tuple)
                else np.random.choice(param_ranges[key])
                for key in param_ranges
            })

            print(f"Trial {trial}/{n_trials}: Testing hyperparams...")
            env = LBFEnv(**self.env_params, seed=seed)
            if model_type == 'iql':
                agents, history = train_iql(
                    env, hyperparams, episodes=training_episodes,
                    eval_interval=training_episodes // 2, verbose=False, seed=seed + trial
                )
                policy_func = lambda o: [agents[i].act(o[i], training=False) for i in range(len(agents))]
            else:  # maddpg
                maddpg, history = train_maddpg(
                    env, hyperparams, episodes=training_episodes,
                    eval_interval=training_episodes // 2, verbose=False, seed=seed + trial
                )
                policy_func = lambda o: maddpg.act(o, training=False)

            final_stats = evaluate_agent(self.env_params, policy_func, n_episodes=100, seed=seed)

            score = (final_stats['mean_return'] +
                     final_stats['success_rate'] * 20 +
                     final_stats['mean_foods'] * 10)

            result = {
                'trial': trial,
                'hyperparams': hyperparams.to_dict(),
                'score': score,
                'stats': final_stats
            }

            self.results.append(result)

            if score > best_score + 0.1:
                best_score = score
                best_params = hyperparams
                no_improvement_count = 0
                print(f"  New best score: {score:.2f}, Return: {final_stats['mean_return']:.2f}, "
                      f"Success: {final_stats['success_rate']:.2f}, Foods: {final_stats['mean_foods']:.2f}")
            else:
                no_improvement_count += 1
                print(f"  Score: {score:.2f}, Return: {final_stats['mean_return']:.2f}, "
                      f"Success: {final_stats['success_rate']:.2f}, Foods: {final_stats['mean_foods']:.2f}, "
                      f"No improvement for {no_improvement_count}/{early_stopping_trials} trials")

            if best_params is not None:
                try:
                    with open(f'intermediate_{model_type}_hyperparams.json', 'w') as f:
                        json.dump(best_params.to_dict(), f, indent=2)
                except Exception as e:
                    print(f"Could not save intermediate hyperparameters: {e}")

        elapsed = time.time() - start_time
        print(f"\nOptimization completed in {elapsed:.1f}s after {trial} trials")
        print(f"Best score: {best_score:.2f}")
        print(f"Best hyperparams: {best_params.to_dict()}")

        return best_params, self.results