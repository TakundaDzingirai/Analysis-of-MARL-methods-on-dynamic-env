import numpy as np
import time
from environment import LBFEnv
from training import train_advanced_iql, evaluate_agent
from hyperparams import HyperParams

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
            env = LBFEnv(**self.env_params, seed=seed)
            agents, history = train_advanced_iql(
                env, hyperparams, episodes=training_episodes,
                eval_interval=training_episodes // 2, verbose=False, seed=seed + trial
            )

            def policy_func(o):
                return [agents[i].act(o[i], training=False) for i in range(len(agents))]

            final_stats = evaluate_agent(self.env_params, policy_func, n_episodes=100, seed=seed)

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

                        env = LBFEnv(**self.env_params, seed=42)
                        agents, history = train_advanced_iql(
                            env, hyperparams, episodes=training_episodes,
                            eval_interval=training_episodes, verbose=False, seed=42 + trial
                        )

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