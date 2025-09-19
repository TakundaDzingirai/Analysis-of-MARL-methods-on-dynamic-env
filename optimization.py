import numpy as np
import time
import json
from environment import LBFEnv
from training import train_iql, train_maddpg, train_qmix, evaluate_agent
from hyperparams import HyperParams

class HyperparameterOptimizer:
    def __init__(self, env_params):
        self.env_params = env_params
        self.results = []

    def random_search(self, model_type='iql', n_trials=30, training_episodes=800, seed=42, early_stopping_trials=15, min_trials=20):
        """Random search with early stopping for IQL, MADDPG, or QMIX"""
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
        elif model_type == 'maddpg':
            param_ranges = {
                'lr_actor': (1e-4, 1e-3),
                'lr_critic': (1e-3, 2e-3),
                'gamma': [0.95, 0.99],
                'tau': (0.005, 0.02),
                'reward_scaling': (0.8, 1.5)
            }
        else:  # qmix
            param_ranges = {
                'lr': (1e-4, 1e-3),  # Learning rate for Q-networks and mixer
                'gamma': [0.95, 0.99],
                'epsilon': (0.1, 0.5),  # Initial epsilon for exploration
                'epsilon_decay': (0.99, 0.999),
                'epsilon_min': (0.01, 0.1),
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
            elif model_type == 'maddpg':
                maddpg, history = train_maddpg(
                    env, hyperparams, episodes=training_episodes,
                    eval_interval=training_episodes // 2, verbose=False, seed=seed + trial
                )
                policy_func = lambda o: maddpg.act(o, training=False)
            else:  # qmix
                qmix, history = train_qmix(
                    env, hyperparams, episodes=training_episodes,
                    eval_interval=training_episodes // 2, verbose=False, seed=seed + trial
                )
                policy_func = lambda o: qmix.act(o, training=False)

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
# NEW: Bayesian search method (adapted simple implementation using NumPy/SciPy)
    def bayesian_search(self, model_type='qmix', n_trials=60, training_episodes=1000, seed=42, early_stopping_trials=20, min_trials=10):
        """Bayesian hyperparameter search for QMIX with early stopping"""
        print(f"Starting Bayesian hyperparameter search for {model_type.upper()} with up to {n_trials} trials, "
              f"minimum {min_trials} trials, and early stopping after {early_stopping_trials} trials without improvement...")
        np.random.seed(seed)
        start_time = time.time()

        best_score = -np.inf
        best_params = None
        no_improvement_count = 0
        trial = 0

        # Expanded param_ranges for QMIX
        if model_type == 'qmix':
            param_ranges = {
                'lr': (1e-4, 5e-3),  # MODIFIED: Expanded LR range for stability
                'gamma': (0.95, 0.99),  # Treat as continuous, snap to [0.95, 0.99]
                'epsilon': (0.1, 0.5),
                'epsilon_decay': (0.99, 0.999),
                'epsilon_min': (0.01, 0.1),
                'reward_scaling': (0.8, 1.5),
                'tau': (0.001, 0.01),  # NEW: Expanded search space
                'hidden_dim': (32, 128),  # NEW: Expanded search space (snap to [32,64,128])
                'batch_size': (32, 128)   # NEW: Expanded search space (snap to [32,64,128])
            }

        param_keys = list(param_ranges.keys())
        bounds = [param_ranges[key] for key in param_keys]
        dim = len(bounds)

        # Discrete snapping options
        discrete_options = {
            'gamma': [0.95, 0.99],
            'hidden_dim': [32, 64, 128],
            'batch_size': [32, 64, 128]
        }

        x_m = []  # Measured points (hyperparams)
        y_m = []  # Scores

        while trial < n_trials and (trial < min_trials or no_improvement_count < early_stopping_trials):
            trial += 1
            print(f"Trial {trial}/{n_trials}: Testing hyperparams...")

            if len(x_m) == 0:
                # Initial random point
                x_next = np.array([np.random.uniform(low, high) for low, high in bounds])
            else:
                # Optimize acquisition function
                x_next = self._optimize_bayesian(np.array(x_m), np.array(y_m), bounds)

            # Snap to discrete values if needed
            params_dict = {}
            for i, key in enumerate(param_keys):
                value = x_next[i]
                if key in discrete_options:
                    options = discrete_options[key]
                    value = options[np.argmin(np.abs(np.array(options) - value))]
                params_dict[key] = value

            hyperparams = HyperParams(**params_dict)

            print(f"  Testing params: {hyperparams.to_dict()}")

            # Train and evaluate (same as random_search)
            env = LBFEnv(**self.env_params, seed=seed)
            qmix, history = train_qmix(
                env, hyperparams, episodes=training_episodes,
                eval_interval=training_episodes // 2, verbose=False, seed=seed + trial
            )
            policy_func = lambda o: qmix.act(o, training=False)

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

            x_m.append(x_next)
            y_m.append(score)

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

    # NEW: Helper functions for Bayesian optimization (simple DumBO adaptation)
    def _surrogate(self, x_m, y_m, x):
        distance = np.sqrt(np.sum((x - x_m)**2, axis=1))
        i = np.argmin(distance)
        y_ex = y_m[i]
        y_var = distance[i]
        y_se = np.sqrt(y_var)
        return y_ex, y_se

    def _acquisition_function(self, y_ex, y_se):
        return y_ex + y_se / 2  # UCB with reduced exploration

    def _optimize_bayesian(self, x_m, y_m, bounds, n_candidates=1000):
        dim = len(bounds)
        x_candidates = np.zeros((n_candidates, dim))
        for d in range(dim):
            low, high = bounds[d]
            x_candidates[:, d] = np.random.uniform(low, high, n_candidates)
        af = [self._acquisition_function(*self._surrogate(x_m, y_m, xx)) for xx in x_candidates]
        af = np.array(af)
        return x_candidates[np.argmax(af)]