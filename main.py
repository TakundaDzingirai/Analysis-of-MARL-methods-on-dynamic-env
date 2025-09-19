import numpy as np
import random
import json
import os
from environment import LBFEnv
from training import train_iql, train_maddpg, train_qmix, evaluate_agent
from optimization import HyperparameterOptimizer
from visualization import plot_optimization_results, plot_training_curves, plot_comparison_results
from hyperparams import HyperParams

def load_optimal_hyperparams(filepath='optimal_hyperparams.json'):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return HyperParams(**data)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Error loading hyperparameters from {filepath}: {e}")
            return None
    return None

def save_optimal_hyperparams(hyperparams, filepath='optimal_hyperparams.json'):
    try:
        with open(filepath, 'w') as f:
            json.dump(hyperparams.to_dict(), f, indent=2)
        print(f"Optimal hyperparameters saved to '{filepath}'")
    except Exception as e:
        print(f"Could not save hyperparameters to '{filepath}': {e}")

def main():
    print("MARL COMPARISON: IQL vs MADDPG vs QMIX")
    print("=" * 60)

    env_params = {
        'grid_size': 4,
        'n_agents': 2,
        'n_foods': 3,
        'agent_levels': [1, 2],
        'food_levels': [1, 2, 2],
        'max_steps': 40
    }

    env = LBFEnv(**env_params, seed=42)
    print(f"Environment: {env.grid_size}x{env.grid_size} grid")
    print(f"Agents: {env.n_agents} (levels {env.agent_levels})")
    print(f"Foods: {env.n_foods} (levels {env.food_levels})")
    print(f"Max steps: {env.max_steps}")

    print("\nBASELINE EVALUATION (Random Policy)")
    print("-" * 40)

    def random_policy(obs):
        return [np.random.randint(0, 5) for _ in range(len(obs))]

    eval_env_params = {k: v for k, v in env.__dict__.items() if k in env_params}
    try:
        baseline_stats = evaluate_agent(eval_env_params, random_policy, n_episodes=500, seed=42)
        print(f"Random baseline - Return: {baseline_stats['mean_return']:.2f}, "
              f"Foods: {baseline_stats['mean_foods']:.2f}, "
              f"Success: {baseline_stats['success_rate']:.2f}")
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
        baseline_stats = {'mean_return': 0.0, 'mean_foods': 0.0, 'success_rate': 0.0}

    optimizer = HyperparameterOptimizer(env_params)

    # IQL Optimization (Load from file, no search)
    print(f"\nIQL HYPERPARAMETER LOADING")
    print("-" * 40)
    iql_hyperparams_file = 'optimal_hyperparams.json'
    iql_params = load_optimal_hyperparams(iql_hyperparams_file)

    if iql_params is None:
        print(f"No valid IQL hyperparameters found in '{iql_hyperparams_file}'. Using defaults...")
        iql_params = HyperParams()
        iql_results = []
    else:
        print(f"Loaded IQL hyperparameters from '{iql_hyperparams_file}':")
        print(iql_params.to_dict())
        iql_results = []

    print(f"\nIQL TRAINING")
    print("-" * 40)
    env = LBFEnv(**env_params, seed=42)
    try:
        iql_agents, iql_history = train_iql(
            env, iql_params, episodes=2000, eval_interval=200, verbose=True, seed=42
        )
    except Exception as e:
        print(f"IQL training failed: {e}")
        return

    print(f"\nIQL FINAL EVALUATION")
    print("-" * 40)
    def iql_policy(obs):
        return [iql_agents[i].act(obs[i], training=False) for i in range(len(obs))]
    try:
        iql_stats = evaluate_agent(eval_env_params, iql_policy, n_episodes=500, seed=42)
        print(f"IQL - Return: {iql_stats['mean_return']:.2f}, Foods: {iql_stats['mean_foods']:.2f}, Success: {iql_stats['success_rate']:.2f}")
    except Exception as e:
        print(f"IQL evaluation failed: {e}")
        iql_stats = {'mean_return': 0.0, 'mean_foods': 0.0, 'success_rate': 0.0}

    # MADDPG Optimization
    print(f"\nMADDPG HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    maddpg_hyperparams_file = 'optimal_maddpg_hyperparams.json'
    maddpg_params = load_optimal_hyperparams(maddpg_hyperparams_file)

    if maddpg_params is None:
        print("No valid MADDPG hyperparameters found. Running random search...")
        try:
            maddpg_params, maddpg_results = optimizer.random_search(
                model_type='maddpg', n_trials=60, training_episodes=1000, seed=42, early_stopping_trials=10, min_trials=10
            )
            save_optimal_hyperparams(maddpg_params, maddpg_hyperparams_file)
        except Exception as e:
            print(f"MADDPG optimization failed: {e}")
            maddpg_params = HyperParams()
            maddpg_results = []
    else:
        print(f"Loaded MADDPG hyperparameters from '{maddpg_hyperparams_file}':")
        print(maddpg_params.to_dict())
        maddpg_results = []

    print(f"\nMADDPG TRAINING")
    print("-" * 40)
    env = LBFEnv(**env_params, seed=42)
    try:
        maddpg, maddpg_history = train_maddpg(
            env, maddpg_params, episodes=2000, eval_interval=200, verbose=True, seed=42
        )
    except Exception as e:
        print(f"MADDPG training failed: {e}")
        return

    print(f"\nMADDPG FINAL EVALUATION")
    print("-" * 40)
    try:
        maddpg_stats = evaluate_agent(eval_env_params, lambda o: maddpg.act(o, training=False), n_episodes=500, seed=42)
        print(f"MADDPG - Return: {maddpg_stats['mean_return']:.2f}, Foods: {maddpg_stats['mean_foods']:.2f}, Success: {maddpg_stats['success_rate']:.2f}")
    except Exception as e:
        print(f"MADDPG evaluation failed: {e}")
        maddpg_stats = {'mean_return': 0.0, 'mean_foods': 0.0, 'success_rate': 0.0}

    # QMIX Optimization
    print(f"\nQMIX HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    qmix_hyperparams_file = 'optimal_qmix_hyperparams.json'
    qmix_params = load_optimal_hyperparams(qmix_hyperparams_file)

    if qmix_params is None:
        print("No valid QMIX hyperparameters found. Running random search...")
        try:
            qmix_params, qmix_results = optimizer.random_search(
                model_type='qmix', n_trials=60, training_episodes=1000, seed=42, early_stopping_trials=10, min_trials=10
            )
            save_optimal_hyperparams(qmix_params, qmix_hyperparams_file)
        except Exception as e:
            print(f"QMIX optimization failed: {e}")
            # qmix_params = HyperParams(lr_actor=0.001, lr_critic=0.001, gamma=0.99, tau=0.01)
            # To this:
            qmix_params = HyperParams(lr=0.001, gamma=0.99, tau=0.01, epsilon=0.1, epsilon_decay=0.995,
                                      epsilon_min=0.01)
            # Default QMIX params
            qmix_results = []
    else:
        print(f"Loaded QMIX hyperparameters from '{qmix_hyperparams_file}':")
        print(qmix_params.to_dict())
        qmix_results = []

    print(f"\nQMIX TRAINING")
    print("-" * 40)
    env = LBFEnv(**env_params, seed=42)
    try:
        qmix, qmix_history = train_qmix(
            env, qmix_params, episodes=2000, eval_interval=200, verbose=True, seed=42
        )
    except Exception as e:
        print(f"QMIX training failed: {e}")
        return

    print(f"\nQMIX FINAL EVALUATION")
    print("-" * 40)
    try:
        qmix_stats = evaluate_agent(eval_env_params, lambda o: qmix.act(o, training=False), n_episodes=500, seed=42)
        print(f"QMIX - Return: {qmix_stats['mean_return']:.2f}, Foods: {qmix_stats['mean_foods']:.2f}, Success: {qmix_stats['success_rate']:.2f}")
    except Exception as e:
        print(f"QMIX evaluation failed: {e}")
        qmix_stats = {'mean_return': 0.0, 'mean_foods': 0.0, 'success_rate': 0.0}

    print(f"\nRESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} {'Random':<12} {'IQL':<12} {'MADDPG':<12} {'QMIX':<12} {'IQL Imp.':<12} {'MADDPG Imp.':<12} {'QMIX Imp.':<12}")
    print("-" * 70)

    metrics = ['mean_return', 'mean_foods', 'success_rate']
    labels = ['Average Return', 'Foods Collected', 'Success Rate']
    for metric, label in zip(metrics, labels):
        baseline_val = baseline_stats[metric]
        iql_val = iql_stats[metric]
        maddpg_val = maddpg_stats[metric]
        qmix_val = qmix_stats[metric]
        iql_imp = iql_val - baseline_val
        maddpg_imp = maddpg_val - baseline_val
        qmix_imp = qmix_val - baseline_val
        print(f"{label:<20} {baseline_val:<12.3f} {iql_val:<12.3f} {maddpg_val:<12.3f} {qmix_val:<12.3f} {iql_imp:+.3f} {maddpg_imp:+.3f} {qmix_imp:+.3f}")

    print(f"\nGENERATING VISUALIZATIONS...")
    if iql_results:
        plot_optimization_results(iql_results, "IQL Hyperparameter Optimization")
    if maddpg_results:
        plot_optimization_results(maddpg_results, "MADDPG Hyperparameter Optimization")
    if qmix_results:
        plot_optimization_results(qmix_results, "QMIX Hyperparameter Optimization")
    plot_training_curves(iql_history, "IQL Training Curves")
    plot_training_curves(maddpg_history, "MADDPG Training Curves")
    plot_training_curves(qmix_history, "QMIX Training Curves")

    plot_comparison_results(baseline_stats, iql_stats, maddpg_stats, qmix_stats, 'marl_comparison.png')
    results_summary = {
        'baseline_stats': baseline_stats,
        'iql_stats': iql_stats,
        'maddpg_stats': maddpg_stats,
        'qmix_stats': qmix_stats,
        'iql_improvement': {m: iql_stats[m] - baseline_stats[m] for m in metrics},
        'maddpg_improvement': {m: maddpg_stats[m] - baseline_stats[m] for m in metrics},
        'qmix_improvement': {m: qmix_stats[m] - baseline_stats[m] for m in metrics}
    }

    try:
        with open('marl_comparison_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"Results saved to 'marl_comparison_results.json'")
    except Exception as e:
        print(f"Could not save results file: {e}")

    print(f"\nCOMPARISON COMPLETE!")
    best_method = max([('IQL', iql_stats['mean_return']), ('MADDPG', maddpg_stats['mean_return']), ('QMIX', qmix_stats['mean_return'])], key=lambda x: x[1])[0]
    print(f"{best_method} performed best!")

if __name__ == "__main__":
    main()