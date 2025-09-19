import numpy as np
import random
import json
import os
from environment import LBFEnv
from training import train_iql, train_maddpg, evaluate_agent
from optimization import HyperparameterOptimizer
from visualization import plot_optimization_results, plot_training_curves
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
    print("MARL COMPARISON: IQL vs MADDPG")
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

    baseline_stats = evaluate_agent(env_params, random_policy, n_episodes=500, seed=42)
    print(f"Random baseline - Return: {baseline_stats['mean_return']:.2f}, "
          f"Foods: {baseline_stats['mean_foods']:.2f}, "
          f"Success: {baseline_stats['success_rate']:.2f}")

    optimizer = HyperparameterOptimizer(env_params)

    # IQL Optimization
    print(f"\nIQL HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    iql_hyperparams_file = 'optimal_iql_hyperparams.json'
    iql_params = load_optimal_hyperparams(iql_hyperparams_file)

    if iql_params is None:
        print("No valid IQL hyperparameters found. Running random search...")
        iql_params, iql_results = optimizer.random_search(
            model_type='iql', n_trials=30, training_episodes=800, seed=42, early_stopping_trials=15, min_trials=20
        )
        save_optimal_hyperparams(iql_params, iql_hyperparams_file)
    else:
        print(f"Loaded IQL hyperparameters from '{iql_hyperparams_file}':")
        print(iql_params.to_dict())
        iql_results = []

    print(f"\nIQL TRAINING")
    print("-" * 40)
    env = LBFEnv(**env_params, seed=42)
    iql_agents, iql_history = train_iql(
        env, iql_params, episodes=1000, eval_interval=200, verbose=True, seed=42
    )

    print(f"\nIQL FINAL EVALUATION")
    print("-" * 40)
    def iql_policy(obs):
        return [iql_agents[i].act(obs[i], training=False) for i in range(len(obs))]
    iql_stats = evaluate_agent(env_params, iql_policy, n_episodes=500, seed=42)
    print(f"IQL - Return: {iql_stats['mean_return']:.2f}, Foods: {iql_stats['mean_foods']:.2f}, Success: {iql_stats['success_rate']:.2f}")

    # MADDPG Optimization
    print(f"\nMADDPG HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    maddpg_hyperparams_file = 'optimal_maddpg_hyperparams.json'
    maddpg_params = load_optimal_hyperparams(maddpg_hyperparams_file)

    if maddpg_params is None:
        print("No valid MADDPG hyperparameters found. Running random search...")
        maddpg_params, maddpg_results = optimizer.random_search(
            model_type='maddpg', n_trials=20, training_episodes=600, seed=42, early_stopping_trials=10, min_trials=15
        )
        save_optimal_hyperparams(maddpg_params, maddpg_hyperparams_file)
    else:
        print(f"Loaded MADDPG hyperparameters from '{maddpg_hyperparams_file}':")
        print(maddpg_params.to_dict())
        maddpg_results = []

    print(f"\nMADDPG TRAINING")
    print("-" * 40)
    env = LBFEnv(**env_params, seed=42)
    maddpg, maddpg_history = train_maddpg(
        env, maddpg_params, episodes=1000, eval_interval=200, verbose=True, seed=42
    )

    print(f"\nMADDPG FINAL EVALUATION")
    print("-" * 40)
    maddpg_stats = evaluate_agent(env_params, lambda o: maddpg.act(o, training=False), n_episodes=500, seed=42)
    print(f"MADDPG - Return: {maddpg_stats['mean_return']:.2f}, Foods: {maddpg_stats['mean_foods']:.2f}, Success: {maddpg_stats['success_rate']:.2f}")

    print(f"\nRESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} {'Random':<12} {'IQL':<12} {'MADDPG':<12} {'IQL Imp.':<12} {'MADDPG Imp.':<12}")
    print("-" * 70)

    metrics = ['mean_return', 'mean_foods', 'success_rate']
    labels = ['Average Return', 'Foods Collected', 'Success Rate']
    for metric, label in zip(metrics, labels):
        baseline_val = baseline_stats[metric]
        iql_val = iql_stats[metric]
        maddpg_val = maddpg_stats[metric]
        iql_imp = iql_val - baseline_val
        maddpg_imp = maddpg_val - baseline_val
        print(f"{label:<20} {baseline_val:<12.3f} {iql_val:<12.3f} {maddpg_val:<12.3f} {iql_imp:+.3f} {maddpg_imp:+.3f}")

    print(f"\nGENERATING VISUALIZATIONS...")
    if iql_results:
        plot_optimization_results(iql_results, "IQL Hyperparameter Optimization")
    if maddpg_results:
        plot_optimization_results(maddpg_results, "MADDPG Hyperparameter Optimization")
    plot_training_curves(iql_history, "IQL Training Curves")
    plot_training_curves(maddpg_history, "MADDPG Training Curves")

    results_summary = {
        'baseline_stats': baseline_stats,
        'iql_stats': iql_stats,
        'maddpg_stats': maddpg_stats,
        'iql_improvement': {m: iql_stats[m] - baseline_stats[m] for m in metrics},
        'maddpg_improvement': {m: maddpg_stats[m] - baseline_stats[m] for m in metrics}
    }

    try:
        with open('marl_comparison_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"Results saved to 'marl_comparison_results.json'")
    except Exception as e:
        print(f"Could not save results file: {e}")

    print(f"\nCOMPARISON COMPLETE!")
    if maddpg_stats['mean_return'] > iql_stats['mean_return']:
        print("MADDPG outperformed IQL!")
    else:
        print("IQL performed better or equally well.")

if __name__ == "__main__":
    main()