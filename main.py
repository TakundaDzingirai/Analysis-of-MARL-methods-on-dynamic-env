import numpy as np
import random
import json
from environment import LBFEnv
from training import evaluate_agent, train_advanced_iql
from optimization import HyperparameterOptimizer
from visualization import plot_optimization_results, plot_training_curves

def main():
    print("ADVANCED IQL WITH HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    env_params = {
        'grid_size': 4,
        'n_agents': 2,
        'n_foods': 3,
        'agent_levels': [1, 2],
        'food_levels': [1, 2, 2],
        'max_steps': 50
    }

    env = LBFEnv(**env_params, seed=42)
    print(f"Environment: {env.grid_size}x{env.grid_size} grid")
    print(f"Agents: {env.n_agents} (levels {env.agent_levels})")
    print(f"Foods: {env.n_foods} (levels {env.food_levels})")
    print(f"Max steps: {env.max_steps}")

    print("\nBASELINE EVALUATION")
    print("-" * 40)

    def random_policy(obs):
        return [np.random.randint(0, 5) for _ in range(len(obs))]

    baseline_stats = evaluate_agent(env_params, random_policy, n_episodes=200, seed=42)
    print(f"Random baseline - Return: {baseline_stats['mean_return']:.2f}, "
          f"Foods: {baseline_stats['mean_foods']:.2f}, "
          f"Success: {baseline_stats['success_rate']:.2f}")

    print(f"\nHYPERPARAMETER OPTIMIZATION")
    print("-" * 40)

    optimizer = HyperparameterOptimizer(env_params)
    use_random_search = True

    if use_random_search:
        best_params, results = optimizer.random_search(n_trials=30, training_episodes=800, seed=42)
        method_name = "Random Search"
    else:
        best_params, results = optimizer.grid_search(training_episodes=600)
        method_name = "Grid Search"

    print(f"\nTOP 5 RESULTS FROM {method_name.upper()}:")
    print("-" * 60)
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
    for i, result in enumerate(sorted_results):
        print(f"Rank {i + 1}: Score={result['score']:.2f}")
        print(f"  Return: {result['stats']['mean_return']:.2f}±{result['stats']['std_return']:.2f}")
        print(f"  Foods: {result['stats']['mean_foods']:.2f}, Success: {result['stats']['success_rate']:.2f}")
        print(f"  Hyperparams: {result['hyperparams']}")
        print()

    print(f"TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("-" * 40)
    print(f"Using best hyperparams: {best_params.to_dict()}")

    env = LBFEnv(**env_params, seed=42)
    best_agents, best_history = train_advanced_iql(
        env, best_params, episodes=5000, eval_interval=200, verbose=True, seed=42
    )

    print(f"\nFINAL EVALUATION")
    print("-" * 40)

    def best_policy(obs):
        return [best_agents[i].act(obs[i], training=False) for i in range(len(obs))]

    final_stats = evaluate_agent(env_params, best_policy, n_episodes=500, seed=42)

    print(f"\nRESULTS COMPARISON")
    print("=" * 62)
    print(f"{'Metric':<20} {'Random':<12} {'Optimized IQL':<15} {'Improvement':<15}")
    print("-" * 62)

    metrics = ['mean_return', 'mean_foods', 'success_rate']
    labels = ['Average Return', 'Foods Collected', 'Success Rate']
    for metric, label in zip(metrics, labels):
        baseline_val = baseline_stats[metric]
        final_val = final_stats[metric]
        improvement = final_val - baseline_val
        print(f"{label:<20} {baseline_val:<12.3f} {final_val:<15.3f} {improvement:+.3f}")

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

    improvement_threshold = 10.0
    success = (final_stats['mean_return'] - baseline_stats['mean_return']) > improvement_threshold
    print(f"\n{'🎉 OPTIMIZATION SUCCESS!' if success else '⚠️ NEEDS MORE WORK'}")
    if success:
        print(f"Optimized IQL significantly outperformed random policy!")
        print(f"Return improvement: {final_stats['mean_return'] - baseline_stats['mean_return']:.2f}")
    else:
        print(f"Try different hyperparameter ranges or longer training.")

    print(f"\nGENERATING VISUALIZATIONS...")
    plot_optimization_results(results, f"Hyperparameter {method_name} Results")
    plot_training_curves(best_history, "Training with Optimized Hyperparameters")

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

    try:
        with open('iql_optimization_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"Results saved to 'iql_optimization_results.json'")
    except Exception as e:
        print(f"Could not save results file: {e}")

    print(f"\nOPTIMIZATION COMPLETE!")
    print(f"Best configuration found and validated.")
    print(f"Use the best hyperparameters for future training runs.")

if __name__ == "__main__":
    main()