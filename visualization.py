import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def plot_optimization_results(results, title="Hyperparameter Optimization Results"):
    """Visualize optimization results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    scores = [r['score'] for r in results]
    returns = [r['stats']['mean_return'] for r in results]
    success_rates = [r['stats']['success_rate'] for r in results]
    foods = [r['stats']['mean_foods'] for r in results]

    axes[0, 0].plot(scores, 'b-', alpha=0.7)
    axes[0, 0].scatter(range(len(scores)), scores, alpha=0.6)
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Composite Score')
    axes[0, 0].set_title('Score Progression')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(returns, bins=20, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Mean Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Returns Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(returns, success_rates, alpha=0.6, c=scores, cmap='viridis')
    axes[1, 0].set_xlabel('Mean Return')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title('Success Rate vs Return')
    axes[1, 0].grid(True, alpha=0.3)

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

    axes[0].plot(episodes, history['returns'], 'b-', linewidth=2, label='Mean Return')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Return')
    axes[0].set_title('Learning Curve: Returns')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(episodes, history['foods'], 'g-', linewidth=2, label='Foods Collected')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Foods Collected')
    axes[1].set_title('Learning Curve: Food Collection')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(episodes, history['success_rates'], 'r-', linewidth=2, label='Success Rate')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_title('Learning Curve: Success Rate')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()