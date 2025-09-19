import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_optimization_results(results, title="Hyperparameter Optimization Results"):
    """Visualize optimization results"""
    if not results:
        print(f"No optimization results to plot for {title}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    scores = [r['score'] for r in results]
    returns = [r['stats']['mean_return'] for r in results]
    success_rates = [r['stats']['success_rate'] for r in results]
    foods = [r['stats']['mean_foods'] for r in results]

    # Score progression
    axes[0, 0].plot(scores, 'b-', alpha=0.7)
    axes[0, 0].scatter(range(len(scores)), scores, alpha=0.6)
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Composite Score')
    axes[0, 0].set_title('Score Progression')
    axes[0, 0].grid(True, alpha=0.3)

    # Returns distribution
    axes[0, 1].hist(returns, bins=min(20, len(returns)), alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Mean Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Returns Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Success rate vs return scatter
    scatter = axes[1, 0].scatter(returns, success_rates, alpha=0.6, c=scores, cmap='viridis')
    axes[1, 0].set_xlabel('Mean Return')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title('Success Rate vs Return')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Score')

    # Alpha vs score
    alphas = [r['hyperparams']['alpha'] for r in results]
    axes[1, 1].scatter(alphas, scores, alpha=0.6, c='red')
    axes[1, 1].set_xlabel('Alpha (Learning Rate)')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Alpha vs Score')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves(history, title="Training Progress"):
    """Plot training curves - Fixed to use correct key names"""
    if not history or not history.get('episode'):
        print(f"No training history to plot for {title}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Use 'episode' (singular) as stored by training functions
    episodes = history['episode']

    # Plot 1: Returns over time
    if 'mean_return' in history:
        axes[0, 0].plot(episodes, history['mean_return'], 'b-', linewidth=2, label='Mean Return')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Return')
        axes[0, 0].set_title('Learning Curve: Returns')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

    # Plot 2: Food collection over time
    if 'mean_foods' in history:
        axes[0, 1].plot(episodes, history['mean_foods'], 'g-', linewidth=2, label='Foods Collected')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Foods Collected (Proportion)')
        axes[0, 1].set_title('Learning Curve: Food Collection')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

    # Plot 3: Success rate over time
    if 'success_rate' in history:
        axes[1, 0].plot(episodes, history['success_rate'], 'r-', linewidth=2, label='Success Rate')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Learning Curve: Success Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

    # Plot 4: Algorithm-specific metrics
    if 'avg_td_error' in history:
        # IQL-specific: TD Error
        axes[1, 1].plot(episodes, history['avg_td_error'], 'm-', linewidth=2, label='Avg TD Error')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('TD Error')
        axes[1, 1].set_title('Learning Curve: TD Error (IQL)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    elif 'avg_critic_loss' in history:
        # MADDPG-specific: Critic Loss
        axes[1, 1].plot(episodes, history['avg_critic_loss'], 'orange', linewidth=2, label='Avg Critic Loss')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Critic Loss')
        axes[1, 1].set_title('Learning Curve: Critic Loss (MADDPG)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        # If no specific metric, show combined performance
        if 'mean_return' in history and 'success_rate' in history:
            ax2 = axes[1, 1]
            ax3 = ax2.twinx()

            line1 = ax2.plot(episodes, history['mean_return'], 'b-', linewidth=2, label='Mean Return')
            line2 = ax3.plot(episodes, history['success_rate'], 'r-', linewidth=2, label='Success Rate')

            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Mean Return', color='b')
            ax3.set_ylabel('Success Rate', color='r')
            ax2.set_title('Combined Performance')
            ax2.grid(True, alpha=0.3)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison_results(baseline_stats, iql_stats, maddpg_stats, save_path=None):
    """Create a comprehensive comparison visualization"""
    metrics = ['mean_return', 'mean_foods', 'success_rate']
    labels = ['Average Return', 'Foods Collected', 'Success Rate']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [baseline_stats[metric], iql_stats[metric], maddpg_stats[metric]]
        colors = ['gray', 'blue', 'red']
        method_labels = ['Random Baseline', 'IQL', 'MADDPG']

        bars = axes[i].bar(method_labels, values, color=colors, alpha=0.7)
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label} Comparison')
        axes[i].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{value:.3f}', ha='center', va='bottom')

    plt.suptitle('MARL Methods Comparison', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()