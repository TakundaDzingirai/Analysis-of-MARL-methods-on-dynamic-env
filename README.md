# Multi-Agent Reinforcement Learning Comparison: IQL vs MADDPG vs QMIX

A comprehensive evaluation of three Multi-Agent Reinforcement Learning algorithms in a Level-Based Foraging environment, with systematic hyperparameter optimization and performance analysis.

## Overview

This project implements and compares three MARL algorithms:
- **Independent Q-Learning (IQL)**: Table-based independent learning approach
- **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient with centralized training
- **QMIX**: Value decomposition method with mixing networks for centralized training

The algorithms are evaluated in a custom Level-Based Foraging (LBF) environment where agents must cooperatively collect food items based on level requirements.

## Environment

**Level-Based Foraging (LBF)**
- **Grid Size**: 4x4 (configurable)
- **Agents**: 2 agents with levels [1, 2]
- **Foods**: 3 food items with levels [1, 2, 2]
- **Actions**: {up, down, left, right, stay}
- **Cooperation**: Agents must combine levels to collect food items
- **Episode Length**: Maximum 40 steps

## Project Structure

```
├── main.py                    # Main experiment runner with hyperparameter optimization
├── environment.py             # Level-Based Foraging environment implementation
├── agent.py                   # IQL agent with advanced features
├── maddpg.py                  # MADDPG implementation with actor-critic networks
├── qmix.py                    # QMIX implementation with mixing networks
├── training.py                # Training functions for all algorithms
├── optimization.py            # Hyperparameter optimization framework
├── hyperparams.py             # Hyperparameter configuration classes
├── visualization.py           # Plotting and visualization utilities
├── test_new_env.py           # Generalization testing on larger environments
├── optimal_hyperparams.json          # Optimized IQL hyperparameters
├── optimal_maddpg_hyperparams.json   # Optimized MADDPG hyperparameters
├── optimal_qmix_hyperparams.json     # Optimized QMIX hyperparameters
└── marl_comparison_results.json      # Final evaluation results
```

## Installation

### Requirements
- Python 3.8+
- NumPy
- PyTorch
- Matplotlib
- scikit-optimize (for Bayesian optimization)

### Setup
```bash
git clone <repository-url>
cd marl-comparison
pip install -r requirements.txt
```

## Usage

### Quick Start
Run the complete comparison with optimized hyperparameters:
```bash
python main.py
```

This will:
1. Load optimal hyperparameters from JSON files (or run optimization if not found)
2. Train all three algorithms for 2000 episodes
3. Evaluate performance over 500 test episodes
4. Generate comparison visualizations
5. Save results to `marl_comparison_results.json`

### Individual Algorithm Training
```python
from environment import LBFEnv
from training import train_iql, train_maddpg, train_qmix
from hyperparams import HyperParams

# Create environment
env = LBFEnv(grid_size=4, n_agents=2, n_foods=3, 
             agent_levels=[1, 2], food_levels=[1, 2, 2])

# Train IQL
hyperparams = HyperParams()
iql_agents, history = train_iql(env, hyperparams, episodes=2000)

# Train MADDPG  
maddpg_agent, history = train_maddpg(env, hyperparams, episodes=2000)

# Train QMIX
qmix_agent, history = train_qmix(env, hyperparams, episodes=2000)
```

### Hyperparameter Optimization
```python
from optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(env_params)

# Random search for MADDPG
maddpg_params, results = optimizer.random_search(
    model_type='maddpg', n_trials=60, training_episodes=1000
)

# Bayesian optimization for QMIX
qmix_params, results = optimizer.bayesian_search(
    model_type='qmix', n_trials=60, training_episodes=1000
)
```

### Testing on Larger Environments
```bash
python test_new_env.py
```
Tests the trained models' generalization to an 8x8 environment.

## Key Features

### Advanced IQL Implementation
- Optimistic Q-table initialization
- Adaptive learning rates and exploration
- State visit counting for exploration bonuses
- Performance-based parameter adjustment

### Sophisticated MADDPG
- Centralized training with decentralized execution
- Shared experience replay buffer
- Soft target network updates
- Individual actor-critic networks per agent

### Enhanced QMIX
- Individual Q-networks with centralized mixing
- Hypernetwork-based mixing network
- Dynamic weight generation based on global state
- Value decomposition for credit assignment

### Hyperparameter Optimization
- **IQL**: Pre-optimized parameters loaded from JSON
- **MADDPG**: Random search optimization (60 trials)
- **QMIX**: Bayesian optimization (60 trials)
- Early stopping and minimum trial requirements
- Automatic parameter persistence

## Results

### Performance Summary (4x4 Environment)
| Algorithm | Mean Return | Foods Collected | Success Rate | Improvement over Random |
|-----------|-------------|-----------------|--------------|-------------------------|
| Random    | 53.06       | 0.70           | 0.35         | Baseline                |
| IQL       | 60.97       | 0.76           | 0.58         | +7.91 / +0.06 / +0.23   |
| MADDPG    | 69.05       | 0.996          | 0.996        | +15.99 / +0.29 / +0.65  |
| QMIX      | 69.91       | 0.96           | 0.93         | +16.85 / +0.26 / +0.58  |

### Key Findings
- **QMIX** achieved the highest mean return (69.91) due to effective centralized training
- **MADDPG** demonstrated the most consistent performance (success rate: 0.996)
- **IQL** showed moderate improvement but was limited by independent learning
- Systematic hyperparameter optimization was crucial for fair comparison

## Configuration

### Environment Parameters
```python
env_params = {
    'grid_size': 4,           # Grid dimensions (4x4)
    'n_agents': 2,            # Number of agents
    'n_foods': 3,             # Number of food items
    'agent_levels': [1, 2],   # Agent capability levels
    'food_levels': [1, 2, 2], # Food requirement levels
    'max_steps': 40,          # Maximum episode length
    'seed': 42                # Random seed for reproducibility
}
```

### Hyperparameter Spaces
The optimization searches across:
- Learning rates (0.0001 - 0.01)
- Discount factors (0.9 - 0.999)
- Exploration parameters (epsilon, decay rates)
- Network architectures (hidden dimensions: 32-128)
- Training dynamics (batch sizes, update frequencies)

## Visualization

The project generates several visualizations:
- Training curves for all algorithms
- Performance comparison bar charts
- Hyperparameter optimization results
- Episode return distributions

## Applications

This framework is designed to evaluate MARL algorithms for:
- **Traffic Control Systems**: Coordinated signal optimization
- **Resource Management**: Distributed allocation problems  
- **Robotics**: Multi-robot coordination tasks
- **Game Theory**: Cooperative and competitive scenarios

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Work

- [ ] Add more MARL algorithms (A3C, PPO variants)
- [ ] Implement continuous action spaces
- [ ] Add communication channels between agents
- [ ] Extend to more complex environments
- [ ] Integration with OpenAI Gym environments

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Albrecht, S. V., et al. (2024). *Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents*
- Lowe, R., et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
- Rashid, T., et al. (2018). QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning

## Citation

```bibtex
@misc{marl-comparison-2024,
  title={Multi-Agent Reinforcement Learning Comparison: IQL vs MADDPG vs QMIX},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```
