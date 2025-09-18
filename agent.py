import numpy as np
import random
from collections import defaultdict, deque
from hyperparams import HyperParams

class AdvancedIQLAgent:
    def __init__(self, obs_dim, n_actions=5, hyperparams: HyperParams = None):
        """Advanced IQL agent with sophisticated learning mechanisms"""
        self.hyperparams = hyperparams or HyperParams()
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        # Initialize parameters from hyperparams
        self.initial_alpha = self.hyperparams.alpha
        self.alpha = self.hyperparams.alpha
        self.gamma = self.hyperparams.gamma
        self.epsilon = self.hyperparams.epsilon
        self.epsilon_decay = self.hyperparams.epsilon_decay
        self.epsilon_min = self.hyperparams.epsilon_min
        self.alpha_decay = self.hyperparams.alpha_decay
        self.alpha_min = self.hyperparams.alpha_min
        self.reward_scaling = self.hyperparams.reward_scaling
        self.exploration_bonus = self.hyperparams.exploration_bonus

        # Q-table with optimistic initialization
        self.Q = defaultdict(lambda: np.ones(n_actions) * 0.5)

        # Advanced tracking
        self.update_count = 0
        self.state_visits = defaultdict(int)
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        self.recent_rewards = deque(maxlen=100)
        self.recent_td_errors = deque(maxlen=100)

        # Adaptive mechanisms
        self.performance_window = deque(maxlen=50)
        self.last_performance_check = 0

    def discretize_obs(self, obs, precision=2):
        """Enhanced discretization with adaptive precision"""
        discretized = []
        for val in obs:
            if val < -1.5:  # Inactive marker
                discretized.append(-999)
            else:
                discretized.append(round(val, precision))
        return tuple(discretized)

    def act(self, obs, training=True):
        state = self.discretize_obs(obs)
        self.state_visits[state] += 1

        if not training:
            q_values = self.Q[state]
            return int(np.argmax(q_values + np.random.normal(0, 1e-8, self.n_actions)))

        visit_bonus = min(self.exploration_bonus, 10.0 / (1 + self.state_visits[state]))
        effective_epsilon = min(0.95, self.epsilon + visit_bonus)

        if random.random() < effective_epsilon:
            action_counts = [self.state_action_counts[state][a] for a in range(self.n_actions)]
            min_count = min(action_counts) if action_counts else 0
            least_tried = [a for a in range(self.n_actions) if action_counts[a] <= min_count + 1]
            return random.choice(least_tried)

        q_values = self.Q[state]
        return int(np.argmax(q_values + np.random.normal(0, 1e-6, self.n_actions)))

    def update(self, state, action, reward, next_obs, done):
        state = self.discretize_obs(state)
        next_state = self.discretize_obs(next_obs)

        scaled_reward = reward * self.reward_scaling
        self.recent_rewards.append(scaled_reward)

        self.state_action_counts[state][action] += 1

        if done:
            td_target = scaled_reward
        else:
            td_target = scaled_reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - self.Q[state][action]
        self.recent_td_errors.append(abs(td_error))

        adaptive_alpha = self.alpha
        if abs(scaled_reward) > 5:
            adaptive_alpha *= 1.3
        if abs(td_error) > 2:
            adaptive_alpha *= 1.2
        visit_count = self.state_action_counts[state][action]
        if visit_count > 10:
            adaptive_alpha *= (10.0 / visit_count) ** 0.3

        self.Q[state][action] += adaptive_alpha * td_error
        self.update_count += 1

        if self.update_count % 500 == 0:
            self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def adapt_parameters(self, episode_return):
        """Adaptive parameter adjustment based on performance"""
        self.performance_window.append(episode_return)
        if len(self.performance_window) >= 25 and self.update_count - self.last_performance_check > 1000:
            recent_performance = np.mean(list(self.performance_window)[-10:])
            older_performance = np.mean(list(self.performance_window)[-25:-10])
            if recent_performance <= older_performance + 1.0:
                self.epsilon = min(0.8, self.epsilon * 1.1)
                self.alpha = min(0.3, self.alpha * 1.05)
            self.last_performance_check = self.update_count

    def get_stats(self):
        """Comprehensive agent statistics"""
        if not self.Q:
            return {'n_states': 0, 'avg_q': 0, 'max_q': 0, 'min_q': 0, 'updates': 0}
        all_q_values = []
        for state_q in self.Q.values():
            all_q_values.extend(state_q)
        return {
            'n_states': len(self.Q),
            'n_state_actions': sum(len(sa) for sa in self.state_action_counts.values()),
            'avg_q': np.mean(all_q_values),
            'max_q': np.max(all_q_values),
            'min_q': np.min(all_q_values),
            'updates': self.update_count,
            'avg_td_error': np.mean(self.recent_td_errors) if self.recent_td_errors else 0,
            'epsilon': self.epsilon,
            'alpha': self.alpha
        }