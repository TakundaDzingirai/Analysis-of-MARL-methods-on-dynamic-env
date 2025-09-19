import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Discrete actions
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=64):
        super(Critic, self).__init__()
        total_state_dim = state_dim * n_agents
        total_action_dim = action_dim * n_agents
        self.network = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, states, actions, rewards, next_states, dones):
        """Store joint multi-agent experience"""
        self.buffer.append((states, actions, rewards, next_states, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),  # [batch_size, n_agents, state_dim]
            np.array(actions),  # [batch_size, n_agents]
            np.array(rewards),  # [batch_size, n_agents]
            np.array(next_states),  # [batch_size, n_agents, state_dim]
            np.array(dones)  # [batch_size, n_agents]
        )

    def __len__(self):
        return len(self.buffer)


class MADDPGAgent:
    def __init__(self, state_dim, action_dim, n_agents, agent_id, lr_actor=0.001, lr_critic=0.002, gamma=0.99,
                 tau=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau

        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim).float()
        self.actor_target = Actor(state_dim, action_dim).float()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim, n_agents).float()
        self.critic_target = Critic(state_dim, action_dim, n_agents).float()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.critic_loss_history = []

    def act(self, state, training=True, exploration_noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state).squeeze(0).numpy()

        if training:
            # Add exploration noise to probabilities
            action_probs = action_probs + np.random.normal(0, exploration_noise, action_probs.shape)
            action_probs = np.maximum(action_probs, 0)  # Ensure non-negative
            action_probs = action_probs / np.sum(action_probs)  # Renormalize
            action = np.random.choice(self.action_dim, p=action_probs)
        else:
            action = np.argmax(action_probs)
        return action

    def update(self, batch_size, replay_buffer, all_agents):
        if len(replay_buffer) < batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Convert to tensors with correct shapes
        states = torch.FloatTensor(states)  # [batch_size, n_agents, state_dim]
        actions = torch.LongTensor(actions)  # [batch_size, n_agents]
        rewards = torch.FloatTensor(rewards)  # [batch_size, n_agents]
        next_states = torch.FloatTensor(next_states)  # [batch_size, n_agents, state_dim]
        dones = torch.FloatTensor(dones)  # [batch_size, n_agents]

        batch_size_actual = states.shape[0]

        # Reshape for critic input
        states_flat = states.view(batch_size_actual, -1)  # [batch_size, n_agents * state_dim]
        next_states_flat = next_states.view(batch_size_actual, -1)

        # Convert actions to one-hot encoding
        actions_one_hot = torch.zeros(batch_size_actual, self.n_agents * self.action_dim)
        for i in range(self.n_agents):
            actions_one_hot.scatter_(1,
                                     (i * self.action_dim + actions[:, i]).unsqueeze(1),
                                     1.0)

        # Compute target Q-values
        with torch.no_grad():
            # Get next actions from all target actors
            next_actions_list = []
            for i, agent in enumerate(all_agents):
                next_action_probs = agent.actor_target(next_states[:, i])
                next_actions = torch.argmax(next_action_probs, dim=-1)
                next_actions_list.append(next_actions)

            next_actions = torch.stack(next_actions_list, dim=1)  # [batch_size, n_agents]

            # Convert next actions to one-hot
            next_actions_one_hot = torch.zeros(batch_size_actual, self.n_agents * self.action_dim)
            for i in range(self.n_agents):
                next_actions_one_hot.scatter_(1,
                                              (i * self.action_dim + next_actions[:, i]).unsqueeze(1),
                                              1.0)

            target_q = self.critic_target(next_states_flat, next_actions_one_hot)
            target_q = rewards[:, self.agent_id].unsqueeze(1) + \
                       self.gamma * (1 - dones[:, self.agent_id].unsqueeze(1)) * target_q

        # Current Q-values
        current_q = self.critic(states_flat, actions_one_hot)

        # Critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_loss_history.append(critic_loss.item())

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Actor loss - compute predicted actions for all agents
        predicted_actions_list = []
        for i, agent in enumerate(all_agents):
            if i == self.agent_id:
                predicted_actions_list.append(self.actor(states[:, i]))
            else:
                with torch.no_grad():
                    predicted_actions_list.append(agent.actor(states[:, i]))

        # Convert predicted actions to one-hot (using soft one-hot for differentiability)
        predicted_actions_one_hot = torch.zeros(batch_size_actual, self.n_agents * self.action_dim)
        for i in range(self.n_agents):
            start_idx = i * self.action_dim
            end_idx = (i + 1) * self.action_dim
            predicted_actions_one_hot[:, start_idx:end_idx] = predicted_actions_list[i]

        # Actor loss
        actor_loss = -self.critic(states_flat, predicted_actions_one_hot).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return critic_loss.item()

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_stats(self):
        return {
            'avg_critic_loss': np.mean(self.critic_loss_history) if self.critic_loss_history else 0.0,
            'recent_critic_loss': self.critic_loss_history[-10:] if self.critic_loss_history else []
        }


class MADDPG:
    def __init__(self, state_dim, action_dim, n_agents, hyperparams):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create agents
        self.agents = [
            MADDPGAgent(state_dim, action_dim, n_agents, agent_id=i,
                        lr_actor=hyperparams.lr_actor,
                        lr_critic=hyperparams.lr_critic,
                        gamma=hyperparams.gamma,
                        tau=hyperparams.tau)
            for i in range(n_agents)
        ]

        # Shared replay buffer for all agents
        self.replay_buffer = ReplayBuffer(capacity=50000)

    def act(self, states, training=True):
        """Get actions for all agents"""
        return [self.agents[i].act(states[i], training) for i in range(self.n_agents)]

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Add joint experience to replay buffer"""
        self.replay_buffer.push(states, actions, rewards, next_states, dones)

    def update(self, batch_size=64):
        """Update all agents"""
        if len(self.replay_buffer) < batch_size:
            return 0.0

        critic_losses = []
        for agent in self.agents:
            loss = agent.update(batch_size, self.replay_buffer, self.agents)
            if loss > 0:
                critic_losses.append(loss)

        return np.mean(critic_losses) if critic_losses else 0.0

    def get_stats(self):
        """Get statistics from all agents"""
        return [agent.get_stats() for agent in self.agents]