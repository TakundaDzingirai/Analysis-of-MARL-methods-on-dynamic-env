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

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, n_agents, lr_actor=0.001, lr_critic=0.002, gamma=0.99, tau=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
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

        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        self.critic_loss_history = []

    def act(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state).squeeze(0).numpy()
        if training:
            action = np.random.choice(self.action_dim, p=action_probs)
        else:
            action = np.argmax(action_probs)
        return action

    def update(self, batch_size=64, all_agents=None):
        if len(self.replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Prepare inputs
        state_batch = states.view(batch_size, self.n_agents, self.state_dim)
        action_batch = actions.view(batch_size, self.n_agents)
        reward_batch = rewards.view(batch_size, self.n_agents)
        next_state_batch = next_states.view(batch_size, self.n_agents, self.state_dim)
        done_batch = dones.view(batch_size, self.n_agents)

        # Reshape for critic
        state_flat = state_batch.view(batch_size, -1)
        action_one_hot = torch.zeros(batch_size, self.n_agents * self.action_dim)
        for i in range(self.n_agents):
            action_one_hot[:, i * self.action_dim:(i + 1) * self.action_dim].scatter_(
                1, action_batch[:, i].unsqueeze(1), 1.0)

        # Critic loss
        with torch.no_grad():
            next_actions = [agent.actor_target(next_state_batch[:, i]).argmax(dim=-1)
                            for i, agent in enumerate(all_agents)]
            next_actions = torch.stack(next_actions, dim=1)
            next_action_one_hot = torch.zeros(batch_size, self.n_agents * self.action_dim)
            for i in range(self.n_agents):
                next_action_one_hot[:, i * self.action_dim:(i + 1) * self.action_dim].scatter_(
                    1, next_actions[:, i].unsqueeze(1), 1.0)
            target_q = self.critic_target(next_state_batch.view(batch_size, -1), next_action_one_hot)
            target_q = reward_batch[:, all_agents.index(self)] + self.gamma * (1 - done_batch[:, all_agents.index(self)]) * target_q

        current_q = self.critic(state_flat, action_one_hot)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_loss_history.append(critic_loss.item())

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actions_pred = [agent.actor(state_batch[:, i]) for i, agent in enumerate(all_agents)]
        actions_pred_one_hot = torch.zeros(batch_size, self.n_agents * self.action_dim)
        for i in range(self.n_agents):
            actions_pred_one_hot[:, i * self.action_dim:(i + 1) * self.action_dim] = actions_pred[i]
        actor_loss = -self.critic(state_flat, actions_pred_one_hot).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item()

    def get_stats(self):
        return {
            'avg_critic_loss': np.mean(self.critic_loss_history) if self.critic_loss_history else 0.0,
            'buffer_size': len(self.replay_buffer)
        }

class MADDPG:
    def __init__(self, state_dim, action_dim, n_agents, hyperparams):
        self.n_agents = n_agents
        self.agents = [
            MADDPGAgent(state_dim, action_dim, n_agents,
                        lr_actor=hyperparams.lr_actor,
                        lr_critic=hyperparams.lr_critic,
                        gamma=hyperparams.gamma,
                        tau=hyperparams.tau)
            for _ in range(n_agents)
        ]

    def act(self, states, training=True):
        return [agent.act(state, training) for agent, state in zip(self.agents, states)]

    def update(self, batch_size=64):
        critic_losses = []
        for agent in self.agents:
            loss = agent.update(batch_size, self.agents)
            if loss > 0:
                critic_losses.append(loss)
        return np.mean(critic_losses) if critic_losses else 0.0

    def add_experience(self, states, actions, rewards, next_states, dones):
        for i, agent in enumerate(self.agents):
            agent.replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def get_stats(self):
        return [agent.get_stats() for agent in self.agents]