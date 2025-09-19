import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):  # MODIFIED: Use hidden_dim
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class MixerNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=64):  # MODIFIED: Use hidden_dim (default 64 now)
        super(MixerNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Hypernetworks that generate weights and biases
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim * 1)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, q_values, states):
        # q_values: [batch_size, n_agents]
        # states: [batch_size, state_dim]
        batch_size = q_values.shape[0]

        # Generate mixing network weights and biases
        w1 = torch.abs(self.hyper_w1(states))  # [batch_size, n_agents * hidden_dim]
        w2 = torch.abs(self.hyper_w2(states))  # [batch_size, hidden_dim * 1]
        b1 = self.hyper_b1(states)  # [batch_size, hidden_dim]
        b2 = self.hyper_b2(states)  # [batch_size, 1]

        # Reshape weights
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)  # [batch_size, n_agents, hidden_dim]
        w2 = w2.view(batch_size, self.hidden_dim, 1)  # [batch_size, hidden_dim, 1]

        # First layer: q_values -> hidden
        q_values = q_values.view(batch_size, self.n_agents, 1)  # [batch_size, n_agents, 1]
        hidden = torch.bmm(w1.transpose(1, 2), q_values).squeeze(-1)  # [batch_size, hidden_dim]
        hidden = hidden + b1
        hidden = torch.relu(hidden)

        # Second layer: hidden -> output
        hidden = hidden.unsqueeze(-1)  # [batch_size, hidden_dim, 1]
        output = torch.bmm(w2.transpose(1, 2), hidden).squeeze(-1)  # [batch_size, 1]
        output = output + b2

        return output


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, states, actions, rewards, next_states, dones):
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


class QMIX:
    def __init__(self, state_dim, action_dim, n_agents, hyperparams):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparams = hyperparams
        self.batch_size = hyperparams.batch_size  # NEW: Configurable batch size (default 64 for stability)

        # Individual Q-networks for each agent
        self.q_networks = [QNetwork(state_dim, action_dim, hyperparams.hidden_dim) for _ in range(n_agents)]
        self.target_q_networks = [QNetwork(state_dim, action_dim, hyperparams.hidden_dim) for _ in range(n_agents)]
        for target, source in zip(self.target_q_networks, self.q_networks):
            target.load_state_dict(source.state_dict())

        # Mixer network
        self.mixer = MixerNetwork(n_agents, state_dim, hyperparams.hidden_dim)
        self.target_mixer = MixerNetwork(n_agents, state_dim, hyperparams.hidden_dim)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Optimizers
        self.q_optimizers = [optim.Adam(q.parameters(), lr=hyperparams.lr) for q in self.q_networks]
        self.mixer_optimizer = optim.Adam(self.mixer.parameters(), lr=hyperparams.lr)

        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.epsilon = hyperparams.epsilon
        self.epsilon_min = hyperparams.epsilon_min
        self.epsilon_decay = hyperparams.epsilon_decay
        self.loss_history = []

    def act(self, states, training=True):
        if training and random.random() < self.epsilon:
            return [np.random.randint(0, self.action_dim) for _ in range(self.n_agents)]
        else:
            with torch.no_grad():
                states_tensor = torch.FloatTensor(np.array(states))
                q_values = [q(states_tensor[i].unsqueeze(0)).squeeze(0) for i, q in enumerate(self.q_networks)]
                return [np.argmax(q_values[i].numpy()) for i in range(self.n_agents)]

    def add_experience(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.push(states, actions, rewards, next_states, dones)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        global_state = states.mean(dim=1)
        next_global_state = next_states.mean(dim=1)

        q_values = []
        for i in range(self.n_agents):
            q_values_i = self.q_networks[i](states[:, i])
            q_values.append(q_values_i.gather(1, actions[:, i].unsqueeze(1)).squeeze(1))
        q_values = torch.stack(q_values, dim=1)
        q_total = self.mixer(q_values, global_state)

        with torch.no_grad():
            next_q_values = []
            for i in range(self.n_agents):
                next_q_values_i = self.target_q_networks[i](next_states[:, i])
                next_q_values.append(next_q_values_i.max(dim=1)[0])
            next_q_values = torch.stack(next_q_values, dim=1)
            next_q_total = self.target_mixer(next_q_values, next_global_state)

            team_reward = rewards.sum(dim=1)
            team_done = dones.max(dim=1)[0]
            target_q = team_reward + self.hyperparams.gamma * (1 - team_done) * next_q_total.squeeze()

        loss = nn.MSELoss()(q_total.squeeze(), target_q)
        self.loss_history.append(loss.item())
        print(f"QMIX Loss: {loss.item():.4f}")  # NEW: Log loss for debugging

        for opt in self.q_optimizers:
            opt.zero_grad()
        self.mixer_optimizer.zero_grad()
        loss.backward()

        for q in self.q_networks:
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 1.0)

        for opt in self.q_optimizers:
            opt.step()
        self.mixer_optimizer.step()

        tau = getattr(self.hyperparams, 'tau', 0.005)
        for target, source in zip(self.target_q_networks, self.q_networks):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return np.mean(self.loss_history[-10:]) if self.loss_history else loss.item()