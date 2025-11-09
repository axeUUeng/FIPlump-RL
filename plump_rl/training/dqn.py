"""Simple PyTorch DQN trainer for the Plump environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..env import EnvConfig, PlumpEnv
from ..policies import BasePolicy


def _flatten_observation(obs: dict, config: EnvConfig) -> np.ndarray:
    phase = np.array([obs["phase"]], dtype=np.float32)
    hand = obs["hand"].astype(np.float32)
    trick = obs["current_trick"].astype(np.float32) / 52.0
    lead = np.array([obs["lead_suit"] / 4.0], dtype=np.float32)
    estimations = obs["estimations"].astype(np.float32) / max(1, config.hand_size)
    tricks_won = obs["tricks_won"].astype(np.float32) / max(1, config.hand_size)
    cards_remaining = obs["cards_remaining"].astype(np.float32) / max(1, config.hand_size)
    tricks_played = np.array([obs["tricks_played"] / max(1, config.hand_size)], dtype=np.float32)
    return np.concatenate(
        [phase, hand, trick, lead, estimations, tricks_won, cards_remaining, tricks_played]
    )


def _obs_size(config: EnvConfig) -> int:
    dummy_obs = {
        "phase": 0,
        "hand": np.zeros(52, dtype=np.int8),
        "current_trick": np.zeros(config.num_players, dtype=np.int16),
        "lead_suit": 4,
        "estimations": np.zeros(config.num_players, dtype=np.int8),
        "tricks_won": np.zeros(config.num_players, dtype=np.int8),
        "cards_remaining": np.zeros(config.num_players, dtype=np.int8),
        "tricks_played": np.int8(0),
    }
    return len(_flatten_observation(dummy_obs, config))


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ReplayBuffer:
    capacity: int

    def __post_init__(self):
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 5000,
        target_update_interval: int = 500,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
        self.target_update_interval = target_update_interval

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.step_count / self.epsilon_decay
        )
        self.step_count += 1
        legal_indices = np.nonzero(legal_mask)[0]
        if len(legal_indices) == 0:
            raise RuntimeError("No legal actions available.")

        if np.random.rand() < epsilon:
            return int(np.random.choice(legal_indices))

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        masked_q = np.full_like(q_values, -1e9)
        masked_q[legal_indices] = q_values[legal_indices]
        return int(np.argmax(masked_q))

    def update(self, buffer: ReplayBuffer, batch_size: int):
        if len(buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        states_tensor = torch.from_numpy(states).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(next_states).float().to(self.device)
        dones_tensor = torch.from_numpy(dones).float().to(self.device)

        q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_tensor).max(1)[0]
            targets = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


@dataclass
class TrainingResult:
    episode_rewards: List[float]
    config: EnvConfig


def train_dqn(
    num_episodes: int = 1000,
    *,
    config: Optional[EnvConfig] = None,
    opponents: Optional[Sequence[Optional[BasePolicy]]] = None,
    seed: Optional[int] = None,
    replay_capacity: int = 50_000,
    batch_size: int = 64,
    warmup_steps: int = 500,
) -> TrainingResult:
    """Train a DQN agent directly on PlumpEnv using PyTorch."""

    env_config = config or EnvConfig()
    env = PlumpEnv(env_config, opponents=opponents, seed=seed)
    state_dim = _obs_size(env_config)
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    buffer = ReplayBuffer(replay_capacity)
    episode_rewards: List[float] = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        state = _flatten_observation(obs, env_config)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state, info["legal_actions"])
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = _flatten_observation(next_obs, env_config)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if agent.step_count > warmup_steps:
                agent.update(buffer, batch_size)

        episode_rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"[Episode {episode + 1}] avg_reward (last 50): {avg:.2f}")

    return TrainingResult(episode_rewards=episode_rewards, config=env_config)
