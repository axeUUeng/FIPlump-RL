"""Simple PPO trainer for Plump."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tqdm import tqdm

from ..encoding import encode_observation, observation_dim
from ..env import EnvConfig, PlumpEnv
from ..policies import BasePolicy


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)


@dataclass
class PPORollout:
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    masks: List[np.ndarray]
    values: List[float]
    logprobs: List[float]


def compute_returns_advantages(rewards, dones, values, gamma, lam):
    advantages = []
    gae = 0.0
    next_value = 0.0
    for r, done, v in zip(reversed(rewards), reversed(dones), reversed(values)):
        delta = r + gamma * next_value * (1 - done) - v
        gae = delta + gamma * lam * (1 - done) * gae
        advantages.insert(0, gae)
        next_value = v
    returns = [a + v for a, v in zip(advantages, values)]
    advantages = np.array(advantages, dtype=np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return np.array(returns, dtype=np.float32), advantages


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor):
    masked = logits.clone()
    masked[mask == 0] = -1e9
    dist = torch.distributions.Categorical(logits=masked)
    return dist


@dataclass
class PPOResult:
    episode_rewards: List[float]
    config: EnvConfig
    model_state: dict


def train_ppo(
    num_episodes: int = 1000,
    *,
    config: Optional[EnvConfig] = None,
    opponents: Optional[Sequence[Optional[BasePolicy]]] = None,
    seed: Optional[int] = None,
    rollout_size: int = 10,
    batch_epochs: int = 4,
    clip_ratio: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    show_progress: bool = False,
) -> PPOResult:
    """Train a PPO agent directly on the Plump environment."""
    env_config = config or EnvConfig()
    env = PlumpEnv(env_config, opponents=opponents, seed=seed)
    obs_dim = len(encode_observation(env.reset()[0], env_config))
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(obs_dim, action_dim).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr)

    episode_rewards: List[float] = []

    iterator: Iterable[int]
    if show_progress:
        iterator = tqdm(range(num_episodes), desc="PPO Training", unit="episode")
    else:
        iterator = range(num_episodes)

    obs, info = env.reset()
    state = encode_observation(obs, env_config)
    total_reward = 0.0

    for episode in iterator:
        rollout = PPORollout([], [], [], [], [], [], [])
        steps = 0
        done = False
        while not done and steps < rollout_size:
            mask = info["legal_actions"]
            rollout.states.append(state)
            rollout.masks.append(mask.copy())
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            logits, value = net(state_tensor)
            mask_tensor = torch.from_numpy(mask).to(device)
            dist = masked_categorical(logits.squeeze(0), mask_tensor)
            action = int(dist.sample().item())
            rollout.actions.append(action)
            rollout.values.append(value.item())
            rollout.logprobs.append(dist.log_prob(torch.tensor(action, device=device)).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rollout.rewards.append(reward)
            rollout.dones.append(done)
            state = encode_observation(next_obs, env_config)
            total_reward += reward
            steps += 1

        returns, advantages = compute_returns_advantages(
            rollout.rewards, rollout.dones, rollout.values, gamma, lam
        )
        states_tensor = torch.from_numpy(np.stack(rollout.states)).float().to(device)
        actions_tensor = torch.from_numpy(np.array(rollout.actions)).long().to(device)
        returns_tensor = torch.from_numpy(returns).float().to(device)
        adv_tensor = torch.from_numpy(advantages).float().to(device)
        old_logprobs_tensor = torch.from_numpy(np.array(rollout.logprobs)).float().to(device)
        masks_tensor = torch.from_numpy(np.stack(rollout.masks)).to(device)

        for _ in range(batch_epochs):
            logits, values = net(states_tensor)
            dist = masked_categorical(logits, masks_tensor)
            log_probs = dist.log_prob(actions_tensor)
            ratios = torch.exp(log_probs - old_logprobs_tensor)
            surr1 = ratios * adv_tensor
            surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * adv_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, returns_tensor)
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

        if done:
            episode_rewards.append(total_reward)
            total_reward = 0.0
            obs, info = env.reset()
            state = encode_observation(obs, env_config)

        if not show_progress and (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            logger.info("[PPO Episode {}] avg_reward (last 50): {:.2f}", episode + 1, avg)

    return PPOResult(
        episode_rewards=episode_rewards,
        config=env_config,
        model_state=net.state_dict(),
    )


def make_ppo_agent_policy(model_state: dict, config: EnvConfig) -> AgentFn:
    """Return a stochastic policy callable from a PPO model state."""

    obs_dim = observation_dim(config)
    action_dim = config.hand_size + 1 + 52
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(obs_dim, action_dim)
    net.load_state_dict(model_state)
    net.to(device)
    net.eval()

    def _policy(obs: dict, info: dict, env: PlumpEnv) -> int:
        del env
        state = encode_observation(obs, config)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = net(state_tensor)
        mask = torch.from_numpy(info["legal_actions"]).to(device)
        dist = masked_categorical(logits.squeeze(0), mask)
        return int(dist.sample().item())

    return _policy
AgentFn = Callable[[dict, dict, PlumpEnv], int]
