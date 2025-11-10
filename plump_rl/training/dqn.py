"""Simple PyTorch DQN trainer for the Plump environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from loguru import logger

from .mlflow_utils import end_run_if_started, ensure_run, log_metric, log_params, set_tag

from ..encoding import encode_observation, observation_dim
from ..env import EnvConfig, PlumpEnv
from ..policies import BasePolicy

AgentFn = Callable[[dict, dict, PlumpEnv], int]


def _flatten_observation(obs: dict, config: EnvConfig) -> np.ndarray:
    return encode_observation(obs, config)


def _obs_size(config: EnvConfig) -> int:
    return observation_dim(config)


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        value = self.value_head(h)
        adv = self.adv_head(h)
        return value + adv - adv.mean(dim=1, keepdim=True)


@dataclass
class ReplayBuffer:
    capacity: int
    reward_clip: Optional[float] = 1.0

    def __post_init__(self):
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ):
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        experience = (state, action, reward, next_state, done, next_legal_mask.astype(np.uint8))
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            np.stack(next_masks),
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
        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
        self.target_update_interval = target_update_interval
        self.illegal_argmax = 0
        self.greedy_calls = 0
        self.q_abs_running = 0.0
        self.q_abs_updates = 0

    def _greedy_action(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        legal_indices = np.nonzero(legal_mask)[0]
        if len(legal_indices) == 0:
            raise RuntimeError("No legal actions available.")
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        masked_q = np.full_like(q_values, -1e9)
        masked_q[legal_indices] = q_values[legal_indices]
        greedy_action = int(np.argmax(q_values))
        if greedy_action not in legal_indices:
            self.illegal_argmax += 1
        self.greedy_calls += 1
        return int(np.argmax(masked_q))

    def current_epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.step_count / self.epsilon_decay
        )

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        epsilon = self.current_epsilon()
        self.step_count += 1
        legal_indices = np.nonzero(legal_mask)[0]
        if len(legal_indices) == 0:
            raise RuntimeError("No legal actions available.")

        if np.random.rand() < epsilon:
            return int(np.random.choice(legal_indices))
        return self._greedy_action(state, legal_mask)

    def act_greedy(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        return self._greedy_action(state, legal_mask)

    def update(self, buffer: ReplayBuffer, batch_size: int):
        if len(buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones, next_masks = buffer.sample(batch_size)
        states_tensor = torch.from_numpy(states).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(next_states).float().to(self.device)
        dones_tensor = torch.from_numpy(dones).float().to(self.device)
        next_masks_tensor = torch.from_numpy(next_masks).bool().to(self.device)

        q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_policy = self.policy_net(next_states_tensor)
            next_q_policy[~next_masks_tensor] = -1e9
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states_tensor).gather(1, next_actions).squeeze(1)
            targets = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q_target
        self.q_abs_running += q_values.detach().abs().mean().item()
        self.q_abs_updates += 1
        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def stats(self) -> dict:
        calls = max(1, self.greedy_calls)
        abs_updates = max(1, self.q_abs_updates)
        return {
            "illegal_argmax_rate": self.illegal_argmax / calls,
            "mean_abs_q": self.q_abs_running / abs_updates,
        }


@dataclass
class TrainingResult:
    episode_rewards: List[float]
    config: EnvConfig
    agent: DQNAgent


def train_dqn(
    num_episodes: int = 1000,
    *,
    config: Optional[EnvConfig] = None,
    opponents: Optional[Sequence[Optional[BasePolicy]]] = None,
    seed: Optional[int] = None,
    replay_capacity: int = 50_000,
    batch_size: int = 64,
    warmup_steps: int = 500,
    show_progress: bool = False,
) -> TrainingResult:
    """Train a DQN agent directly on PlumpEnv using PyTorch."""
    run_started = ensure_run("train_dqn")
    try:
        env_config = config or EnvConfig()
        env = PlumpEnv(env_config, opponents=opponents, seed=seed)
        state_dim = _obs_size(env_config)
        action_dim = env.action_space.n
        agent = DQNAgent(state_dim, action_dim)
        buffer = ReplayBuffer(replay_capacity)
        episode_rewards: List[float] = []
        update_counter = 0

        set_tag("algorithm", "dqn")
        log_params(
            {
                "algo": "dqn",
                "dqn.num_episodes": num_episodes,
                "dqn.replay_capacity": replay_capacity,
                "dqn.batch_size": batch_size,
                "dqn.warmup_steps": warmup_steps,
                "env.num_players": env_config.num_players,
                "env.hand_size": env_config.hand_size,
                "env.agent_id": env_config.agent_id,
                "training.seed": seed if seed is not None else -1,
            }
        )
        log_params(
            {
                "dqn.gamma": agent.gamma,
                "dqn.lr": agent.lr,
                "dqn.epsilon_start": agent.epsilon_start,
                "dqn.epsilon_end": agent.epsilon_end,
                "dqn.epsilon_decay": agent.epsilon_decay,
                "dqn.target_update_interval": agent.target_update_interval,
            }
        )

        iterator: Iterable[int]
        if show_progress:
            iterator = tqdm(range(num_episodes), desc="Training", unit="episode")
        else:
            iterator = range(num_episodes)

        for episode in iterator:
            obs, info = env.reset()
            state = _flatten_observation(obs, env_config)
            done = False
            total_reward = 0.0
            episode_losses: List[float] = []
            episode_steps = 0

            while not done:
                action = agent.select_action(state, info["legal_actions"])
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = _flatten_observation(next_obs, env_config)
                buffer.push(state, action, reward, next_state, done, info["legal_actions"])
                state = next_state
                total_reward += reward
                episode_steps += 1

                if agent.step_count > warmup_steps:
                    loss_value = agent.update(buffer, batch_size)
                    if loss_value is not None:
                        update_counter += 1
                        episode_losses.append(loss_value)
                        log_metric("dqn/batch_loss", loss_value, step=update_counter)

            step_idx = episode + 1
            episode_rewards.append(total_reward)
            log_metric("dqn/episode_reward", total_reward, step=step_idx)
            log_metric("dqn/episode_length", episode_steps, step=step_idx)
            log_metric("dqn/epsilon", agent.current_epsilon(), step=step_idx)
            if episode_losses:
                log_metric("dqn/episode_loss", float(np.mean(episode_losses)), step=step_idx)

            if not show_progress and step_idx % 50 == 0:
                avg = np.mean(episode_rewards[-50:])
                logger.info("[Episode {}] avg_reward (last 50): {:.2f}", step_idx, avg)
                log_metric("dqn/avg_reward_last_50", float(avg), step=step_idx)

        if not show_progress:
            stats = agent.stats()
            logger.info(
                "Illegal argmax rate: {:.3f} | mean |Q|: {:.3f}",
                stats["illegal_argmax_rate"],
                stats["mean_abs_q"],
            )
            log_metric("dqn/illegal_argmax_rate", stats["illegal_argmax_rate"], step=num_episodes)
            log_metric("dqn/mean_abs_q", stats["mean_abs_q"], step=num_episodes)

        return TrainingResult(episode_rewards=episode_rewards, config=env_config, agent=agent)
    finally:
        end_run_if_started(run_started)


def make_dqn_agent_policy(agent: DQNAgent, config: EnvConfig) -> AgentFn:
    """Wrap a trained agent so it can be passed into tournament helpers."""

    def _policy(obs: dict, info: dict, env: PlumpEnv) -> int:
        del env
        state = _flatten_observation(obs, config)
        return agent.act_greedy(state, info["legal_actions"])

    return _policy


def make_dqn_agent_policy_from_state(state_dict: dict, config: EnvConfig) -> AgentFn:
    """Return a greedy policy callable from a serialized network state."""

    state_dim = observation_dim(config)
    action_dim = config.hand_size + 1 + 52
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()

    def _policy(obs: dict, info: dict, env: PlumpEnv) -> int:
        del env
        state = encode_observation(obs, config)
        return agent.act_greedy(state, info["legal_actions"])

    return _policy
