"""Proximal Policy Optimization (PPO) trainer for the Plump environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tqdm import tqdm

from .mlflow_utils import end_run_if_started, ensure_run, log_metric, log_params, set_tag
from ..encoding import encode_observation, observation_dim
from ..env import EnvConfig, PlumpEnv
from ..policies import BasePolicy

AgentFn = Callable[[dict, dict, PlumpEnv], int]
RNG = np.random.Generator


class PolicyValueNet(nn.Module):
    """Shared-backbone actor-critic network producing logits and value estimates."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    """Return a categorical distribution with illegal actions masked out."""

    if mask.dtype != torch.bool:
        mask = mask != 0
    if logits.dim() == 2:
        if (~mask).all(dim=1).any():
            raise RuntimeError("MaskedCategorical: encountered batch row with no legal actions.")
    else:
        if (~mask).all():
            raise RuntimeError("MaskedCategorical: no legal actions available.")
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
    return torch.distributions.Categorical(logits=masked_logits)


def compute_returns_advantages(
    rewards: List[float],
    dones: List[bool],
    values: List[float],
    gamma: float,
    lam: float,
    last_value: float,
    last_done: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation with proper bootstrapping."""

    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    next_advantage = 0.0
    next_value = last_value
    next_nonterminal = 1.0 - float(last_done)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        next_advantage = delta + gamma * lam * next_nonterminal * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
        next_nonterminal = 1.0 - float(dones[t])
    returns = advantages + np.array(values, dtype=np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


@dataclass
class PPOResult:
    episode_rewards: List[float]
    config: EnvConfig
    model_bundle: dict


def train_ppo(
    num_updates: int = 1000,
    *,
    config: Optional[EnvConfig] = None,
    opponents: Optional[Sequence[Optional[BasePolicy]]] = None,
    seed: Optional[int] = None,
    rollout_steps: int = 512,
    batch_epochs: int = 4,
    minibatch_size: int = 64,
    clip_ratio: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    vf_clip_eps: float = 0.2,
    target_kl: float = 0.03,
    max_grad_norm: float = 1.0,
    show_progress: bool = False,
) -> PPOResult:
    """Train a PPO agent on Plump."""
    run_started = ensure_run("train_ppo")
    try:
        env_config = config or EnvConfig()
        env = PlumpEnv(env_config, opponents=opponents, seed=seed)
        obs_dim = observation_dim(env_config)
        action_dim = env.action_space.n
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = PolicyValueNet(obs_dim, action_dim).to(device)
        optimizer = optim.AdamW(net.parameters(), lr=lr)

        episode_rewards: List[float] = []
        completed_episodes = 0
        iterator = tqdm(range(num_updates), desc="PPO Training", unit="update") if show_progress else range(num_updates)

        set_tag("algorithm", "ppo")
        log_params(
            {
                "algo": "ppo",
                "ppo.num_updates": num_updates,
                "ppo.rollout_steps": rollout_steps,
                "ppo.batch_epochs": batch_epochs,
                "ppo.minibatch_size": minibatch_size,
                "ppo.clip_ratio": clip_ratio,
                "ppo.gamma": gamma,
                "ppo.lam": lam,
                "ppo.lr": lr,
                "ppo.ent_coef": ent_coef,
                "ppo.vf_coef": vf_coef,
                "ppo.vf_clip_eps": vf_clip_eps,
                "ppo.target_kl": target_kl,
                "ppo.max_grad_norm": max_grad_norm,
                "env.num_players": env_config.num_players,
                "env.hand_size": env_config.hand_size,
                "env.agent_id": env_config.agent_id,
                "training.seed": seed if seed is not None else -1,
            }
        )

        obs, info = env.reset()
        state = encode_observation(obs, env_config)
        episode_return = 0.0

        for update in iterator:
            rollout_states: List[np.ndarray] = []
            rollout_actions: List[int] = []
            rollout_masks: List[np.ndarray] = []
            rollout_rewards: List[float] = []
            rollout_dones: List[bool] = []
            rollout_values: List[float] = []
            rollout_logprobs: List[float] = []

            steps_collected = 0
            while steps_collected < rollout_steps:
                mask_np = info["legal_actions"]
                rollout_states.append(state)
                rollout_masks.append(mask_np.copy())

                with torch.inference_mode():
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    logits, value = net(state_tensor)
                    mask_tensor = torch.from_numpy(mask_np).to(device)
                    dist = masked_categorical(logits.squeeze(0), mask_tensor)
                    action = int(dist.sample().item())
                    logprob = dist.log_prob(torch.tensor(action, device=device)).item()
                    value_scalar = float(value.item())

                rollout_actions.append(action)
                rollout_values.append(value_scalar)
                rollout_logprobs.append(logprob)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rollout_rewards.append(float(reward))
                rollout_dones.append(done)
                episode_return += reward
                steps_collected += 1

                if done:
                    episode_rewards.append(episode_return)
                    completed_episodes += 1
                    log_metric("ppo/episode_reward", episode_return, step=completed_episodes)
                    episode_return = 0.0
                    obs, info = env.reset()
                    state = encode_observation(obs, env_config)
                else:
                    state = encode_observation(next_obs, env_config)

            with torch.inference_mode():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                _, last_value_tensor = net(state_tensor)
            last_value = float(last_value_tensor.item())
            last_done = rollout_dones[-1] if rollout_dones else False

            returns, advantages = compute_returns_advantages(
                rollout_rewards,
                rollout_dones,
                rollout_values,
                gamma,
                lam,
                last_value,
                last_done,
            )

            states_tensor = torch.from_numpy(np.stack(rollout_states)).float().to(device)
            actions_tensor = torch.tensor(rollout_actions, dtype=torch.long, device=device)
            returns_tensor = torch.from_numpy(returns).float().to(device)
            advantages_tensor = torch.from_numpy(advantages).float().to(device)
            old_logprobs_tensor = torch.tensor(rollout_logprobs, dtype=torch.float32, device=device)
            old_values_tensor = torch.tensor(rollout_values, dtype=torch.float32, device=device)
            masks_tensor = torch.from_numpy(np.stack(rollout_masks)).to(device)

            dataset_size = actions_tensor.shape[0]
            minibatch = min(minibatch_size, dataset_size)
            indices = np.arange(dataset_size)

            batch_policy_losses: List[float] = []
            batch_value_losses: List[float] = []
            batch_entropies: List[float] = []
            batch_losses: List[float] = []
            batch_clip_fractions: List[float] = []

        for epoch in range(batch_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, minibatch):
                mb_idx = indices[start : start + minibatch]

                logits, values = net(states_tensor[mb_idx])
                dist = masked_categorical(logits, masks_tensor[mb_idx])
                logp = dist.log_prob(actions_tensor[mb_idx])
                entropy = dist.entropy().mean()

                ratios = torch.exp(logp - old_logprobs_tensor[mb_idx])
                surr1 = ratios * advantages_tensor[mb_idx]
                surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages_tensor[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                clip_fraction = (
                    torch.gt(ratios, 1 + clip_ratio) | torch.lt(ratios, 1 - clip_ratio)
                ).float().mean().item()

                if vf_clip_eps > 0:
                    value_pred = values.squeeze(-1)
                    v_clipped = old_values_tensor[mb_idx] + torch.clamp(
                        value_pred - old_values_tensor[mb_idx], -vf_clip_eps, vf_clip_eps
                    )
                    unclipped = (value_pred - returns_tensor[mb_idx]) ** 2
                    clipped = (v_clipped - returns_tensor[mb_idx]) ** 2
                    value_loss = 0.5 * torch.max(unclipped, clipped).mean()
                else:
                    value_loss = 0.5 * nn.functional.mse_loss(values.squeeze(-1), returns_tensor[mb_idx])

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

                batch_policy_losses.append(policy_loss.item())
                batch_value_losses.append(value_loss.item())
                batch_entropies.append(entropy.item())
                batch_losses.append(loss.item())
                batch_clip_fractions.append(clip_fraction)

            with torch.inference_mode():
                logits_new, _ = net(states_tensor)
                dist_new = masked_categorical(logits_new, masks_tensor)
                approx_kl = (old_logprobs_tensor - dist_new.log_prob(actions_tensor)).mean().clamp_min(0).item()
            if approx_kl > target_kl:
                if not show_progress:
                    logger.debug("Early stopping epoch due to KL {:.4f} > {:.4f}", approx_kl, target_kl)
                break

        step_idx = update + 1
        if batch_losses:
            log_metric("ppo/loss", float(np.mean(batch_losses)), step=step_idx)
        if batch_policy_losses:
            log_metric("ppo/policy_loss", float(np.mean(batch_policy_losses)), step=step_idx)
        if batch_value_losses:
            log_metric("ppo/value_loss", float(np.mean(batch_value_losses)), step=step_idx)
        if batch_entropies:
            log_metric("ppo/entropy", float(np.mean(batch_entropies)), step=step_idx)
        if batch_clip_fractions:
            log_metric("ppo/clip_fraction", float(np.mean(batch_clip_fractions)), step=step_idx)
        log_metric("ppo/approx_kl", approx_kl, step=step_idx)
        log_metric("ppo/rollout_steps", steps_collected, step=step_idx)
        if rollout_rewards:
            log_metric("ppo/rollout_reward_mean", float(np.mean(rollout_rewards)), step=step_idx)

        if not show_progress and step_idx % 50 == 0:
            recent = episode_rewards[-50:] or episode_rewards
            avg_recent = float(np.mean(recent)) if recent else 0.0
            logger.info("[PPO Update {}] avg_reward (last 50 episodes): {:.2f}", step_idx, avg_recent)
            log_metric("ppo/avg_reward_last_50", avg_recent, step=step_idx)

        log_metric("ppo/completed_episodes", completed_episodes, step=step_idx)

    model_bundle = {"state_dict": net.state_dict(), "action_dim": action_dim}
    return PPOResult(episode_rewards=episode_rewards, config=env_config, model_bundle=model_bundle)
    finally:
        end_run_if_started(run_started)


def make_ppo_agent_policy(model_bundle: dict, config: EnvConfig) -> AgentFn:
    """Return a callable policy for inference from a saved PPO checkpoint."""

    state_dict = model_bundle.get("state_dict", model_bundle)
    action_dim = model_bundle.get("action_dim")
    if action_dim is None:
        action_dim = config.hand_size + 52 + 1
    obs_dim = observation_dim(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(obs_dim, action_dim)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    def _policy(obs: dict, info: dict, env: PlumpEnv) -> int:
        del env
        state = encode_observation(obs, config)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            logits, _ = net(state_tensor)
        mask_np = info["legal_actions"]
        if mask_np.shape[0] != action_dim:
            padded = np.zeros(action_dim, dtype=mask_np.dtype)
            length = min(action_dim, mask_np.shape[0])
            padded[:length] = mask_np[:length]
            mask_np = padded
        mask = torch.from_numpy(mask_np).to(device)
        dist = masked_categorical(logits.squeeze(0), mask)
        return int(dist.sample().item())

    return _policy


def load_agent_policy_from_state(model_bundle: dict, config: EnvConfig) -> AgentFn:
    """Compatibility helper matching the DQN interface."""

    return make_ppo_agent_policy(model_bundle, config)
