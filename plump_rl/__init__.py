"""Utilities for training reinforcement-learning agents on Plump."""

from .env import EnvConfig, PlumpEnv
from .policies import (
    DressedCardPolicy,
    MiddleManager,
    RandomLegalPolicy,
    RuleBasedPolicy,
    ShortSuitAggressor,
    ZeroBidDodger,
)
from .tournament import (
    PolicyAggregate,
    RoundResult,
    TournamentBatchResult,
    default_schedule,
    format_round_history,
    run_schedule,
    round_results_to_dict,
    simulate_random_tournaments,
)
from .training.dqn import (
    TrainingResult,
    make_dqn_agent_policy,
    make_dqn_agent_policy_from_state,
    train_dqn,
)
from .training.ppo import PPOResult, make_ppo_agent_policy, train_ppo

__all__ = [
    "PlumpEnv",
    "EnvConfig",
    "DressedCardPolicy",
    "RuleBasedPolicy",
    "ZeroBidDodger",
    "ShortSuitAggressor",
    "MiddleManager",
    "RandomLegalPolicy",
    "PolicyAggregate",
    "RoundResult",
    "TournamentBatchResult",
    "default_schedule",
    "format_round_history",
    "run_schedule",
    "round_results_to_dict",
    "simulate_random_tournaments",
    "train_dqn",
    "TrainingResult",
    "make_dqn_agent_policy",
    "make_dqn_agent_policy_from_state",
    "train_ppo",
    "PPOResult",
    "make_ppo_agent_policy",
]
