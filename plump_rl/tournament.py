"""Helpers for running multi-hand Plump tournaments (10→1→10 style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cards import card_to_str
from .env import EnvConfig, PlumpEnv
from .policies import BasePolicy, TrickContext

AgentPolicy = Callable[[dict, dict, PlumpEnv], int]


@dataclass
class RoundResult:
    hand_size: int
    reward: float
    round_points: Optional[List[int]]
    info: dict
    history: Optional[dict] = None


def format_round_history(history: Optional[dict], player_labels: Optional[Sequence[str]] = None) -> str:
    if history is None:
        return "No history recorded."
    num_players = len(history.get("players", []))
    if player_labels is None:
        player_labels = [f"P{i}" for i in range(num_players)]
    lines = [
        f"Round {history.get('round_index', '?')} | hand_size={history.get('hand_size')}",
        f"Dealer: {player_labels[history.get('dealer', 0)] if history.get('dealer') is not None else '?'}",
    ]
    lines.append("Estimations:")
    for entry in history.get("estimations", []):
        pid = entry["player"]
        lines.append(f"  {player_labels[pid]} -> {entry['estimate']}")

    lines.append("Tricks:")
    for trick in history.get("tricks", []):
        winner = trick.get("winner")
        lines.append(f"  Trick {trick.get('trick_index')} winner={player_labels[winner] if winner is not None else '?'}")
        for card in trick.get("cards", []):
            card_id = card.get("card")
            if card_id is None or card_id == -1:
                continue
            lines.append(f"    {player_labels[card['player']]} played {card_to_str(card_id)}")

    points = history.get("round_points")
    if points is not None:
        lines.append("Round points:")
        for idx, val in enumerate(points):
            lines.append(f"  {player_labels[idx]}: {val}")
    return "\n".join(lines)


def round_results_to_dict(round_results: Sequence[RoundResult]) -> List[dict]:
    payload: List[dict] = []
    for result in round_results:
        payload.append(
            {
                "hand_size": result.hand_size,
                "reward": result.reward,
                "round_points": result.round_points,
                "history": result.history,
            }
        )
    return payload


def default_schedule(max_hand_size: int = 10, min_hand_size: int = 1) -> List[int]:
    """Return the canonical schedule: max→min then back up (without repeating endpoints)."""
    down = list(range(max_hand_size, min_hand_size - 1, -1))
    up = list(range(min_hand_size + 1, max_hand_size + 1))
    return down + up


def random_policy_factory(seed: Optional[int] = None) -> AgentPolicy:
    rng = np.random.default_rng(seed)

    def _policy(obs: dict, info: dict, env: PlumpEnv) -> int:
        del obs, env
        legal_idx = np.nonzero(info["legal_actions"])[0]
        if len(legal_idx) == 0:
            raise RuntimeError("No legal actions available.")
        return int(rng.choice(legal_idx))

    return _policy


def agent_from_policy(policy: BasePolicy) -> AgentPolicy:
    """Wrap a BasePolicy so it can control the learning agent slot."""

    def _agent(obs: dict, info: dict, env: PlumpEnv) -> int:
        legal = np.nonzero(info["legal_actions"])[0]
        if len(legal) == 0:
            raise RuntimeError("No legal actions available for heuristic agent.")
        hand_cards = [idx for idx, flag in enumerate(obs["hand"]) if flag]

        if obs["phase"] == 0:
            cant_say = env._cant_say_value(env.config.agent_id)  # pylint: disable=protected-access
            estimation = policy.estimate(hand_cards, len(hand_cards), cant_say, env.np_random)
            estimation = max(0, min(env.config.hand_size, estimation))
            if estimation in legal:
                return estimation
            # fall back to nearest legal estimation
            estimations = [idx for idx in legal if idx <= env.config.hand_size]
            return int(estimations[0]) if estimations else int(legal[0])

        # Playing phase
        playable_cards = [
            action - env.estimation_action_count
            for action in legal
            if action >= env.estimation_action_count
        ]
        if not playable_cards:
            return int(legal[0])

        lead_suit = obs["lead_suit"]
        lead_suit = -1 if lead_suit == 4 else lead_suit
        ctx = TrickContext(lead_suit=lead_suit, trick_cards=tuple(obs["current_trick"]))
        choice = policy.play(hand_cards, playable_cards, ctx, env.np_random)
        if choice not in playable_cards:
            choice = playable_cards[0]
        return env.estimation_action_count + choice

    return _agent


def run_schedule(
    agent: Optional[AgentPolicy] = None,
    *,
    schedule: Optional[Iterable[int]] = None,
    base_config: Optional[EnvConfig] = None,
    opponents: Optional[Sequence[Optional[BasePolicy]]] = None,
    seed: Optional[int] = None,
    record_games: bool = False,
) -> List[RoundResult]:
    """Play a series of Plump rounds with varying hand sizes."""
    base_config = base_config or EnvConfig()
    hand_sizes = list(schedule or default_schedule(base_config.hand_size, 1))
    results: List[RoundResult] = []
    rng = np.random.default_rng(seed)
    policy = agent or random_policy_factory(seed)

    for idx, hand_size in enumerate(hand_sizes):
        cfg = EnvConfig(
            num_players=base_config.num_players,
            hand_size=hand_size,
            agent_id=base_config.agent_id,
            match_bonus=base_config.match_bonus,
            invalid_action_penalty=base_config.invalid_action_penalty,
        )
        env_seed = None if seed is None else int(seed + idx)
        env = PlumpEnv(cfg, opponents=opponents, seed=env_seed, record_history=record_games)
        obs, info = env.reset()
        done = False
        reward_acc = 0.0
        while not done:
            legal_mask = info["legal_actions"]
            if not legal_mask.any():
                raise RuntimeError("Agent policy produced an invalid state with no legal moves.")
            action = policy(obs, info, env)
            if legal_mask[action] != 1:
                # fall back to random legal move to keep sim going
                legal_idx = np.nonzero(legal_mask)[0]
                action = int(rng.choice(legal_idx))
            obs, reward, terminated, truncated, info = env.step(action)
            reward_acc += reward
            done = terminated or truncated
        history = None
        if record_games and getattr(env, "completed_round_logs", None):
            history = env.completed_round_logs[-1]
        results.append(RoundResult(hand_size, reward_acc, info.get("round_points"), info, history))
    return results


@dataclass
class PolicyAggregate:
    name: str
    seats_played: int = 0
    total_points: float = 0.0
    tournament_wins: int = 0

    @property
    def average_points(self) -> float:
        return self.total_points / self.seats_played if self.seats_played else 0.0


@dataclass
class TournamentBatchResult:
    policy_stats: Dict[str, PolicyAggregate]
    seat_history: List[List[str]]
    rounds: Optional[List[List[RoundResult]]] = None


def simulate_random_tournaments(
    num_tournaments: int,
    *,
    policy_factories: Sequence[Callable[[], BasePolicy]],
    base_config: Optional[EnvConfig] = None,
    schedule: Optional[Iterable[int]] = None,
    seed: Optional[int] = None,
    record_games: bool = False,
) -> TournamentBatchResult:
    """Run multiple tournaments with random heuristic assignments per seat."""

    if num_tournaments <= 0:
        raise ValueError("num_tournaments must be positive")
    if not policy_factories:
        raise ValueError("policy_factories cannot be empty")

    base_config = base_config or EnvConfig(num_players=5, hand_size=10)
    rng = np.random.default_rng(seed)
    stats: Dict[str, PolicyAggregate] = {}
    seat_history: List[List[str]] = []
    tournament_rounds: List[List[RoundResult]] = []

    for tournament_idx in range(num_tournaments):
        seat_assignments: List[Tuple[BasePolicy, str]] = []
        for _ in range(base_config.num_players):
            factory = rng.choice(policy_factories)
            policy = factory()
            name = getattr(policy, "name", policy.__class__.__name__)
            seat_assignments.append((policy, name))

        agent_policy_obj, agent_name = seat_assignments[base_config.agent_id]
        agent_fn = agent_from_policy(agent_policy_obj)
        opponents = [
            None if idx == base_config.agent_id else seat_assignments[idx][0]
            for idx in range(base_config.num_players)
        ]
        results = run_schedule(
            agent=agent_fn,
            base_config=base_config,
            opponents=opponents,
            schedule=schedule,
            seed=None if seed is None else int(seed + tournament_idx),
            record_games=record_games,
        )
        if record_games:
            tournament_rounds.append(results)

        totals = [0.0] * base_config.num_players
        for round_result in results:
            points = round_result.round_points or [0] * base_config.num_players
            for idx, value in enumerate(points):
                totals[idx] += value

        max_total = max(totals)
        seat_history.append([name for _, name in seat_assignments])

        for idx, total in enumerate(totals):
            _, name = seat_assignments[idx]
            entry = stats.setdefault(name, PolicyAggregate(name=name))
            entry.seats_played += 1
            entry.total_points += total
            if total == max_total:
                entry.tournament_wins += 1

    rounds_payload = tournament_rounds if record_games else None
    return TournamentBatchResult(policy_stats=stats, seat_history=seat_history, rounds=rounds_payload)
