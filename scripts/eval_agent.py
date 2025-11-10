"""Evaluate a trained Plump agent against heuristic/random opponents."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plump_rl import (
    DressedCardPolicy,
    EnvConfig,
    MiddleManager,
    RandomLegalPolicy,
    ShortSuitAggressor,
    ZeroBidDodger,
    make_dqn_agent_policy_from_state,
    make_ppo_agent_policy,
    load_agent_policy_from_state,
    run_schedule,
)

POLICY_POOL = [DressedCardPolicy, ShortSuitAggressor, MiddleManager, ZeroBidDodger]


def build_opponents(
    config: EnvConfig,
    rng: Optional[np.random.Generator],
    random_prob: float,
) -> List[Optional[RandomLegalPolicy]]:
    opponents: List[Optional[RandomLegalPolicy]] = []
    for seat in range(config.num_players):
        if seat == config.agent_id:
            opponents.append(None)
            continue
        use_random = rng.random() < random_prob if rng is not None else False
        if use_random:
            opponents.append(RandomLegalPolicy())
            continue
        policy_cls = rng.choice(POLICY_POOL) if rng is not None else POLICY_POOL[seat % len(POLICY_POOL)]
        opponents.append(policy_cls())
    return opponents


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        logger.warning("Retrying checkpoint load for %s with weights_only=False (trusted source assumed).", path)
        return torch.load(path, map_location="cpu", weights_only=False)


def load_agent_policy(checkpoint: Path, algo: str, config: EnvConfig):
    state = load_checkpoint(checkpoint)
    if algo == "dqn":
        return make_dqn_agent_policy_from_state(state, config)
    if algo == "ppo":
        return make_ppo_agent_policy(state, config)
    raise ValueError(f"Unsupported algorithm: {algo}")


def collect_stats(rounds, agent_id: int):
    totals = 0.0
    hand_points: Dict[int, List[float]] = {}
    for round_result in rounds:
        points = round_result.round_points or [0] * (agent_id + 1)
        totals += points[agent_id]
        hand_points.setdefault(round_result.hand_size, []).append(points[agent_id])
    return totals, hand_points


def simulate_tournament(
    tournament_idx: int,
    config: EnvConfig,
    agent_state: dict,
    algo: str,
    random_prob: float,
    record_games: bool,
    seed: Optional[int],
) -> Tuple[int, List, List[Optional[RandomLegalPolicy]], float, Dict[int, List[float]]]:
    rng = np.random.default_rng(seed)
    if algo == "dqn":
        policy = make_dqn_agent_policy_from_state(agent_state, config)
    else:
        policy = make_ppo_agent_policy(agent_state, config)
    opponents = build_opponents(config, rng, random_prob)
    results = run_schedule(
        agent=policy,
        base_config=config,
        opponents=opponents,
        seed=rng.integers(0, 1_000_000) if rng is not None else None,
        record_games=record_games,
    )
    total_points, hand_stats = collect_stats(results, config.agent_id)
    return tournament_idx, results, opponents, total_points, hand_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Plump agent.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained agent weights.")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn", help="Agent architecture.")
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--hand-size", type=int, default=10)
    parser.add_argument("--agent-id", type=int, default=0)
    parser.add_argument("--tournaments", type=int, default=20)
    parser.add_argument("--random-opponent-prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record-games", type=str, default=None, help="Optional path to dump round histories.")
    parser.add_argument("--stats-output", type=str, default=None, help="Optional path to write JSON stats.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()

    config = EnvConfig(
        num_players=args.num_players,
        hand_size=args.hand_size,
        agent_id=args.agent_id,
    )

    tournaments: List[Tuple[int, List, List[Optional[RandomLegalPolicy]]]] = []
    totals = []
    per_hand: Dict[int, List[float]] = {}
    per_policy: Dict[str, List[float]] = defaultdict(list)
    agent_state = load_checkpoint(Path(args.checkpoint))
    record_games_flag = bool(args.record_games)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for idx in range(args.tournaments):
            seed = args.seed + idx if args.seed is not None else None
            futures.append(
                executor.submit(
                    simulate_tournament,
                    idx,
                    config,
                    agent_state,
                    args.algo,
                    args.random_opponent_prob,
                    record_games_flag,
                    seed,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating tournaments"):
            idx, results, opponents, total_points, hand_stats = future.result()
            tournaments.append((idx, results, opponents))
            totals.append(total_points)
            for hand_size, values in hand_stats.items():
                per_hand.setdefault(hand_size, []).extend(values)
            for seat, opponent in enumerate(opponents):
                if seat == config.agent_id or opponent is None:
                    continue
                per_policy[opponent.__class__.__name__].append(total_points)

    tournaments.sort(key=lambda item: item[0])
    totals_arr = np.array(totals, dtype=np.float32)
    logger.info(
        "Evaluated {} tournaments | avg_points={:.2f} ± {:.2f}",
        args.tournaments,
        totals_arr.mean(),
        totals_arr.std(),
    )
    for hand_size in sorted(per_hand):
        values = np.array(per_hand[hand_size], dtype=np.float32)
        logger.info(
            "Hand size {:2d}: mean={:.2f} std={:.2f} count={}",
            hand_size,
            values.mean(),
            values.std(),
            len(values),
        )

    if args.record_games:
        path = Path(args.record_games)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for idx, rounds, opponents in tournaments:
            payload.append(
                {
                    "tournament_index": idx,
                    "opponents": [
                        opponent.__class__.__name__ if opponent is not None else "Agent"
                        for opponent in opponents
                    ],
                    "rounds": [
                        {
                            "hand_size": round_result.hand_size,
                            "reward": round_result.reward,
                            "round_points": round_result.round_points,
                            "history": round_result.history,
                        }
                        for round_result in rounds
                    ],
                }
            )
        path.write_text(json.dumps(payload, indent=2))
        logger.success("Stored {} tournament histories in {}", len(payload), path)

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "avg_points": float(totals_arr.mean()),
            "std_points": float(totals_arr.std()),
            "hand_stats": {
                str(hand_size): {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": len(values),
                }
                for hand_size, values in per_hand.items()
            },
            "opponent_stats": {
                name: {
                    "mean": float(np.mean(values)) if values else 0.0,
                    "std": float(np.std(values)) if values else 0.0,
                    "count": len(values),
                }
                for name, values in per_policy.items()
            },
        }
        stats_path.write_text(json.dumps(summary, indent=2))
        logger.success("Wrote summary statistics to {}", stats_path)


if __name__ == "__main__":
    main()
