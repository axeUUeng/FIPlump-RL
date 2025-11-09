import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger

from plump_rl import (
    DressedCardPolicy,
    EnvConfig,
    MiddleManager,
    RandomLegalPolicy,
    SelfPlayOpponent,
    ShortSuitAggressor,
    ZeroBidDodger,
    RoundResult,
    make_dqn_agent_policy,
    make_dqn_agent_policy_from_state,
    make_ppo_agent_policy,
    round_results_to_dict,
    run_schedule,
    train_dqn,
    train_ppo,
)

POLICY_POOL = [DressedCardPolicy, ShortSuitAggressor, MiddleManager, ZeroBidDodger, RandomLegalPolicy]


def build_opponents(
    config: EnvConfig,
    rng: Optional[np.random.Generator] = None,
    random_prob: float = 0.0,
    checkpoint_policies: Optional[List] = None,
):
    """Return a list of opponent policies for the current run."""
    opponents = []
    checkpoint_policies = checkpoint_policies or []
    for idx in range(config.num_players):
        if idx == config.agent_id:
            opponents.append(None)
            continue
        use_random = False
        if rng is not None:
            use_random = rng.random() < random_prob
        elif random_prob > 0:
            use_random = np.random.rand() < random_prob
        if use_random:
            opponents.append(RandomLegalPolicy())
            continue
        if checkpoint_policies:
            agent_fn = rng.choice(checkpoint_policies) if rng else checkpoint_policies[idx % len(checkpoint_policies)]
            opponents.append(SelfPlayOpponent(agent_fn))
            continue
        if rng is None:
            policy_cls = POLICY_POOL[(idx - 1) % len(POLICY_POOL)]
        else:
            policy_cls = rng.choice(POLICY_POOL)
        opponents.append(policy_cls())
    return opponents


def evaluate_agent(
    agent_fn,
    config: EnvConfig,
    tournaments: int,
    seed: Optional[int],
    record_games: bool = False,
    random_prob: float = 0.0,
    checkpoint_policies: Optional[List] = None,
):
    """Evaluate an agent across multiple tournaments."""
    rng = np.random.default_rng(seed)
    totals: List[float] = []
    seat_assignments: List[List[str]] = []
    tournament_rounds: List[List[RoundResult]] = []
    for t in range(tournaments):
        opponents = build_opponents(
            config,
            rng,
            random_prob=random_prob,
            checkpoint_policies=checkpoint_policies,
        )
        seat_assignments.append(
            ["Agent" if idx == config.agent_id else opp.__class__.__name__ for idx, opp in enumerate(opponents)]
        )
        rounds = run_schedule(
            agent=agent_fn,
            base_config=config,
            opponents=opponents,
            seed=None if seed is None else int(seed + t),
            record_games=record_games,
        )
        total = 0.0
        for round_result in rounds:
            points = round_result.round_points or [0] * config.num_players
            total += points[config.agent_id]
        totals.append(total)
        if record_games:
            tournament_rounds.append(rounds)
    return totals, seat_assignments, tournament_rounds


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a DQN agent for Plump.")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes.")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn", help="RL algorithm to train.")
    parser.add_argument("--num-players", type=int, default=4, help="Number of players in the environment.")
    parser.add_argument("--hand-size", type=int, default=10, help="Cards dealt to each player.")
    parser.add_argument("--agent-id", type=int, default=0, help="Index of the learning agent.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--eval-tournaments", type=int, default=20, help="Number of evaluation tournaments.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training.")
    parser.add_argument("--save-model", type=str, default=None, help="Optional path to save the trained policy network.")
    parser.add_argument("--record-games", type=str, default=None, help="If set, dump evaluation game histories to this JSON file.")
    parser.add_argument("--random-opponent-prob", type=float, default=0.25, help="Probability that any opponent seat is filled by a random legal policy.")
    parser.add_argument(
        "--self-play-checkpoints",
        nargs="*",
        default=[],
        help="Paths to saved agent checkpoints that will be sampled as opponents (self-play).",
    )
    args = parser.parse_args()

    config = EnvConfig(num_players=args.num_players, hand_size=args.hand_size, agent_id=args.agent_id)
    self_play_policies = []
    for path in args.self_play_checkpoints:
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            logger.warning("Checkpoint {} not found, skipping.", ckpt_path)
            continue
        state = torch.load(ckpt_path, map_location="cpu")
        if args.algo == "dqn":
            policy = make_dqn_agent_policy_from_state(state, config)
        else:
            policy = make_ppo_agent_policy(state, config)
        self_play_policies.append(policy)

    training_opponents = build_opponents(
        config,
        random_prob=args.random_opponent_prob,
        checkpoint_policies=self_play_policies,
    )

    logger.info(
        "Training {} agent for {} episodes (players={}, hand_size={})",
        args.algo.upper(),
        args.episodes,
        args.num_players,
        args.hand_size,
    )
    if args.algo == "dqn":
        training_result = train_dqn(
            num_episodes=args.episodes,
            config=config,
            opponents=training_opponents,
            seed=args.seed,
            show_progress=True,
        )
        agent_policy = make_dqn_agent_policy(training_result.agent, config)
        model_state = training_result.agent.policy_net.state_dict()
    else:
        training_result = train_ppo(
            num_episodes=args.episodes,
            config=config,
            opponents=training_opponents,
            seed=args.seed,
            show_progress=True,
        )
        agent_policy = make_ppo_agent_policy(training_result.model_state, config)
        model_state = training_result.model_state

    if args.save_model:
        path = Path(args.save_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_state, path)
        logger.success("Saved policy network to {}", path)

    if args.skip_eval:
        return

    agent_fn = agent_policy
    record_games = args.record_games is not None
    totals, seat_history, tournament_rounds = evaluate_agent(
        agent_fn,
        config,
        args.eval_tournaments,
        args.seed,
        record_games=record_games,
        random_prob=args.random_opponent_prob,
        checkpoint_policies=self_play_policies,
    )
    avg = float(np.mean(totals)) if totals else 0.0
    std = float(np.std(totals)) if totals else 0.0
    logger.info(
        "Evaluation across {} tournaments: avg_points={:.2f} ± {:.2f}",
        len(totals),
        avg,
        std,
    )
    if totals:
        logger.info("Individual totals: {}", ", ".join(f"{t:.1f}" for t in totals))

    if record_games and tournament_rounds:
        path = Path(args.record_games)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for idx, (rounds, seats) in enumerate(zip(tournament_rounds, seat_history)):
            payload.append(
                {
                    "tournament_index": idx,
                    "seat_policies": seats,
                    "rounds": round_results_to_dict(rounds),
                }
            )
        path.write_text(json.dumps(payload, indent=2))
        logger.success("Recorded {} tournaments to {}", len(payload), path)


if __name__ == "__main__":
    main()
