import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plump_rl import EnvConfig, PlumpEnv, run_schedule


def test_cards_played_history_updates():
    env = PlumpEnv(EnvConfig(hand_size=2, num_players=4))
    obs, info = env.reset()
    total_cards_seen = int(np.sum(obs["cards_played"]))
    done = False
    while not done:
        legal_actions = np.nonzero(info["legal_actions"])[0]
        action = int(legal_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        total_cards_seen = max(total_cards_seen, int(np.sum(obs["cards_played"])))
        done = terminated or truncated
    assert total_cards_seen > 0


def test_cant_say_rule_masks_action():
    env = PlumpEnv(EnvConfig(hand_size=3, num_players=4))
    env.reset()
    env.phase = "estimation"
    env.dealer_id = env.config.agent_id  # ensure cant-say applies to agent seat
    env.estimations = [-1, 1, 1, 0]
    mask = env.get_legal_actions_mask(env.config.agent_id)
    cant_say = env.config.hand_size - (1 + 1 + 0)
    assert mask[cant_say] == 0


def test_run_schedule_records_history():
    results = run_schedule(
        base_config=EnvConfig(hand_size=3, num_players=4),
        record_games=True,
    )
    assert any(round_result.history is not None for round_result in results)
