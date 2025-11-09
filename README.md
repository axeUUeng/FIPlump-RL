# FIPlump-RL
Fédération Internationale de Plump presents RL in the card game Plump.

## Project layout
- `plump_rl/`: Gymnasium-compatible environment, opponent policies, tournament helpers, and card utilities for training agents.
- `plump.ipynb` / `PLUMP_RL.ipynb`: notebooks used for experimenting with heuristics and RL agents.
- `legacy/`: the original card/hand/player utilities (Hearts env, early Plump drafts, notebooks, etc.) preserved for reference.

## Using the new environment
```python
import numpy as np
from plump_rl import PlumpEnv

env = PlumpEnv()
obs, info = env.reset()
done = False
while not done:
    legal = np.nonzero(info["legal_actions"])[0]
    action = int(np.random.choice(legal))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

Observations expose the agent hand, public trick state, estimations, and meta data such as `phase`. The action space has two segments:

1. `0..hand_size` for estimation phase (cant-say rule enforced for the dealer).
2. `hand_size+1..hand_size+52` for selecting a specific card (offset + card id 0-51).

`info["legal_actions"]` masks the invalid entries so common RL libraries can respect turn-specific constraints.

Scoring currently awards `match_bonus + tricks_won` when a player hits their estimate and `0` otherwise, mirroring the house rules described.

### Tournament helpers
To simulate the classic schedule (10 → 1 → 10) without manually reconfiguring the environment each round:
```python
from plump_rl import EnvConfig, run_schedule

results = run_schedule(base_config=EnvConfig(hand_size=10))
for r in results:
    print(f"hand_size={r.hand_size} reward={r.reward} points={r.round_points}")
```
Provide your own `agent` callable to replace the default random policy (for example, use `DressedCardPolicy` for a face-card counting heuristic) or supply custom opponent policies via the `opponents` argument.

To sample full tournaments with random heuristic assignments:
```python
from plump_rl import EnvConfig, DressedCardPolicy, ZeroBidDodger, ShortSuitAggressor, MiddleManager
from plump_rl.tournament import simulate_random_tournaments

policy_pool = [DressedCardPolicy, ZeroBidDodger, ShortSuitAggressor, MiddleManager]
result = simulate_random_tournaments(
    100,
    policy_factories=policy_pool,
    base_config=EnvConfig(num_players=5, hand_size=10),
)

for name, stats in result.policy_stats.items():
    print(f"{name:20s} avg_points={stats.average_points:.2f} wins={stats.tournament_wins}/{stats.seats_played}")
```

### Available opponent policies
- `RuleBasedPolicy`: conservative baseline using high-card counts.
- `DressedCardPolicy`: estimates via face cards and plays straighter.
- `ZeroBidDodger`: always bids 0 and tries to shed every trick.
- `ShortSuitAggressor`: bids based on short suits, pushing aggressive leads in those suits.
- `MiddleManager`: balances estimation using mid ranks and plays around suit control.

### PyTorch DQN quick start
```python
from plump_rl import train_dqn, EnvConfig, DressedCardPolicy

# Optional: pit the learning agent against heuristics
opponents = [None, DressedCardPolicy(), DressedCardPolicy(), DressedCardPolicy()]
result = train_dqn(
    num_episodes=500,
    config=EnvConfig(num_players=4, hand_size=10),
    opponents=opponents,
)
print("Final average reward:", sum(result.episode_rewards[-50:]) / 50)
```
`train_dqn` returns all per-episode rewards so you can visualize learning curves or checkpoint models for later evaluation.

### Dependencies
- Python 3.10+
- `gymnasium`
- `numpy`
- `torch`
