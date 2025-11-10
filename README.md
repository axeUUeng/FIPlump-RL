# FIPlump-RL
Gymnasium-compatible tooling for training reinforcement-learning agents to play the Swedish trick-taking game **Plump**.

---

## Project layout
- `plump_rl/`: production code – environment, card utilities, heuristic opponents, tournament simulators, and PyTorch trainers.
- `plump.ipynb`, `PLUMP_RL.ipynb`: scratchpads for experiments and visualizations.
- `legacy/`: the “code museum” containing the original card utilities, Hearts env, and early Plump drafts (kept for reference only).

---

## Game rules (house variant)
1. **Deck / players** – standard 52-card deck, 3–5 players (configurable). Each round deals the same number of cards to every player.
2. **Round schedule** – common tournament pattern is 10 cards down to 1, then back up to 10 (implemented via `default_schedule` / `run_schedule`).
3. **Estimation phase** – starting left of the dealer and ending with the dealer, each player estimates how many tricks they will win. The dealer must avoid the “cant-say” value that would make the sum of estimates equal the number of tricks in that round.
4. **Trick play** – highest card of the led suit wins (no trump). Players must follow suit if possible; otherwise they may discard any card. Trick winners lead next.
5. **Scoring** – when a player’s actual tricks match their estimate they receive `match_bonus + tricks_won` points (default 10 + tricks). A successful zero bid pays a flat `zero_bid_bonus` (default 5). Otherwise they score `0`. Invalid actions (e.g., bidding the cant-say number) terminate the episode with a small negative reward (default -5).

---

## Environment interface
```python
import numpy as np
from plump_rl import PlumpEnv, EnvConfig

env = PlumpEnv(EnvConfig(num_players=4, hand_size=10))
obs, info = env.reset()
legal = np.nonzero(info["legal_actions"])[0]
```

### Observations
When flattened for DQN training, the state vector contains:

| Component | Description |
|-----------|-------------|
| `phase` | 0 (estimation) or 1 (play). |
| `hand` | 52-length binary mask of the agent’s cards. |
| `current_trick` | Card IDs (0–51, normalized) for each seat this trick; `-1` if empty. |
| `lead_suit` | Suit index (0–3) or 4 for “no suit yet”. |
| `estimations` | Current bids per player, normalized by hand size. |
| `tricks_won` | Tricks collected per player, normalized. |
| `cards_remaining` | Cards left per player, normalized. |
| `tricks_played` | Completed tricks in this round, normalized. |
| `cards_played` | 52-length history mask of every card already seen in the round. |

### Action space
Discrete with two segments:
1. `0 .. hand_size` – estimation choices (dealer’s cant-say rule enforced automatically).
2. `hand_size+1 .. hand_size+52` – play phase: select a card by ID (`action - (hand_size+1)`).

`info["legal_actions"]` provides a binary mask over the entire action space so you can zero logits for invalid moves.

### Rewards
- Intermediate steps return `0`.
- End-of-round reward equals the scoring rule above.
- Illegal actions end the episode with `-invalid_action_penalty` (default `-5`).

---

## Heuristic opponents
Every heuristic extends `BasePolicy` with `estimate()` and `play()`:

| Policy | Behaviour summary |
|--------|-------------------|
| `RuleBasedPolicy` | Bids using high-card counts, plays conservatively (low when following suit). |
| `DressedCardPolicy` | Counts face cards (“klädda”) for bids, leads high face cards, dumps low off-suit cards. |
| `ZeroBidDodger` | Always bids 0 (unless cant-say forces 1) and aggressively loses tricks. |
| `ShortSuitAggressor` | Bids on the number of short suits (≤2 cards) and tries to void suits quickly. |
| `MiddleManager` | Values mid ranks (7–10) and tries to maintain suit control with balanced play. |
| `RandomLegalPolicy` | Samples bids and cards uniformly from the legal set; good chaotic opponent baseline. |

Plug them into the environment via the `opponents` argument or use them as baselines in tournaments.

---

## Tournament helpers
- `run_schedule(...)` – plays a full hand-size schedule (default 10→1→10) for a single agent policy and returns per-round results.
- `simulate_random_tournaments(...)` – runs many tournaments with random heuristic assignments per seat and aggregates stats (average points, wins, seat counts).

Example:
```python
from plump_rl import EnvConfig, DressedCardPolicy, run_schedule, format_round_history

opponents = [None, DressedCardPolicy(), DressedCardPolicy(), DressedCardPolicy()]
results = run_schedule(agent=None, base_config=EnvConfig(hand_size=10), opponents=opponents, record_games=True)
for round_result in results:
    if round_result.history:
        print(format_round_history(round_result.history))
```
(Provide an `agent` callable to control the learning seat; omit it to auto-play with the provided opponents. Use `record_games=True` to capture trick-by-trick histories.)

---

## PyTorch DQN tooling
### Library API
```python
from plump_rl import EnvConfig, DressedCardPolicy, train_dqn, train_ppo

opponents = [None, DressedCardPolicy(), DressedCardPolicy(), DressedCardPolicy()]
dqn_result = train_dqn(
    num_episodes=500,
    config=EnvConfig(num_players=4, hand_size=10),
    opponents=opponents,
)
ppo_result = train_ppo(
    num_updates=500,
    rollout_steps=256,
    config=EnvConfig(num_players=4, hand_size=10),
    opponents=opponents,
)
print("DQN last-50 avg:", sum(dqn_result.episode_rewards[-50:]) / 50)
print("PPO last-50 avg:", sum(ppo_result.episode_rewards[-50:]) / 50)
```
`train_dqn` returns all episode rewards plus the trained `DQNAgent`, so you can wrap it with `make_dqn_agent_policy(...)` and evaluate in tournaments. `train_ppo` supplies a policy/value model state that can be passed to `make_ppo_agent_policy(...)`.

> Training episodes play a **single round** at the configured `hand_size`. To mimic the full 10→1→10 schedule, use `run_schedule(...)` or `simulate_random_tournaments(...)` for evaluation/validation.

### CLI script
```
python main.py --episodes 800 --num-players 4 --hand-size 10 --eval-tournaments 30 \
  --save-model checkpoints/dqn.pt --record-games eval_log.json
```
- `--episodes`: number of single-round training episodes at the chosen hand size.
- `--num-players`, `--hand-size`, `--agent-id`: configure the environment layout.
- `--algo`: choose between `dqn` and `ppo`.
- `--random-opponent-prob`: probability that any opponent seat is controlled by `RandomLegalPolicy` (otherwise a heuristic policy is used).
- `--save-model`: optional path for persisting the PyTorch policy network.
- `--eval-tournaments`: how many 10→1→10 schedules to play after training.
- `--record-games`: if set, writes every evaluation round (including trick history) to the given JSON file so you can replay specific games later.
- `--self-play-checkpoints`: list of saved agent weights that can be sampled as self-play opponents; useful for progressive training.
- All status messages go through Loguru, and `train_dqn` shows a tqdm bar by default.

### Offline evaluation
To evaluate a saved agent without retraining, use the helper script:
```
python scripts/eval_agent.py --checkpoint checkpoints/dqn.pt --algo dqn --tournaments 50 \
  --num-players 4 --hand-size 10 --stats-output logs/stats.json --record-games logs/eval.json
```
It runs tournaments against the heuristic/random pool, prints aggregate metrics (mean/std of points, hand-size breakdowns), and optionally stores raw round histories or summary JSON.

---

## Dependencies
- Python 3.10+
- `gymnasium`
- `numpy`
- `torch`
- `torchvision`
- `tqdm`
- `loguru`
- `loguru`
