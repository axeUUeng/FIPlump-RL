"""Gymnasium-compatible environment for Plump."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils import seeding

from .cards import DECK_SIZE, card_rank, card_suit, card_to_str, hand_to_binary
from .policies import BasePolicy, RuleBasedPolicy, TrickContext


@dataclass
class EnvConfig:
    num_players: int = 4
    hand_size: int = 10
    agent_id: int = 0
    match_bonus: int = 10
    zero_bid_bonus: int = 5
    invalid_action_penalty: float = 5.0
    record_history: bool = False


class PlumpEnv(Env):
    """Single-agent RL environment for Plump card play."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: EnvConfig | None = None,
        opponents: Optional[Sequence[Optional[BasePolicy]]] = None,
        seed: Optional[int] = None,
        record_history: Optional[bool] = None,
    ):
        self.config = config or EnvConfig()
        assert self.config.num_players >= 2, "Need at least two players"
        assert (
            self.config.hand_size * self.config.num_players <= DECK_SIZE
        ), "Not enough cards in deck for the requested hand size"
        assert 0 <= self.config.agent_id < self.config.num_players, "Invalid agent id"
        self.record_history = self.config.record_history if record_history is None else record_history

        self.estimation_action_count = self.config.hand_size + 1
        self.action_space = spaces.Discrete(self.estimation_action_count + DECK_SIZE)
        self.observation_space = spaces.Dict(
            {
                "phase": spaces.Discrete(2),
                "hand": spaces.MultiBinary(DECK_SIZE),
                "current_trick": spaces.Box(
                    low=-1, high=DECK_SIZE, shape=(self.config.num_players,), dtype=np.int16
                ),
                "lead_suit": spaces.Discrete(5),  # 0-3 suits, 4=no lead yet
                "estimations": spaces.Box(
                    low=-1, high=self.config.hand_size, shape=(self.config.num_players,), dtype=np.int8
                ),
                "tricks_won": spaces.Box(
                    low=0, high=self.config.hand_size, shape=(self.config.num_players,), dtype=np.int8
                ),
                "cards_remaining": spaces.Box(
                    low=0, high=self.config.hand_size, shape=(self.config.num_players,), dtype=np.int8
                ),
                "tricks_played": spaces.Discrete(self.config.hand_size + 1),
            }
        )

        self.np_random, _ = seeding.np_random(seed)
        self.opponents = self._build_opponents(opponents)

        # Mutable state placeholders
        self.hands: List[List[int]] = []
        self.estimations: List[int] = []
        self.tricks_won: List[int] = []
        self.trick_cards: List[int] = []
        self.trick_leader: int = -1
        self.lead_suit: int = -1
        self.trick_progress: int = 0
        self.tricks_played: int = 0
        self.phase: str = "estimation"
        self.current_player: int = 0
        self.dealer_id: int = self.config.num_players - 1
        self.estimation_count: int = 0
        self.done: bool = False
        self.last_reward: float = 0.0
        self.final_points: Optional[List[int]] = None
        self.round_counter: int = 0
        self.current_round_log: Optional[Dict] = None
        self.completed_round_logs: List[Dict] = []

    def _build_opponents(self, opponents: Optional[Sequence[Optional[BasePolicy]]]) -> List[Optional[BasePolicy]]:
        if opponents is None:
            result: List[Optional[BasePolicy]] = []
            for idx in range(self.config.num_players):
                result.append(None if idx == self.config.agent_id else RuleBasedPolicy())
            return result
        assert len(opponents) == self.config.num_players, "Opponents list must match player count"
        built: List[Optional[BasePolicy]] = []
        for idx, policy in enumerate(opponents):
            if idx == self.config.agent_id:
                built.append(None)
            else:
                built.append(policy or RuleBasedPolicy())
        return built

    # Recording helpers -------------------------------------------------------
    def _start_round_log(self):
        if not self.record_history:
            return
        self.current_round_log = {
            "round_index": self.round_counter,
            "hand_size": self.config.hand_size,
            "dealer": self.dealer_id,
            "players": list(range(self.config.num_players)),
            "estimations": [],
            "plays": [],
            "tricks": [],
        }

    def _log_estimation(self, player_id: int, estimate: int):
        if not self.record_history or self.current_round_log is None:
            return
        self.current_round_log["estimations"].append(
            {"player": player_id, "estimate": estimate}
        )

    def _log_card_play(self, player_id: int, card_id: int):
        if not self.record_history or self.current_round_log is None:
            return
        self.current_round_log["plays"].append(
            {
                "player": player_id,
                "card": card_id,
                "card_str": card_to_str(card_id),
                "trick_index": self.tricks_played,
                "lead_suit": self.lead_suit if self.lead_suit != -1 else None,
            }
        )

    def _log_trick_end(self, winner: int):
        if not self.record_history or self.current_round_log is None:
            return
        cards_snapshot = [
            {
                "player": idx,
                "card": card,
                "card_str": card_to_str(card) if card != -1 else None,
            }
            for idx, card in enumerate(self.trick_cards)
        ]
        self.current_round_log["tricks"].append(
            {
                "trick_index": self.tricks_played,
                "winner": winner,
                "cards": cards_snapshot,
            }
        )

    def _finalize_round_log(self):
        if not self.record_history or self.current_round_log is None:
            return
        self.current_round_log["round_points"] = list(self.final_points or [])
        self.current_round_log["tricks_won"] = list(self.tricks_won)
        self.current_round_log["estimations_final"] = list(self.estimations)
        self.completed_round_logs.append(self.current_round_log)
        self.current_round_log = None

    # Gymnasium API -----------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.dealer_id = (self.dealer_id + 1) % self.config.num_players
        self._deal_new_round()
        self._advance_until_agent_turn()
        return self._get_observation(), self._info()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Call reset() before using the environment again.")

        if self.current_player != self.config.agent_id:
            raise RuntimeError("Environment is waiting for an opponent action; did you forget to call reset?")

        if self.phase == "estimation":
            result = self._handle_agent_estimation(action)
        else:
            result = self._handle_agent_play(action)

        if result is not None:
            return result

        self._advance_until_agent_turn()
        reward = self.last_reward if self.done else 0.0
        terminated = self.done
        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._info()

    # Round helpers -----------------------------------------------------------------
    def _deal_new_round(self):
        self.round_counter += 1
        if self.record_history:
            self._start_round_log()
        cards_needed = self.config.num_players * self.config.hand_size
        deck = np.arange(DECK_SIZE)
        self.np_random.shuffle(deck)
        deck = deck[:cards_needed].tolist()
        self.hands = []
        for idx in range(self.config.num_players):
            start = idx * self.config.hand_size
            stop = start + self.config.hand_size
            hand = sorted(deck[start:stop])
            self.hands.append(hand)

        self.estimations = [-1] * self.config.num_players
        self.tricks_won = [0] * self.config.num_players
        self.trick_cards = [-1] * self.config.num_players
        self.trick_progress = 0
        self.tricks_played = 0
        self.trick_leader = -1
        self.lead_suit = -1
        self.phase = "estimation"
        self.estimation_count = 0
        self.done = False
        self.last_reward = 0.0
        self.final_points = None
        self.current_player = (self.dealer_id + 1) % self.config.num_players
        if self.record_history and self.current_round_log is not None:
            self.current_round_log["start_player"] = self.current_player

    def _handle_agent_estimation(self, action: int):
        if not 0 <= action <= self.config.hand_size:
            return self._invalid_step("Estimation must be between 0 and hand size.")

        cant_say = self._cant_say_value(self.config.agent_id)
        if cant_say is not None and action == cant_say:
            return self._invalid_step(f"Estimation {action} violates cant-say rule ({cant_say}).")

        self.estimations[self.config.agent_id] = action
        self._log_estimation(self.config.agent_id, action)
        self.estimation_count += 1
        self.current_player = (self.config.agent_id + 1) % self.config.num_players
        if self.estimation_count == self.config.num_players:
            self._start_play_phase()
        return None

    def _handle_agent_play(self, action: int):
        card_id = action - self.estimation_action_count
        legal = self._legal_cards(self.config.agent_id)
        if card_id < 0 or card_id >= DECK_SIZE or card_id not in legal:
            return self._invalid_step("Played card is not legal.")

        self._execute_card(self.config.agent_id, card_id)
        return None

    def _advance_until_agent_turn(self):
        while not self.done and self.current_player != self.config.agent_id:
            if self.phase == "estimation":
                self._auto_estimation(self.current_player)
            else:
                self._auto_play(self.current_player)

    def _auto_estimation(self, player_id: int):
        policy = self.opponents[player_id]
        assert policy is not None, "Missing policy for opponent"
        cant_say = self._cant_say_value(player_id)
        estimate = policy.estimate(self.hands[player_id], self.config.hand_size, cant_say, self.np_random)
        if cant_say is not None and estimate == cant_say:
            alternatives = [v for v in range(self.config.hand_size + 1) if v != cant_say]
            estimate = int(self.np_random.choice(alternatives))
        self.estimations[player_id] = estimate
        self._log_estimation(player_id, estimate)
        self.estimation_count += 1
        self.current_player = (player_id + 1) % self.config.num_players
        if self.estimation_count == self.config.num_players:
            self._start_play_phase()

    def _auto_play(self, player_id: int):
        policy = self.opponents[player_id]
        assert policy is not None, "Missing policy for opponent"
        legal = self._legal_cards(player_id)
        ctx = TrickContext(lead_suit=self.lead_suit, trick_cards=tuple(self.trick_cards))
        card = policy.play(self.hands[player_id], legal, ctx, self.np_random)
        if card not in legal:
            card = legal[0]
        self._execute_card(player_id, card)

    def _start_play_phase(self):
        self.phase = "play"
        self.current_player = (self.dealer_id + 1) % self.config.num_players
        self.lead_suit = -1
        self.trick_leader = self.current_player
        self.trick_cards = [-1] * self.config.num_players
        self.trick_progress = 0

    def _execute_card(self, player_id: int, card_id: int):
        hand = self.hands[player_id]
        hand.remove(card_id)
        self.trick_cards[player_id] = card_id
        if self.trick_progress == 0:
            self.lead_suit = card_suit(card_id)
            self.trick_leader = player_id
        self.trick_progress += 1
        self._log_card_play(player_id, card_id)

        if self.trick_progress == self.config.num_players:
            winner = self._resolve_trick()
            self.tricks_won[winner] += 1
            self.tricks_played += 1
            self.trick_cards = [-1] * self.config.num_players
            self.trick_progress = 0
            self.lead_suit = -1
            self.current_player = winner
            self.trick_leader = winner
            if self.tricks_played == self.config.hand_size:
                self._finalize_round()
        else:
            self.current_player = (player_id + 1) % self.config.num_players

    def _resolve_trick(self) -> int:
        lead = self.lead_suit
        winner = self.trick_leader
        winning_rank = card_rank(self.trick_cards[winner])
        for pid, card in enumerate(self.trick_cards):
            if card == -1 or card_suit(card) != lead:
                continue
            rank = card_rank(card)
            if rank > winning_rank:
                winning_rank = rank
                winner = pid
        self._log_trick_end(winner)
        return winner

    def _finalize_round(self):
        self.done = True
        self.final_points = [self._score_player(pid) for pid in range(self.config.num_players)]
        self.last_reward = float(self.final_points[self.config.agent_id])
        self._finalize_round_log()

    def _score_player(self, player_id: int) -> int:
        est = self.estimations[player_id]
        actual = self.tricks_won[player_id]
        if est == actual:
            if actual == 0:
                return self.config.zero_bid_bonus
            return self.config.match_bonus + actual
        return 0

    # Utilities ---------------------------------------------------------------------
    def _cant_say_value(self, player_id: int) -> Optional[int]:
        if player_id != self.dealer_id or self.phase != "estimation":
            return None
        total = sum(e for e in self.estimations if e != -1)
        remaining = self.config.hand_size - total
        if 0 <= remaining <= self.config.hand_size:
            return remaining
        return None

    def _legal_cards(self, player_id: int) -> List[int]:
        hand = self.hands[player_id]
        if self.lead_suit == -1:
            return list(hand)
        suited = [card for card in hand if card_suit(card) == self.lead_suit]
        return suited if suited else list(hand)

    def _invalid_step(self, message: str):
        self.done = True
        self.last_reward = -float(self.config.invalid_action_penalty)
        info = self._info()
        info["error"] = message
        info["invalid_action"] = True
        return self._get_observation(), self.last_reward, True, False, info

    def _get_observation(self):
        obs = {
            "phase": 0 if self.phase == "estimation" else 1,
            "hand": np.array(hand_to_binary(self.hands[self.config.agent_id]), dtype=np.int8),
            "current_trick": np.array(self.trick_cards, dtype=np.int16),
            "lead_suit": self.lead_suit if self.lead_suit != -1 else 4,
            "estimations": np.array(self.estimations, dtype=np.int8),
            "tricks_won": np.array(self.tricks_won, dtype=np.int8),
            "cards_remaining": np.array([len(hand) for hand in self.hands], dtype=np.int8),
            "tricks_played": np.int8(self.tricks_played),
        }
        return obs

    def _legal_actions_mask(self):
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        if self.done:
            return mask
        if self.phase == "estimation":
            mask[: self.estimation_action_count] = 1
            cant_say = self._cant_say_value(self.config.agent_id)
            if cant_say is not None and 0 <= cant_say < self.estimation_action_count:
                mask[cant_say] = 0
        else:
            for card in self._legal_cards(self.config.agent_id):
                mask[self.estimation_action_count + card] = 1
        return mask

    def _info(self):
        info = {
            "phase": self.phase,
            "current_player": self.current_player,
            "dealer": self.dealer_id,
            "estimations": list(self.estimations),
            "tricks_won": list(self.tricks_won),
            "tricks_played": self.tricks_played,
            "legal_actions": self._legal_actions_mask(),
        }
        if self.final_points is not None:
            info["round_points"] = list(self.final_points)
        return info

    # Rendering ---------------------------------------------------------------------
    def render(self, mode: str = "ansi"):
        del mode
        trick_strings = [card_to_str(c) if c != -1 else "--" for c in self.trick_cards]
        return (
            f"Phase: {self.phase}, Dealer: {self.dealer_id}, Current: {self.current_player}\n"
            f"Estimations: {self.estimations}\n"
            f"Tricks won: {self.tricks_won}\n"
            f"Current trick: {trick_strings}"
        )
