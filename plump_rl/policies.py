"""Opponent policies used inside the Plump RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .cards import card_rank, card_suit

if TYPE_CHECKING:
    from .env import PlumpEnv

AgentPolicy = Callable[[dict, dict, "PlumpEnv"], int]
RNG = np.random.Generator


@dataclass
class TrickContext:
    """Public information about the current trick."""

    lead_suit: int
    trick_cards: Sequence[int]


class BasePolicy:
    """Interface implemented by all handcrafted opponents."""

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: RNG) -> int:
        """Return the number of tricks this policy expects to win."""

        raise NotImplementedError

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: RNG) -> int:
        """Return the card identifier to play for the given trick."""

        raise NotImplementedError


class RuleBasedPolicy(BasePolicy):
    """Baseline opponent that bids high-card counts and generally plays low cards."""

    def __init__(self, aggressive: bool = False) -> None:
        self.aggressive = aggressive

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: RNG) -> int:
        high_cards = sum(1 for card in hand if card_rank(card) >= 11)
        power_cards = sum(1 for card in hand if card_rank(card) >= 13)
        bid = min(hand_size, max(0, high_cards + (1 if self.aggressive else 0)))
        if power_cards == 0 and bid > 0:
            bid -= 1
        if cant_say is not None and bid == cant_say:
            candidates = [value for value in range(hand_size + 1) if value != cant_say]
            bid = int(rng.choice(candidates))
        return bid

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: RNG) -> int:
        del hand, rng
        lead = ctx.lead_suit
        if lead != -1:
            suited = [card for card in legal_cards if card_suit(card) == lead]
            if suited:
                legal_cards = suited
        key_fn = (lambda cid: -card_rank(cid)) if self.aggressive else card_rank
        return min(legal_cards, key=key_fn)


class _BaseHeuristic(BasePolicy):
    """Common helper functions shared across handcrafted policies."""

    @staticmethod
    def _lead_low(legal_cards: List[int]) -> int:
        return min(legal_cards, key=card_rank)

    @staticmethod
    def _lead_high(legal_cards: List[int]) -> int:
        return max(legal_cards, key=card_rank)


class DressedCardPolicy(_BaseHeuristic):
    """Counts face cards when estimating and prefers playing them early."""

    FACE_THRESHOLD = 11  # J, Q, K, A

    def __init__(self, lead_with_high: bool = True) -> None:
        self.lead_with_high = lead_with_high

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: RNG) -> int:
        estimate = min(hand_size, sum(1 for card in hand if card_rank(card) >= self.FACE_THRESHOLD))
        if cant_say is not None and estimate == cant_say:
            estimate = max(0, estimate - 1)
        return estimate

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: RNG) -> int:
        del hand, rng
        lead = ctx.lead_suit
        if lead == -1:
            return self._lead_high(legal_cards) if self.lead_with_high else self._lead_low(legal_cards)
        suited = [card for card in legal_cards if card_suit(card) == lead]
        if suited:
            high = max(suited, key=card_rank)
            low = min(suited, key=card_rank)
            return high if card_rank(high) >= self.FACE_THRESHOLD else low
        return self._lead_low(legal_cards)


class ZeroBidDodger(_BaseHeuristic):
    """Always bids zero and tries to lose every trick."""

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: RNG) -> int:
        del hand, rng
        if cant_say == 0:
            return 1
        return 0

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: RNG) -> int:
        del hand, rng
        lead = ctx.lead_suit
        if lead == -1:
            return self._lead_low(legal_cards)
        suited = [card for card in legal_cards if card_suit(card) == lead]
        if suited:
            non_faces = [card for card in suited if card_rank(card) < 11]
            pool = non_faces if non_faces else suited
            return min(pool, key=card_rank)
        return self._lead_low(legal_cards)


class ShortSuitAggressor(_BaseHeuristic):
    """Bids the number of short suits and aggressively voids them."""

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: RNG) -> int:
        del hand_size, rng
        suit_counts = [0, 0, 0, 0]
        for card in hand:
            suit_counts[card_suit(card)] += 1
        short_suits = sum(1 for count in suit_counts if count <= 2)
        estimate = max(1, short_suits)
        if cant_say is not None and estimate == cant_say:
            estimate = min(estimate + 1, len(hand))
        return estimate

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: RNG) -> int:
        del rng
        lead = ctx.lead_suit
        if lead == -1:
            suit_counts = {suit: 0 for suit in range(4)}
            for card in hand:
                suit_counts[card_suit(card)] += 1
            target_suit = min(suit_counts, key=lambda suit: (suit_counts[suit], suit))
            suited = [card for card in legal_cards if card_suit(card) == target_suit]
            return max(suited, key=card_rank) if suited else self._lead_high(legal_cards)
        suited = [card for card in legal_cards if card_suit(card) == lead]
        if suited:
            suited.sort(key=card_rank)
            return suited[len(suited) // 2]
        return self._lead_high(legal_cards)


class MiddleManager(_BaseHeuristic):
    """Balances estimation by valuing mid ranks and keeping suit control."""

    MID_LOW = 7
    MID_HIGH = 10

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: RNG) -> int:
        del rng
        mids = sum(1 for card in hand if self.MID_LOW <= card_rank(card) <= self.MID_HIGH)

        suit_pairs = 0
        suit_low_high = {suit: [False, False] for suit in range(4)}
        for card in hand:
            rank = card_rank(card)
            suit = card_suit(card)
            if rank <= 6:
                suit_low_high[suit][0] = True
            if rank >= 12:
                suit_low_high[suit][1] = True
        suit_pairs = sum(1 for has_low, has_high in suit_low_high.values() if has_low and has_high)

        estimate = min(hand_size, mids + suit_pairs)
        if cant_say is not None and estimate == cant_say:
            estimate = max(0, estimate - 1)
        return estimate

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: RNG) -> int:
        del rng
        lead = ctx.lead_suit
        if lead == -1:
            variance_by_suit = {
                suit: self._suit_variance(hand, suit) for suit in range(4)
            }
            best_suit = min(variance_by_suit, key=variance_by_suit.get)
            suited = [card for card in legal_cards if card_suit(card) == best_suit]
            return suited[len(suited) // 2] if suited else self._lead_low(legal_cards)

        suited = [card for card in legal_cards if card_suit(card) == lead]
        if suited:
            sorted_suited = sorted(suited, key=card_rank)
            highest_in_trick = max(
                (card_rank(card) for card in ctx.trick_cards if card_suit(card) == lead and card != -1),
                default=-1,
            )
            higher_than_lead = [card for card in sorted_suited if card_rank(card) > highest_in_trick]
            return higher_than_lead[0] if higher_than_lead else sorted_suited[0]

        longest_suit = max(range(4), key=lambda suit: self._suit_length(hand, suit))
        discard_candidates = [card for card in legal_cards if card_suit(card) == longest_suit]
        return discard_candidates[0] if discard_candidates else self._lead_low(legal_cards)

    @staticmethod
    def _suit_variance(hand: List[int], suit: int) -> int:
        ranks = [card_rank(card) for card in hand if card_suit(card) == suit]
        if not ranks:
            return 0
        return max(ranks) - min(ranks)

    @staticmethod
    def _suit_length(hand: List[int], suit: int) -> int:
        return sum(1 for card in hand if card_suit(card) == suit)


class RandomLegalPolicy(BasePolicy):
    """Uniform random policy that respects the cant-say rule and legal cards."""

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: RNG) -> int:
        del hand
        choices = [value for value in range(hand_size + 1) if value != cant_say]
        if not choices:
            return 0
        return int(rng.choice(choices))

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: RNG) -> int:
        del hand
        lead = ctx.lead_suit
        if lead != -1:
            suited = [card for card in legal_cards if card_suit(card) == lead]
            if suited:
                legal_cards = suited
        else:
            suited = []
        return int(rng.choice(legal_cards))


class SelfPlayOpponent(BasePolicy):
    """Allows a learned agent policy to occupy an opponent seat."""

    def __init__(self, agent_policy: AgentPolicy, name: str = "SelfPlay") -> None:
        self.agent_policy = agent_policy
        self.name = name
        self._env: Optional[PlumpEnv] = None

    def bind(self, env: PlumpEnv) -> None:
        """Attach the opponent to a running environment."""

        self._env = env

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: RNG) -> int:
        del hand, hand_size, cant_say, rng
        return self._invoke_policy()

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: RNG) -> int:
        del hand, legal_cards, ctx, rng
        return self._invoke_policy()

    def _invoke_policy(self) -> int:
        if self._env is None:
            raise RuntimeError("SelfPlayOpponent must be bound to an environment via `bind()`.")
        obs = self._env.get_player_observation(self._env.current_player)
        mask = self._env.get_legal_actions_mask(self._env.current_player)
        info = {
            "legal_actions": mask,
            "dealer": self._env.dealer_id,
            "phase": self._env.phase,
        }
        return int(self.agent_policy(obs, info, self._env))
