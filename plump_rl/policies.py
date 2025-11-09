"""Simple opponent policies for Plump RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from .cards import card_rank, card_suit


@dataclass
class TrickContext:
    lead_suit: int
    trick_cards: Sequence[int]


class BasePolicy:
    """Interface for opponents used in the environment."""

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: np.random.Generator) -> int:
        raise NotImplementedError

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: np.random.Generator) -> int:
        raise NotImplementedError


class RuleBasedPolicy(BasePolicy):
    """Heuristic opponent that plays conservatively."""

    def __init__(self, aggressive: bool = False):
        self.aggressive = aggressive

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: np.random.Generator) -> int:
        # Count strong cards as a proxy for trick potential.
        high_cards = sum(1 for c in hand if card_rank(c) >= 11)
        trumps = sum(1 for c in hand if card_rank(c) >= 13)
        base = min(hand_size, max(0, high_cards + (1 if self.aggressive else 0)))
        if trumps == 0 and base > 0:
            base -= 1
        # Ensure estimate complies with cant_say.
        if cant_say is not None and base == cant_say:
            options = [e for e in range(hand_size + 1) if e != cant_say]
            base = int(rng.choice(options))
        return base

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: np.random.Generator) -> int:
        # Prefer following suit with the lowest card; otherwise discard lowest overall.
        lead = ctx.lead_suit
        if lead != -1:
            suited = [card for card in legal_cards if card_suit(card) == lead]
            if suited:
                legal_cards = suited
        key_fn = (lambda cid: (-card_rank(cid))) if self.aggressive else (lambda cid: card_rank(cid))
        return min(legal_cards, key=key_fn)


class _BaseHeuristic(BasePolicy):
    """Common helpers shared across handcrafted heuristics."""

    def _lead_low(self, legal_cards: List[int]) -> int:
        return min(legal_cards, key=card_rank)

    def _lead_high(self, legal_cards: List[int]) -> int:
        return max(legal_cards, key=card_rank)

    def _follow_low(self, legal_cards: List[int], lead_suit: int) -> int:
        suited = [c for c in legal_cards if card_suit(c) == lead_suit]
        return min(suited, key=card_rank) if suited else min(legal_cards, key=card_rank)

    def _follow_high(self, legal_cards: List[int], lead_suit: int) -> int:
        suited = [c for c in legal_cards if card_suit(c) == lead_suit]
        return max(suited, key=card_rank) if suited else max(legal_cards, key=card_rank)


class DressedCardPolicy(_BaseHeuristic):
    """Estimates tricks via face-card counts (“klädda kort”) and plays simple suit-aware heuristics."""

    FACE_THRESHOLD = 11  # J, Q, K, A

    def __init__(self, lead_with_high: bool = True):
        self.lead_with_high = lead_with_high

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: np.random.Generator) -> int:
        estimate = min(hand_size, sum(1 for c in hand if card_rank(c) >= self.FACE_THRESHOLD))
        if cant_say is not None and estimate == cant_say:
            if estimate > 0:
                estimate -= 1
            else:
                estimate += 1
        return estimate

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: np.random.Generator) -> int:
        del rng, hand
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

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: np.random.Generator) -> int:
        del hand, hand_size, rng
        if cant_say == 0:
            return 1
        return 0

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: np.random.Generator) -> int:
        del hand, rng
        lead = ctx.lead_suit
        if lead == -1:
            return self._lead_low(legal_cards)
        suited = [card for card in legal_cards if card_suit(card) == lead]
        if suited:
            non_face = [c for c in suited if card_rank(c) < 11]
            target_pool = non_face if non_face else suited
            target_pool.sort(key=card_rank)
            return target_pool[0]
        return self._lead_low(legal_cards)


class ShortSuitAggressor(_BaseHeuristic):
    """Bids based on number of short suits and plays to void suits quickly."""

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: np.random.Generator) -> int:
        del hand_size, rng
        suit_counts = [0, 0, 0, 0]
        for card in hand:
            suit_counts[card_suit(card)] += 1
        short_suits = sum(1 for count in suit_counts if count <= 2)
        estimate = max(1, short_suits)
        if cant_say is not None and estimate == cant_say:
            estimate = min(estimate + 1, len(hand))
        return estimate

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: np.random.Generator) -> int:
        del rng
        lead = ctx.lead_suit
        if lead == -1:
            suit_counts = {s: 0 for s in range(4)}
            for card in hand:
                suit_counts[card_suit(card)] += 1
            shortest_suit = min(suit_counts, key=lambda s: (suit_counts[s], s))
            options = [c for c in legal_cards if card_suit(c) == shortest_suit]
            return max(options, key=card_rank) if options else self._lead_high(legal_cards)
        suited = [card for card in legal_cards if card_suit(card) == lead]
        if suited:
            suited.sort(key=card_rank)
            return suited[len(suited) // 2]
        return self._lead_high(legal_cards)


class MiddleManager(_BaseHeuristic):
    """Balances estimation/play by focusing on mid-value cards."""

    MID_LOW = 7
    MID_HIGH = 10

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: np.random.Generator) -> int:
        del rng
        mids = sum(1 for c in hand if self.MID_LOW <= card_rank(c) <= self.MID_HIGH)
        suit_pairs = 0
        suit_has_low_high = {s: [False, False] for s in range(4)}
        for card in hand:
            rank = card_rank(card)
            suit = card_suit(card)
            if rank <= 6:
                suit_has_low_high[suit][0] = True
            if rank >= 12:
                suit_has_low_high[suit][1] = True
        suit_pairs = sum(1 for low, high in suit_has_low_high.values() if low and high)
        estimate = min(hand_size, mids + suit_pairs)
        if cant_say is not None and estimate == cant_say:
            estimate = max(0, estimate - 1)
        return estimate

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: np.random.Generator) -> int:
        del rng
        lead = ctx.lead_suit
        if lead == -1:
            best_suit = min(range(4), key=lambda s: self._suit_var(hand, s))
            suited = [c for c in legal_cards if card_suit(c) == best_suit]
            return suited[len(suited) // 2] if suited else self._lead_low(legal_cards)
        suited = [card for card in legal_cards if card_suit(card) == lead]
        if suited:
            sorted_suited = sorted(suited, key=card_rank)
            # try to play just above current best if still chasing tricks
            highest_in_trick = max(
                (card_rank(card) for card in ctx.trick_cards if card_suit(card) == lead and card != -1), default=-1
            )
            candidates = [card for card in sorted_suited if card_rank(card) > highest_in_trick]
            return candidates[0] if candidates else sorted_suited[0]
        longest_suit = max(range(4), key=lambda s: self._suit_length(hand, s))
        discard = [c for c in legal_cards if card_suit(c) == longest_suit]
        return discard[0] if discard else self._lead_low(legal_cards)

    @staticmethod
    def _suit_var(hand: List[int], suit: int) -> int:
        ranks = [card_rank(card) for card in hand if card_suit(card) == suit]
        if not ranks:
            return 0
        return max(ranks) - min(ranks)

    @staticmethod
    def _suit_length(hand: List[int], suit: int) -> int:
        return sum(1 for card in hand if card_suit(card) == suit)


class RandomLegalPolicy(BasePolicy):
    """Baseline opponent that randomly estimates/plays while respecting legality."""

    def estimate(self, hand: List[int], hand_size: int, cant_say: Optional[int], rng: np.random.Generator) -> int:
        del hand
        choices = list(range(hand_size + 1))
        if cant_say is not None and cant_say in choices:
            choices.remove(cant_say)
        if not choices:
            return 0
        return int(rng.choice(choices))

    def play(self, hand: List[int], legal_cards: List[int], ctx: TrickContext, rng: np.random.Generator) -> int:
        lead = ctx.lead_suit
        if lead != -1:
            suited = [card for card in legal_cards if card_suit(card) == lead]
        if suited:
            legal_cards = suited
        return int(rng.choice(legal_cards))


class SelfPlayOpponent(BasePolicy):
    """Adaptor that lets a learned `AgentPolicy` control non-agent seats."""

    def __init__(self, agent_policy: Callable, name: str = "SelfPlay"):
        self.agent_policy = agent_policy
        self.name = name
        self._env = None

    def attach_env(self, env):
        self._env = env

    def select_action(self, player_id: int) -> int:
        if self._env is None:
            raise RuntimeError("SelfPlayOpponent must be attached to an environment.")
        obs = self._env._make_observation(player_id)  # pylint: disable=protected-access
        mask = self._env._legal_actions_mask_for(player_id)  # pylint: disable=protected-access
        info = {"legal_actions": mask, "dealer": self._env.dealer_id, "phase": self._env.phase}
        return int(self.agent_policy(obs, info, self._env))
