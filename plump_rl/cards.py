"""Card helpers for the Plump RL environment."""

from __future__ import annotations

from typing import Iterable, List, Tuple

RANKS: List[int] = list(range(2, 15))  # 2-14 (Ace high)
RANK_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_LABELS = ["C", "D", "H", "S"]
CARDS_PER_SUIT = len(RANKS)
NUM_SUITS = len(SUIT_LABELS)
DECK_SIZE = CARDS_PER_SUIT * NUM_SUITS  # 52


def card_to_id(rank: int, suit: int) -> int:
    """Return a 0-51 identifier for the given rank/suit."""
    return suit * CARDS_PER_SUIT + (rank - 2)


def id_to_card(card_id: int) -> Tuple[int, int]:
    """Return (rank, suit) for a 0-51 identifier."""
    suit = card_id // CARDS_PER_SUIT
    rank = RANKS[card_id % CARDS_PER_SUIT]
    return rank, suit


def card_rank(card_id: int) -> int:
    return RANKS[card_id % CARDS_PER_SUIT]


def card_suit(card_id: int) -> int:
    return card_id // CARDS_PER_SUIT


def card_to_str(card_id: int) -> str:
    rank, suit = id_to_card(card_id)
    return f"{RANK_LABELS[rank - 2]}{SUIT_LABELS[suit]}"


def hand_to_binary(hand: Iterable[int]) -> List[int]:
    """Return a 52-length binary vector encoding the provided hand."""
    mask = [0] * DECK_SIZE
    for card in hand:
        mask[card] = 1
    return mask
