"""Observation encoding utilities for Plump agents.

The helper functions in this module translate raw environment dictionaries into
flat vectors that are easier for neural networks to consume. Encodings combine
discrete embeddings (card rank/suit), one-hot trick representations, and
contextual scalars such as bids or tricks won.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from .cards import DECK_SIZE


class SupportsObservationConfig(Protocol):
    """Subset of EnvConfig required for encoding helpers."""

    num_players: int
    hand_size: int


@lru_cache(maxsize=1)
def _card_embedding_matrix(rank_dims: int = 13, suit_dims: int = 4) -> np.ndarray:
    """Return a cached (52 x (rank_dims+suit_dims)) embedding matrix."""

    embeddings = np.zeros((DECK_SIZE, rank_dims + suit_dims), dtype=np.float32)
    for card_id in range(DECK_SIZE):
        rank_index = card_id % 13
        suit_index = card_id // 13
        embeddings[card_id, rank_index] = 1.0
        embeddings[card_id, rank_dims + suit_index] = 1.0
    return embeddings


def encode_hand(cards: Iterable[int]) -> np.ndarray:
    """Return the mean embedding of all cards in the agent's hand."""

    matrix = _card_embedding_matrix()
    cards = list(cards)
    if not cards:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return matrix[cards].mean(axis=0, dtype=np.float32)


def encode_current_trick(trick: Sequence[int], num_players: int) -> np.ndarray:
    """Return one-hot encodings for the cards currently on the table."""

    encoded = np.zeros((num_players, DECK_SIZE), dtype=np.float32)
    for player_id, card_id in enumerate(trick):
        if 0 <= card_id < DECK_SIZE:
            encoded[player_id, card_id] = 1.0
    return encoded.reshape(-1)


def encode_observation(obs: Mapping[str, np.ndarray], config: SupportsObservationConfig) -> np.ndarray:
    """Encode a raw `PlumpEnv` observation into a flat feature vector."""

    hand_cards = np.nonzero(obs["hand"])[0]
    hand_vec = encode_hand(hand_cards)
    trick_vec = encode_current_trick(obs["current_trick"], config.num_players)

    phase = np.zeros(2, dtype=np.float32)
    phase[int(obs["phase"])] = 1.0

    lead = np.zeros(5, dtype=np.float32)
    lead[int(obs["lead_suit"])] = 1.0

    estimations = obs["estimations"].astype(np.float32) / max(1, config.hand_size)
    tricks_won = obs["tricks_won"].astype(np.float32) / max(1, config.hand_size)
    cards_remaining = obs["cards_remaining"].astype(np.float32) / max(1, config.hand_size)
    tricks_played = np.array([obs["tricks_played"] / max(1, config.hand_size)], dtype=np.float32)

    played_cards = np.zeros(DECK_SIZE, dtype=np.float32)
    for card_id in np.atleast_1d(obs.get("cards_played", [])):
        if 0 <= card_id < DECK_SIZE:
            played_cards[card_id] = 1.0

    return np.concatenate(
        [
            phase,
            hand_vec,
            trick_vec,
            lead,
            estimations,
            tricks_won,
            cards_remaining,
            tricks_played,
            played_cards,
        ]
    )


def observation_dim(config: SupportsObservationConfig) -> int:
    """Return the length of the encoded observation vector for ``config``."""

    dummy_obs = {
        "phase": 0,
        "hand": np.zeros(52, dtype=np.int8),
        "current_trick": np.zeros(config.num_players, dtype=np.int16),
        "lead_suit": 4,
        "estimations": np.zeros(config.num_players, dtype=np.int8),
        "tricks_won": np.zeros(config.num_players, dtype=np.int8),
        "cards_remaining": np.zeros(config.num_players, dtype=np.int8),
        "tricks_played": np.int8(0),
        "cards_played": np.zeros(0, dtype=np.int16),
    }
    return encode_observation(dummy_obs, config).shape[0]
