import numpy as np
import random
from collections import defaultdict

class CardDeck():
    def __init__(self):
        # Initialize a standard deck of 52 cards
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        self.rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14}
        self.cards = [{'rank': rank, 'suit': suit} for suit in suits for rank in ranks]
        self.shuffle_deck()

    def shuffle_deck(self):
        random.shuffle(self.cards)

    def deal(self, num_cards):
        if num_cards > len(self.cards):
            raise ValueError("Not enough cards in the deck")
        return [self.cards.pop() for _ in range(num_cards)]
    
class PlumpPlayer:
    def __init__(self, player_id, initial_hand=None, stick_strategy = 'Random'):
        self.player_id = player_id
        self.hand = initial_hand if initial_hand else []
        self.points = 0
        self.is_dealer = False
        self.starts_round = False
        self.stick_strategy = stick_strategy
        self.estimated_sticks = -1
        self.actual_sticks = 0

    def estimate_sticks(self, cant_say = None):
        if cant_say == None:
            if self.stick_strategy == 'Random':
                estimate = random.randint(0, len(self.hand))
            elif self.stick_strategy == 'Royal Count':
                estimate = sum([1 if card['rank'] in ['King', 'Queen', 'Jack', 'Ace'] else 0 for card in self.hand])
        else:
            if self.stick_strategy == 'Random':
                estimate = random.randint(0, len(self.hand))
                while estimate == cant_say:
                    estimate = random.randint(0, len(self.hand))
            elif self.stick_strategy == 'Royal Count':
                estimate = sum([1 if card['rank'] in ['King', 'Queen', 'Jack', 'Ace'] else 0 for card in self.hand])

                if estimate == cant_say:
                    estimate +=  random.choice([-1, 1]) if cant_say > 0 else 1

        self.estimated_sticks = estimate
        return estimate
        
    def weird_strategy(self):
        rank_count = defaultdict(lambda: defaultdict(int))
        for card in self.hand:
            rank_count[card['suit']][card['rank']] += 1
        highest_rank = max(rank_count, key=lambda suit: max(rank_count[suit], key=rank_count[suit].get))
        top_suits = [suit for suit in rank_count if max(rank_count[suit].values()) == max(rank_count[highest_rank].values())]
        # If there's a tie for the highest rank, choose based on the number of cards
        if len(top_suits) > 1:
            top_suits = max(top_suits, key=lambda suit: sum(rank_count[suit].values()))
        if isinstance(top_suits, list):
            top_suits = top_suits[0]
        return top_suits
    
    def choose_card(self, suit):
        suit_cards = [card for card in self.hand if card['suit'] == suit]
        if suit_cards:
        # If there are cards of the specified suit, choose one of them
            chosen_card = suit_cards[0]
        else:
        # If the specified suit doesn't exist, take a random 
            chosen_card = random.choice(self.hand)
        return chosen_card
    
    def play_card(self, card):
    # Remove the played card from the player's hand
        if card in self.hand:
            self.hand.remove(card)
        else:
            raise ValueError("Card not in player's hand")
    
    def play_round(self, suit = None):
        if self.starts_round:
            suit = self.weird_strategy()
            card = self.choose_card(suit=suit)
        else:
            card = self.choose_card(suit=suit)
        
        self.play_card(card)
        return card, suit
    
    def turn_to_deal(self, deal_bool):
        self.is_dealer = deal_bool