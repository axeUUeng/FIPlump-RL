import gym
from gym import spaces
import numpy as np
import random
from collections import defaultdict

class CardDeck:
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

class PlumpEnv:
    def __init__(self, num_players, stick_strategies):
        self.num_players = num_players
        self.players = [PlumpPlayer(player_id=i, stick_strategy=stick_strategy) for i, stick_strategy in zip(range(num_players), stick_strategies)]
        self.players[0].turn_to_deal(deal_bool=True)
        self.card_deck = CardDeck()
        self.current_player = 0  # Player 0 starts the game
        self.current_stick = []
        self.current_round = 0  # Keep track of the number of rounds
        self.stick_number = 0  # Keep track of the number of sticks in a round
        self.protocol = {player.player_id: [] for player in self.players}  # Initialize protocol with zeros

    def deal_cards(self, num_cards):
        for player in self.players:
            player.hand = self.card_deck.deal(num_cards=num_cards)
    
    def get_estimates(self):
        self.dealer_id = None
        for player in self.players:
            if player.is_dealer:
                self.dealer_id = player.player_id
                break
        self.estimates = [-1] * self.num_players
        estimates_relative_dealer = []
        self.start_id = None

        for i in range(self.dealer_id + 1, self.dealer_id + self.num_players + 1):
            player = self.players[i % self.num_players]  # Use modulo to loop around the list
            if player.player_id == self.dealer_id: # If the player is dealer, constraint on estimation
                cant_say = len(player.hand) - sum(self.estimates) - 1
                est = player.estimate_sticks(cant_say = cant_say)
                if all(est > e for e in self.estimates):
                    self.start_id = player.player_id
                self.estimates[player.player_id] = est
                estimates_relative_dealer.append(est)
                
            else:
                est = player.estimate_sticks()
                if all(est > e for e in self.estimates):
                    self.start_id = player.player_id
                self.estimates[player.player_id] = est
                estimates_relative_dealer.append(est)              

        assert sum(self.estimates) != len(self.players[0].hand), f"Estimations going even, something wrong! Cant_say = {cant_say}, The estimates {self.estimates}, Number of cards being played {len(self.players[0].hand)}"
        self.players[self.start_id].starts_round = True
    
    def play_one_round(self):
        round_hand = {}
        chosen_suit = None
        for i in range(self.start_id, self.start_id + self.num_players):
            player = self.players[i % self.num_players]
            card, chosen_suit = player.play_round(suit=chosen_suit)
            round_hand[player.player_id] = card
    
        highest_card = None
        highest_rank = -1
        player_with_highest_card = None
        for player_id, card in round_hand.items():
            # Check if the card's suit matches the target suit
            if card['suit'] == chosen_suit:
                card_rank = self.card_deck.rank_values[card['rank']]
                if card_rank > highest_rank:
                    highest_rank = card_rank
                    highest_card = card
                    player_with_highest_card = player_id
        
        self.players[self.start_id].starts_round = False
        self.start_id = player_with_highest_card
        self.players[player_with_highest_card].starts_round = True
        self.players[player_with_highest_card].actual_sticks += 1
    
    def play_the_hands(self):
        N = len(self.players[0].hand)
        for i in range(N):
            self.play_one_round()
     
        for player in self.players:
            if player.estimated_sticks == player.actual_sticks:
                player.points += 10 + player.actual_sticks if player.estimate_sticks != 0 else 5
                self.protocol[player.player_id].append(10 + player.actual_sticks)
                player.actual_sticks = 0
                player.estimated_sticks = 0
            else:
                self.protocol[player.player_id].append(0)
                player.actual_sticks = 0
                player.estimated_sticks = 0

    def play_full_hand(self, num_cards):
        self.card_deck = CardDeck()
        self.deal_cards(num_cards = num_cards)
        self.get_estimates()
        self.play_the_hands()
        self.players[self.dealer_id].is_dealer = False
        self.players[(self.dealer_id + 1) % len(self.players)].is_dealer = True


    def reset(self):
        # Reset the game for a new round
        pass

    def start_estimation_phase(self):
        # Start the estimation phase before the gameplay phase
        pass

    def step(self, action):
        pass

    
