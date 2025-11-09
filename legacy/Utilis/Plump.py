from Deck import Deck
from Card import Card, Suit, Rank
from Player import Player

from gymnasium import Env

class PlumpEnv(Env):

    def __init__(self, num_players):
        
        self.players = [Player(player_id=i) for i in range(num_players)]
        self.players[0].is_dealer = True
        