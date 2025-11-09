from Hand import Hand

class Player:
    def __init__(self, player_id):
            self.name = player_id
            self.hand = Hand()
            self.score = 0
            self.is_dealer = False
            self.starts_round = False
            self.estimated_sticks = -1
            self.sticks_won = 0

    def estimateSticks(self, estimation, count, cant_say = None):
        pass

    def play(self, card):
        return self.hand.playCard(card)

    def trickWon(self, cards):
        self.CardsInRound += cards

    def hasSuit(self, suit):
        return len(self.hand.hand[suit.iden]) > 0

    def removeCard(self, card):
        self.hand.removeCard(card)

    def discardTricks(self):
        self.tricksWon = []

    def resetRoundCards(self):
        self.CardsInRound = []

    def hasOnlyHearts(self):
        return self.hand.hasOnlyHearts()