from random import randint
from Card import Suit

clubs = 0
diamonds = 1
spades = 2
hearts = 3
suits = ["c", "d", "s", "h"]

class Hand:
    def __init__(self):

        self.clubs = []
        self.diamonds = []
        self.spades = []
        self.hearts = []
        
        # create hand of cards split up by suit
        self.hand = [self.clubs, self.diamonds,
                    self.spades, self.hearts]


    def size(self):
        return len(self.clubs) + len(self.diamonds) + len(self.spades) + len(self.hearts)


    def updateHand(self):
        self.hand = [self.clubs, self.diamonds,
                    self.spades, self.hearts]

    def getRandomCard(self):
        suit = randint(0,3)
        suit = self.hand[suit]
        while len(suit) == 0:
            suit = randint(0,3)
            suit = self.hand[suit]
        index = randint(0, len(suit)-1)

        return suit[index]


    @classmethod
    def strToCard(self, card):
        if len(card) == 0: return None
        
        suit = card[len(card)-1].lower() # get the suit from the string
        
        try:
            suitIden = suits.index(suit)
        except Exception as e:
            print ('Invalid suit')
            print(e)
            return None

        cardRank = card[0:len(card)-1] # get rank from string
        
        try:
            cardRank = cardRank.upper()
        except AttributeError:
            pass

        # convert rank to int
        if cardRank == "T":
            cardRank = 10
        elif cardRank == "J":
            cardRank = 11
        elif cardRank == "Q":
            cardRank = 12
        elif cardRank == "K":
            cardRank = 13
        elif cardRank == "A":
            cardRank = 14
        else:
            try:
                cardRank = int(cardRank)
            except Exception as e:
                print ("Invalid card rank.")
                print(e)
                return None

        return cardRank, suitIden

    def have_suit(self, suit):
        return len(self.hand[suit]) > 0
    
    def containsCard(self, cardRank, suitIden):
        for card in self.hand[suitIden]:
            if card.rank.rank == cardRank:
                cardToPlay = card
                    
                # remove cardToPlay from hand
                # self.hand[suitIden].remove(card)

                # update hand representation
                # self.updateHand()
                return cardToPlay
        return None

    def playCard(self, card):
        cardInfo = self.strToCard(card)

        if cardInfo is None:
            return None
        else:
            cardRank, suitIden = cardInfo[0], cardInfo[1]

        # see if player has that card in hand
        return self.containsCard(cardRank, suitIden)

    def removeCard(self, card):
        suitId = card.suit.iden
        for c in self.hand[suitId]:
            if c == card:
                if suitId == clubs and card.rank.rank == 2:
                    self.contains2ofclubs = False
                # print "Removing:", c.__str__()
                self.hand[card.suit.iden].remove(c)
                self.updateHand()

    def hasOnlyHearts(self):
        return len(self.hearts) == self.size()



    def __str__(self):
        handStr = ''
        for suit in self.hand:
            for card in suit:
                handStr += card.__str__() + ' '
        return handStr