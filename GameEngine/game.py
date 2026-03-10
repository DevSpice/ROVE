import random
from collections import namedtuple

suits = ["s", "h", "c", "d"]
values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

Action = namedtuple('Action', ['check', 'fold', 'call', 'bet'])

class Card:
    suit = suits[0]
    value = values[0]

    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
    
    def __str__(self):
        return f"{self.value}{self.suit}"

class Deck:

    def __init__(self):
        self.cards = []
        self.size = 0
        for s in suits:
            for v in values:
                newCard = Card(s, v)
                self.cards.append(newCard)
                self.size += 1

    # Returns the number of cards currently in the deck
    def GetSize(self):
        return self.size

    # Pop method wrapper, removes the next card from the top of the deck and returns it
    def GetCard(self):
        if self.GetSize() > 0:
            self.size -= 1
            return self.cards.pop()
        else:
            print("No more cards!")

    def RShuffle(self):
        random.shuffle(self.cards)


class Hand:

    def __init__(self):
        self.cards = []
        self.table = False

    def AddCard(self, card: Card):
        self.cards.append(card)

    def IsTable(self):
        self.table = True

    # This is just for testing display, not for internal use
    def ShowHand(self):
        string = ""
        for card in self.cards:
            string += " " + str(card)
        print("Hand" + string)
    
    def ClearHand(self):
        self.cards = []

