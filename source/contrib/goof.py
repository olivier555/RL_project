from random import randint, choice


class Deck(object):
    """Deck is a class with one attribute: cards.  This is where one can choose the set
   of cards/value that will make up the Players' hands and the prize deck. The
   standard deck is a full suit: values 1 through 13 (i.e. range(1,14).  The
   Player class has an attribute called 'hand', which is a Deck object. Prize is
   the only sub-class of Deck.

    """

    def __init__(self, nb_cards):
        self.nb_cards = nb_cards
        self.cards = list(range(1, nb_cards+1))


class Player(object):
    """Player is a class with 4 attributes: name, hand, score, and current_bid. A
    Player can be the end-user or an AI, programmed in the Engine class. Player
    contains a `reset_current_bid` method and several methods for players to
    make bids.  Currently, a bid can be determined by user input using
    `bid_from_input` or by random selection, using `random_bid`. The range of
    bid methods may grow as the AI is developed.

    """

    def __init__(self, name, nb_cards, previous=None):
        self.name = name
        self.hand = Deck(nb_cards=nb_cards).cards
        self.previous = []
        self.score = 0
        self.current_bid = None

        if previous is not None:
            self.previous = previous

    def bid(self, card):
        """Method `bid` takes self and card parameters, removes the card from the
        player's hand, and returns the card (an integer). `bid` is not used
        outside the Player class. `bid` does not handle cases in which the card
        argument is not an integer or is not found in the player's hand. Those
        situations are handled by the functions that call bid, such as
        `bid_from_input`.

        """
        i = self.hand.index(card)
        self.hand.pop(i)
        self.previous.append(card)
        return card

    def reset_current_bid(self):
        self.current_bid = None


class Prize(Deck):
    def __init__(self, nb_cards, all_cards=None, current_round=None):
        super(Prize, self).__init__(nb_cards=nb_cards)
        self.showing = []

        if all_cards is not None and current_round is not None:
            self.cards = all_cards[current_round+1:]
            self.showing = all_cards[:current_round+1]

    def next_card(self):
        i = randint(0, len(self.cards) - 1)
        card = self.cards.pop(i)
        self.showing.append(card)
        return card

    def reset_count(self):
        self.showing = []


class Engine(object):
    def __init__(self, p1: Player, p2: Player, prize: Prize):
        self.prize = prize
        self.p1 = p1
        self.p2 = p2

    def identify_winner(self, player1, player2):
        if player1.current_bid > player2.current_bid:
            winner = player1
        else:
            winner = player2
        player1.reset_current_bid()
        player2.reset_current_bid()
        return winner

    def get_infoset(self, active_player=None, is_chance=False):
        infoset = {}
        current_round = len(self.prize.showing)
        if active_player == 0:
            infoset['active_player'] = 0
            infoset['current_round'] = current_round
            infoset['prizes'] = self.prize.showing
            infoset['actions_P0'] = self.p1.previous
            infoset['actions_P1'] = self.p2.previous

        elif active_player == 1:
            infoset['active_player'] = 1
            infoset['current_round'] = current_round
            infoset['prizes'] = self.prize.showing
            infoset['actions_P0'] = self.p1.previous
            infoset['actions_P1'] = self.p2.previous
        elif is_chance:
            infoset['is_chance'] = True
            infoset['current_round'] = current_round
            infoset['prizes'] = self.prize.showing[:-1]  # Current card sampled not yet in info set!
            infoset['actions_P0'] = self.p1.previous
            infoset['actions_P1'] = self.p2.previous
        else:
            raise ValueError('Invalid command')
        return infoset

    def get_child_infoset(self, active_player=None, is_chance=False):
        infoset = {}
        current_round = len(self.prize.showing)
        if active_player == 0:
            infoset['active_player'] = 1
            infoset['current_round'] = current_round
            infoset['prizes'] = self.prize.showing
            infoset['actions_P0'] = self.p1.previous[:-1]  # P1 does not know what action P0 did
            infoset['actions_P1'] = self.p2.previous

        elif active_player == 1:
            infoset['is_chance'] = True
            infoset['current_round'] = current_round + 1
            infoset['prizes'] = self.prize.showing
            infoset['actions_P0'] = self.p1.previous
            infoset['actions_P1'] = self.p2.previous

        elif is_chance:
            infoset['active_player'] = 0
            infoset['current_round'] = current_round
            infoset['prizes'] = self.prize.showing
            infoset['actions_P0'] = self.p1.previous
            infoset['actions_P1'] = self.p2.previous
        else:
            raise ValueError('Invalid command')
        return infoset

    def is_terminal(self):
        return len(self.prize.cards) == self.prize.nb_cards

    def award_prize(self, player1, player2, prize):
        winner = self.identify_winner(player1, player2)
        winner.score += prize
        print("%s won that round, for a prize of %d points." %
              (winner.name, prize))


if __name__ == "__main__":
    new_game = Engine()
    new_game.play()
