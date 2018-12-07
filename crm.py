import numpy as np
from copy import deepcopy

from info_set import hash_dict


class History:
    def __init__(self):
        """
        Implements history object for Kuhn Poker.
        - 2 first nodes of traj are the respective cards of players 1 and 2
        - other nodes, in {0, 1} correspond to respectively pass and bet

        :param terminal_nodes:
        """

        self.terminal_nodes = [
            [0, 0], [0, 1, 0], [0, 1, 1], [1, 0], [1, 1]
        ]
        self.active_player = 0
        self.traj =[]

    def is_terminal(self):
        """
        Needs to be adapted to considered problem
        :return: Bool
        """
        if len(self.traj) <= 3:
            # This means that no more than 1 player has played, game will continue
            return False
        else:
            play_seqs = self.get_play_sequences()
            return play_seqs in self.terminal_nodes

    def utility(self, player):
        """
        Returns utility of selected player assuming we are in a terminal node.
        :param player: player for which we want utility
        :return:
        """
        assert self.is_terminal()
        play_seqs = np.array(self.get_play_sequences())
        player_wins = 2.0*(self.traj[player] > self.traj[1-player]) - 1.0  # valued in -1, 1
        if np.array_equal(play_seqs, [0, 0]):
            util = 1.0*player_wins
        elif np.array_equal(play_seqs, [0, 1, 0]):
            util = 2.0*(player == 1) - 1
        elif np.array_equal(play_seqs, [0, 1, 1]):
            util = 2.0*player_wins
        elif np.array_equal(play_seqs, [1, 0]):
            util = 2.0*(player == 0) - 1
        elif np.array_equal(play_seqs, [1, 1]):
            util = 2.0*player_wins
        else:
            raise ValueError('Trajectory does not appear to be terminated: {}'.format(play_seqs))

        return util

    def update(self, action):
        self.traj += [action]
        self.active_player = 1 - self.active_player

    def get_information_set(self):
        card_info = self.traj[self.active_player]
        information_set = {
            'card_info': card_info,
            'active_player': self.active_player,
            'trajectory': self.traj[2:]
        }
        return information_set

    def get_play_sequences(self):
        return self.traj[2:]

    def is_chance(self):
        return len(self.traj) < 2

    def chance_sample(self):
        if len(self.traj) == 0:
            self.traj += [np.random.randint(low=0, high=3)]
            self.active_player = 1 - self.active_player
        elif len(self.traj) == 1:
            previous_val = self.traj[0]
            candidates = [i for i in range(3) if i != previous_val]
            self.traj += [np.random.choice(candidates)]
            self.active_player = 1 - self.active_player
        else:
            raise PermissionError


# def utility(history, player, player1card, player2card):
#
#     #TODO: implement for Kuhn Poker
#     return


# We consider below more information sets that appear in reality
all_info_sets = [{'card_info': card,
                  'active_player': player,
                  'trajectory': traj}
                 for card in range(3) for player in range(2) for traj in [[], [0], [1], [0, 1]]]
all_hashs = [hash_dict(dico) for dico in all_info_sets]
hash_to_index = {hashc: index for (index, hashc) in enumerate(all_hashs)}

n_info_sets = len(all_info_sets)

regrets_info_sets = np.zeros((n_info_sets, 2))
strategies_info_sets = np.zeros((n_info_sets, 2))
strategy = 0.5 * np.ones((n_info_sets, 2))


def strategy_update(regrets):
    new_strat = np.maximum(regrets, 0.0)
    if np.max(new_strat) >  1e-8:
        new_strat = new_strat / np.sum(new_strat)
        return new_strat
    else:
        n_actions = len(new_strat)
        return 1.0/n_actions*np.ones(n_actions)


def crm(history: History, player, pis):
    pi1, pi2 = pis
    if history.is_terminal():
        return history.utility(player)
    if history.is_chance():
        history.chance_sample()
        return crm(history, player, [pi1, pi2])

    information_set = history.get_information_set()
    info_set_index = hash_to_index[hash_dict(information_set)]
    value = 0
    value_to_action = [0.0, 0.0]

    for a in range(2):
        new_hist = deepcopy(history)
        new_hist.update(a)
        if history.active_player == 0:
            value_to_action[a] = crm(new_hist, player, [strategy[info_set_index, a]*pi1, pi2])
        else:
            value_to_action[a] = crm(new_hist, player, [pi1, strategy[info_set_index, a]*pi2])
        value += strategy[info_set_index, a]*value_to_action[a]

    if history.active_player == player:
        for a in range(2):
            regrets_info_sets[info_set_index, a] += pis[player-1]*(value_to_action[a] - value)
            strategies_info_sets[info_set_index, a] += pis[player]*strategy[info_set_index, a]
        strategy[info_set_index, :] = strategy_update(regrets_info_sets[info_set_index, :])
    return value

def Solve(nb_iter):
    for _ in range(nb_iter):
        for p in range(2):
            h = History()
            crm(h, p, [1, 1])

