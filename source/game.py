import numpy as np

import info_set


class Node:
    def __init__(self, actions=[], available_information={}, is_chance=False,
                 is_decision=False, is_terminal=False, is_initial=False,
                 player=None, utility=None, line=None, topological_idx=0):
        """
        Corresponds to INFORMATION SET
        :param actions:
        :param is_chance:
        :param is_decision:
        :param is_terminal:
        :param is_initial:
        :param player:
        :param utility:
        :param line:
        """
        self.available_information = available_information

        self.actions = actions
        self.is_chance = is_chance
        self.is_decision = is_decision
        self.is_terminal = is_terminal
        self.is_initial = is_initial
        self.player = player
        self.nb_visits = 0
        self.nb_players = 2
        self.utility = utility
        self.line = line
        self.topological_idx = topological_idx

        self.p_sum = np.zeros(2)
        self.sigma_sum = np.zeros(len(self.actions))
        self.sigma = np.zeros(len(self.actions))
        self.regrets = np.zeros(len(self.actions))
        self.value_action = np.zeros(len(self.actions))
        self.value = 0
        self.nb_visits_total = 0
        self.chance_child = None
        self.idx_action_child = None

        # Useful only for Outcome Sampling
        self.last_visit = 0

    def compute_chance(self, game, history, sigma=None):
        """
        This method allows to sample a child from this node.
        Can also work for a decision node (see outcome-sampling MC CRF)
        :param game:
        :param history:
        :return:
        """
        # TODO: verify that in algos everything is coherent
        if sigma is None:
            self.idx_action_child = np.random.choice(np.arange(len(self.actions)))
        else:
            try:
                self.idx_action_child = np.random.choice(np.arange(len(self.actions)), p=sigma)
            except ValueError:
                self.idx_action_child = np.random.choice(np.arange(len(self.actions)))
                # print('Non admissible strategy for sampling')
        action = self.actions[self.idx_action_child]
        self.chance_child = game.get_child(self, action=action, history=history)
        return self.chance_child, action

    def is_reachable(self):
        return (self.nb_visits != 0) or self.is_initial  # initial node always reachable

    def reset(self):
        self.nb_visits = 0
        self.p_sum.fill(0)
        self.value_action.fill(0)
        self.value = 0
        if self.is_chance:
            self.player = None

    #OLIVIER: J'AI AJOUTE CES DEUX METHODES CAR SINON LA SUPPRESSION DES INFORMATION
    # SET DUPLIQUES NE FONCTIONNE PAS
    def __eq__(self, other):
        return info_set.hash_dict(self.available_information) == \
               info_set.hash_dict(other.available_information)

    def __hash__(self):
        return info_set.hash_dict(self.available_information)


class Game:
    def __init__(self, info_sets=None):
        """
        Skeletton class (specific to each game).
        We only need this class to provide child node given a starting node (corresponding
        to an information set), an action, and an history.
        :param info_sets: List of nodes
        """
        self.info_sets = info_sets

    def get_child(self, starting_node, action, history):
        return


class History:
    def __init__(self):
        self.history = None

    def update(self, node: Node, action):
        pass

    def reset(self):
        self.history = None


