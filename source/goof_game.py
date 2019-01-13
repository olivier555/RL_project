from itertools import permutations
import numpy as np

from info_set import hash_dict
from game import Game, Node, History
from contrib import goof


class GoofGame(Game):
    def __init__(self, nb_cards):
        super(GoofGame, self).__init__(info_sets=None)
        self.nb_cards = nb_cards
        self.hash_to_node = None
        self.nodes = None
        self.reward_to_node = None

        self.initialize_nodes()

    def initialize_nodes(self):
        nodes_li = []
        cards = [i for i in range(1, self.nb_cards+1)]

        # Step 1: creating all nodes and information sets
        for nb_rounds in range(self.nb_cards):
            for all_cards in permutations(cards):
                for previous_p0 in permutations(cards, nb_rounds):
                    for previous_p1 in permutations(cards, nb_rounds):
                        all_cards = list(all_cards)
                        previous_p0 = list(previous_p0)
                        previous_p1 = list(previous_p1)

                        p1 = goof.Player('P1', nb_cards=self.nb_cards, previous=previous_p0)
                        p2 = goof.Player('P2', nb_cards=self.nb_cards, previous=previous_p1)
                        prizes = goof.Prize(nb_cards=self.nb_cards, all_cards=all_cards,
                                            current_round=nb_rounds)

                        engine = goof.Engine(p1=p1, p2=p2, prize=prizes)

                        info_chance = engine.get_infoset(is_chance=True)
                        info_p0 = engine.get_infoset(active_player=0)
                        info_p1 = engine.get_infoset(active_player=1)

                        if nb_rounds == 0:
                            dec_chance = Node(actions=cards,
                                              available_information=info_chance, is_chance=True,
                                              is_initial=True,
                                              line=nb_rounds*3)
                        else:
                            dec_chance = Node(actions=[card for card in cards
                                                       if card not in prizes.showing],
                                              available_information=info_chance, is_chance=True,
                                              line=nb_rounds*3)
                        dec_p0 = Node(actions=[card for card in cards if card not in previous_p0],
                                      available_information=info_p0, is_decision=True,
                                      line=nb_rounds*3+1)
                        dec_p1 = Node(actions=[card for card in cards if card not in previous_p1],
                                      available_information=info_p1, is_decision=True,
                                      line=nb_rounds*3+2)

                        nodes_li.append(dec_chance)
                        nodes_li.append(dec_p0)
                        nodes_li.append(dec_p1)

        # Step 2: removing duplicates (appearing for chance nodes)
        nodes_li = list(set(nodes_li))

        # Step 3: sort by line
        nodes_li = sorted(nodes_li, key=lambda node: node.line)

        # Step 4: create terminal nodes
        max_reward = int(self.nb_cards * (self.nb_cards + 1) / 2)
        reward_to_node = {}
        for reward in range(- max_reward, max_reward + 1):
            term_node = Node(is_terminal=True, utility=reward, player=0)
            nodes_li.append(term_node)
            reward_to_node[reward] = term_node

        # Step 5: initialize topological index
        for index_node, node in enumerate(nodes_li):
            node.topological_idx = index_node

        # Step 6: creating hash --> node dictionnary
        self.hash_to_node = {}
        for node in nodes_li:
            node_hash = hash_dict(node.available_information)
            self.hash_to_node[node_hash] = node
        self.info_sets = nodes_li
        self.reward_to_node = reward_to_node
        self.nodes = nodes_li

    def get_child(self, starting_node: Node, action, history):
        """

        :param starting_node:
        :param action:
        :param history: last action of player 0 (int)
        :return:
        """
        # First case: starting_node has a terminal child
        if starting_node.line == 3*(self.nb_cards - 1) + 2:
            actions0 = starting_node.available_information["actions_P0"]
            if history is not None:
                actions0.append(history)
            reward = self.get_reward(actions0=actions0,
                                     actions1=starting_node.available_information["actions_P1"] + [action],
                                     prizes=starting_node.available_information["prizes"])
            return self.reward_to_node[reward]

        # Second case: child node is not terminal
        # We will construct the information set of the child node to find it
        infoset = {}
        if starting_node.player == 0:
            infoset['active_player'] = 1
            infoset['current_round'] = starting_node.available_information["current_round"]
            infoset['prizes'] = starting_node.available_information["prizes"]
            infoset['actions_P0'] = starting_node.available_information["actions_P0"]  # P1 does not know what action P0 did
            infoset['actions_P1'] = starting_node.available_information["actions_P1"]

        elif starting_node.player == 1:
            actions0 = starting_node.available_information["actions_P0"]
            if history is not None:
                actions0.append(history)

            infoset['is_chance'] = True
            infoset['current_round'] = starting_node.available_information["current_round"] + 1
            infoset['prizes'] = starting_node.available_information["prizes"]
            infoset['actions_P0'] = actions0
            infoset['actions_P1'] = starting_node.available_information["actions_P1"] + [action]

        elif starting_node.is_chance:
            infoset['active_player'] = 0
            infoset['current_round'] = starting_node.available_information["current_round"]
            infoset['prizes'] = starting_node.available_information["prizes"] + [action]
            infoset['actions_P0'] = starting_node.available_information["actions_P0"]
            infoset['actions_P1'] = starting_node.available_information["actions_P1"]

        # Converting infoset to hash
        try:
            child = self.hash_to_node[hash_dict(infoset)]
        except KeyError:
            raise KeyError('following infoset unknown : {}'.format(infoset))
        return child

    def get_reward(self, actions0, actions1, prizes):
        """
        Computes reward for player 0
        gets points_0 - points_1

        Convention:
        Definition: winning player gets
        :param actions0:
        :param actions1:
        :param prizes:
        :return:
        """
        actions0_arr = np.array(actions0)
        actions1_arr = np.array(actions1)
        prizes_arr = np.array(prizes)

        points_0 = (actions0_arr > actions1_arr)*prizes_arr
        points_0 = points_0.sum()

        points_1 = (actions1_arr > actions0_arr)*prizes_arr
        points_1 = points_1.sum()

        return points_0 - points_1


class GoofHistory(History):
    def __init__(self):
        super(GoofHistory, self).__init__()
        self.history = None

    def update(self, node: Node, action):
        if node.player == 0:
            self.history = action

    def reset(self):
        self.history = None


if __name__ == '__main__':

    game = GoofGame(3)
    print(game.info_sets[0].available_information)
    print(game.info_sets[100].available_information)
    print(game.info_sets[200].available_information)