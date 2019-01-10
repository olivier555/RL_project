from game import Game, Node


class KuhnGame(Game):
    def __init__(self):
        info_sets = self.compute_info_sets()
        self.terminal_nodes = info_sets[-4:]
        super(KuhnGame, self).__init__(info_sets)

    @staticmethod
    def compute_info_sets():
        """
        Proceeding depth wise
        WE CONSIDER AT FIRST ONLY NON TERMINAL NODES

        AND CONSIDER ONLY THE FOUR POSSIBLE TERMINAL NODES AND SUPPOSE THEY APPEAR AT DEPTH (AKA LINE)
        5
        :return:
        """
        idx = 0
        info_sets = [
            Node(actions=[0, 1, 2], available_information={}, is_chance=True, is_initial=True,
                 line=0, topological_idx=idx)]
        idx += 1
        info_sets += [Node(actions=a, available_information={}, is_chance=True, line=1,
                           topological_idx=new_id)
                      for (a, new_id) in zip(
                [[1, 2], [0, 2], [0, 1]],
                range(idx, idx + 3)
            )]
        idx = 4

        # First Player First Call
        line2 = []
        for j in range(3):
            line2.append(
                Node(actions=[0, 1], available_information={'active_player': 0, 'list_act': [],
                                                            'card': j},
                     is_decision=True, player=0, line=2, topological_idx=idx))
            idx += 1

        # Second Player information set
        line3 = []
        for card_2 in range(3):
            for last_call in range(2):
                line3.append(
                    Node(actions=[0, 1],
                         available_information={'active_player': 1, 'list_act': [last_call],
                                                'card': card_2},
                         is_decision=True, player=1, line=3, topological_idx=idx))
                idx += 1

        # First player last action (if any)
        line4 = []
        for card_1 in range(3):
            line4.append(
                Node(actions=[0, 1], available_information={'active_player': 0,
                                                            'list_act': [0, 1],
                                                            'card': card_1},
                     is_decision=True, player=0, line=4, topological_idx=idx))
            idx += 1

        # Terminal Nodes: 4 in total
        terminal_nodes = [
            Node(actions=[], available_information={'active_player': 0,
                                                    'list_act': None,
                                                    'card': None},
                 is_terminal=True, player=0, line=5, topological_idx=idx, utility=-1),
            Node(actions=[], available_information={'active_player': 0,
                                                    'list_act': None,
                                                    'card': None},
                 is_terminal=True, player=0, line=5, topological_idx=idx + 1, utility=1),
            Node(actions=[], available_information={'active_player': 0,
                                                    'list_act': None,
                                                    'card': None},
                 is_terminal=True, player=0, line=5, topological_idx=idx + 2, utility=-2),
            Node(actions=[], available_information={'active_player': 0,
                                                    'list_act': None,
                                                    'card': None},
                 is_terminal=True, player=0, line=5, topological_idx=idx + 3, utility=2)
        ]

        info_sets = info_sets + line2 + line3 + line4 + terminal_nodes
        return info_sets

    def get_child(self, starting_node, action, history):
        # First case: chance node
        if starting_node.line == 0:
            return self.info_sets[1 + action]
        elif starting_node.line == 1:
            card = history[0]  # card of first player
            for candidate in self.info_sets:
                if candidate.available_information == {'card': card, 'active_player': 0,
                                                       'list_act': []}:
                    return candidate

        # Second case: playing mode
        new_player = (starting_node.line + 1) % 2

        keys = sorted(history.keys(), reverse=False)
        result = []
        for key in keys:
            result.append(history[key])

        card0, card1 = result[:2]

        new_card = card0 if new_player == 0 else card1

        actions = starting_node.available_information['list_act'] + [action]
        # Go to terminal nodes
        if actions == [0, 0]:
            if card0 > card1:
                return self.terminal_nodes[1]
            else:
                return self.terminal_nodes[0]
        elif actions == [0, 1, 0]:
            return self.terminal_nodes[0]
        elif actions == [0, 1, 1] or actions == [1, 1]:
            if card0 > card1:
                return self.terminal_nodes[3]
            else:
                return self.terminal_nodes[2]
        elif actions == [1, 0]:
            return self.terminal_nodes[1]

        else:
            for candidate in self.info_sets:
                if candidate.available_information == {'card': new_card,
                                                       'active_player': new_player,
                                                       'list_act': actions}:
                    return candidate
            raise ValueError

    @staticmethod
    def are_terminal_actions(list_actions):
        return list_actions in [[0, 0], [0, 1, 0], [0, 1, 1], [1, 0], [1, 1]]