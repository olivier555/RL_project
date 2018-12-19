# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 09:42:06 2018

@author: olivi
"""

import numpy as np
from copy import deepcopy

class Node:
    
    def __init__(self, children=[], is_chance=False, 
                 is_decision=False, is_terminal=False, is_initial=False,
                 player=None, utility=None, line=None):
        self.children = children
        self.is_chance = is_chance
        self.is_decision = is_decision
        self.is_terminal = is_terminal
        self.is_initial = is_initial
        self.player = player
        self.nb_visits = 0
        self.nb_players = 2
        self.utility = utility
        self.line = line
        
        self.p_sum = [0] * self.nb_players
        self.sigma_sum = [0] * len(self.children)
        self.sigma = [0] * len(self.children)
        self.regrets = [0] * len(self.children)
        self.value_action = [0] * len(self.children)
        self.value = 0
        self.nb_visits_total = 0
        self.chance_child = None

    def compute_chance(self):
        assert(self.is_chance)
        self.chance_child = np.random.choice(self.children)
        return self.chance_child

    def is_reachable(self):
        return self.nb_visits != 0

    def reset(self):
        self.nb_visits = 0
        self.p_sum.fill(0)
        self.value_action.fill(0)
        self.value = 0


def strategy_update(regrets):
    new_strat = np.maximum(regrets, 0.0)
    if np.max(new_strat) >  1e-8:
        new_strat = new_strat / np.sum(new_strat)
        return new_strat
    else:
        n_actions = len(new_strat)
        return 1.0/n_actions*np.ones(n_actions)


def fsicrm(list_nodes, nb_iter):
    for _ in range(nb_iter):
        for n in list_nodes:
            if n.is_reachable():
                if n.is_initial:
                    n.nb_visits = 1
                    n.p_sum = [1] * len(n.p_sum)
                if n.is_decision:
                    n.sigma = strategy_update(n.regrets)
                    n_sigma_sum += n.sigma * n.p_sum[n.player]
                    for index_c, c in enumerate(n.children):
                        c.nb_visits += n.nb_visits
                        p_sum_update = n.p_sum
                        p_sum_update[n.player] = n.sigma[index_c] * p_sum_update[n.player]
                        c.p_sum += p_sum_update
                elif n.is_chance:
                    c = n.compute_chance()
                    c.nb_visits += n.nb_visits
                    c.p_sum += n.p_sum
        for n in list_nodes[::-1]:
            if n.is_reachable():
                if n.is_decision:
                    n.value = 0
                    for index_c, c in enumerate(n.children):
                        n.value_action[index_c] = c.value if c.player == n.player else - c.value
                    n.value += (n.sigma * n.value_action).sum()
                    cfp = n.p_sum[1 - n.player]
                    n.regrets[index_c] = (1 / (n.nb_visits_total + n.nb_visits)) * (n.nb_visits_total * n.regrets + n.nb_visits * cfp * (n.value_action - n.value))
                    n.nb_visits_total += n.nb_visits
                elif n.is_chance:
                    n.player = n.chance_child.player
                    n.value = n.chance_child.value
                else:
                    n.value = n.utility
        for n in list_nodes:
            n.reset()


def kuhn_poker(node, line, history):
    print(line, history)
    if line == 0:
        node = Node(is_initial=True, is_chance=True, line=line)
        children = [kuhn_poker(node, line=1, history=[i]) for i in range(3)]
        node.children = children
        return node
    elif line == 1:
        node = Node(is_chance=True, line=line)
        l = list(range(3))
        l.remove(history[0])
        children = [kuhn_poker(node, line=2, history=history + [i]) for i in l]
        node.children = children
        return node
    else:
        if len(history) <= 3:
            is_terminal = False
        else:
            is_terminal = not (history[-2] == 0 and history[-1] == 1)
        node = Node(is_decision=True, line=line, is_terminal=is_terminal, player=line%2)
        if not is_terminal:
            children = [kuhn_poker(node, line=line+1, history=history + [i]) for i in range(2)]
            node.children = children
        else:
            assert(len(history) >= 4, history)
            if history[-1] == 0 and history[-2] == 0:
                utility = 1 if history[1] > history[0] else -1
            elif history[-1] == 0 and history[-2] == 1:
                utility = -1
            elif history[-1] == 1 and history[-2] == 1 and line == 3:
                utility = 2 if history[1] > history[0] else -2
            else:
                utility = 2 if history[1] < history[0] else -2
            node.utility = utility
        return node


def create_kuhn_list(node):
    if node.is_terminal:
        return [node]
    else:
        l = [node]
        for c in node.children:
            l += create_kuhn_list(c)
        return l


if __name__ == '__main__':

    initial_node = kuhn_poker(None, 0, None)
    kuhn_list = create_kuhn_list(initial_node)
    kuhn_list = sorted(kuhn_list, key=lambda x: x.line)
