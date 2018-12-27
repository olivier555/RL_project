# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 09:42:06 2018

@author: olivi
"""

import numpy as np
from copy import deepcopy

from game import Game, KuhnGame


def strategy_update(regrets):
    """
    :param regrets:
    :return:
    """
    new_strat = np.maximum(regrets, 0.0)
    if np.max(new_strat) >  1e-8:
        new_strat = new_strat / np.sum(new_strat)
        return new_strat
    else:
        n_actions = len(new_strat)
        return 1.0/n_actions*np.ones(n_actions)


def fsicrm(my_game: Game, nb_iter):
    # TODO: COMPRENDRE POURQUOI EN INDICES 11 ET 12 ON CALCULE TOUJOURS LES MEMES VALEURS
    # TODO: Pour cela, regarder fonctionnement et mise a jour de cette fonction.
    for _ in range(nb_iter):
        history = {}
        for n in my_game.info_sets:
            if n.is_reachable():
                if n.is_initial:
                    n.nb_visits = 1
                    n.p_sum = np.ones(len(n.p_sum))
                if n.is_decision:
                    n.sigma = strategy_update(n.regrets)
                    n.sigma_sum += n.sigma * n.p_sum[n.player]

                    for idx_a, a in enumerate(n.actions):
                        child = my_game.get_child(starting_node=n, action=a, history=history)
                        p_sum_update = n.p_sum
                        p_sum_update[n.player] = n.sigma[idx_a] * p_sum_update[n.player]
                        child.p_sum += p_sum_update
                        child.nb_visits += n.nb_visits

                elif n.is_chance:
                    a, action = n.compute_chance(game=my_game, history=history)
                    a.nb_visits += n.nb_visits
                    a.p_sum += n.p_sum

                    history[n.topological_idx] = action  # n.actions[idx]

        for n in my_game.info_sets[::-1]:
            if n.is_reachable():
                if n.is_decision:
                    n.value = 0
                    for index_a, a in enumerate(n.actions):
                        c = my_game.get_child(starting_node=n, action=a, history=history)
                        n.value_action[index_a] = c.value if c.player == n.player else -c.value
                    n.value += (n.sigma * n.value_action).sum()
                    cfp = n.p_sum[1 - n.player]
                    n.regrets = (1 / (n.nb_visits_total + n.nb_visits)) * \
                                (n.nb_visits_total * n.regrets + n.nb_visits * cfp * (n.value_action - n.value))
                    n.nb_visits_total += n.nb_visits
                elif n.is_chance:
                    n.player = n.chance_child.player
                    n.value = n.chance_child.value
                else:
                    n.value = n.utility
        for n in my_game.info_sets:
            n.reset()


if __name__ == '__main__':
    kuhn_game = KuhnGame()
    fsicrm(kuhn_game, 10000)
