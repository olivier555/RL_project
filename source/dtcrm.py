# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 09:42:06 2018

@author: olivi
"""

import numpy as np

from game import Game
from kuhn_game import KuhnGame, KuhnHistory
from goof_game import GoofGame, GoofHistory



def strategy_update_threshold(regrets, t_iter, threshold_constant):
    new_strat = np.maximum(regrets, 0.0)
    n_actions = len(new_strat)
    if np.max(new_strat) > 1e-8:
        new_strat = new_strat / np.sum(new_strat)
    else:
        return (1.0 / n_actions) * np.ones(n_actions)
    threshold = (threshold_constant - 1) * np.sqrt(np.log(n_actions) / (2 * t_iter)) / (n_actions ** 2)
    threshold_indices = new_strat <= threshold
    if np.sum(threshold_indices) >= 1:
        print(threshold_indices)
    if threshold_indices.all():
        return (1.0 / n_actions) * np.ones(n_actions)
    new_strat[threshold_indices] = 0
    return new_strat / np.sum(new_strat)


def dtcrm(my_game: Game, nb_iter, threshold_constant, history):
    """
    Pruning-based method based on:
    Brown, Noam, Christian Kroer, and Tuomas Sandholm.,
    "Dynamic Thresholding and Pruning for Regret Minimization." AAAI. 2017.

    :param my_game:
    :param nb_iter:
    :param threshold_constant:
    :return:
    """
    mean_node_regrets = []

    for t in range(nb_iter):
        history.reset()
        for n in my_game.info_sets:
            if n.is_reachable():
                action = None
                if n.is_initial:
                    n.nb_visits = 1
                    n.p_sum = np.ones(len(n.p_sum))
                if n.is_decision:
                    n.sigma = strategy_update_threshold(n.regrets, t, threshold_constant)
                    n.sigma_sum += n.sigma * n.p_sum[n.player]

                    for idx_a, a in enumerate(n.actions):
                        
                        child = my_game.get_child(starting_node=n, action=a, history=history.history)
                        p_sum_update = n.p_sum
                        p_sum_update[n.player] = n.sigma[idx_a] * p_sum_update[n.player]
                        child.p_sum += p_sum_update
                        child.nb_visits += n.nb_visits

                elif n.is_chance:
                    a, action = n.compute_chance(game=my_game, history=history.history)
                    a.nb_visits += n.nb_visits
                    a.p_sum += n.p_sum

                    # history[n.topological_idx] = action  # n.actions[idx]
                history.update(n, action)

        for n in my_game.info_sets[::-1]:
            if n.is_reachable():
                if n.is_decision:
                    n.value = 0
                    for index_a, a in enumerate(n.actions):
                        c = my_game.get_child(starting_node=n, action=a, history=history.history)
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

        expected_regrets_0 = []
        expected_regrets_1 = []
        for n in my_game.info_sets:
            if n.is_decision:
                if n.player == 0:
                    expected_regrets_0.append((n.regrets*n.sigma).mean())
                else:
                    expected_regrets_1.append((n.regrets * n.sigma).mean())
        mean_0 = np.mean(expected_regrets_0)
        mean_1 = np.mean(expected_regrets_1)

        mean_node_regrets.append([mean_0, mean_1])

        for n in my_game.info_sets:
            n.reset()

    return np.array(mean_node_regrets)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    game = KuhnGame()
    history = KuhnHistory()
    mean_node_regrets = dtcrm(game,
                              nb_iter=1000,
                              threshold_constant=1,
                              history=history)
    mean_node_regrets = mean_node_regrets[10:]

    for node in game.info_sets:
        print(node.available_information, node.sigma_sum / node.sigma_sum.sum())

    plt.plot(mean_node_regrets[:, 0], label='Average node regrets Player 0')
    plt.plot(mean_node_regrets[:, 1], label='Average node regrets Player 1')
    plt.legend()
    plt.xscale('log')
    plt.show()
