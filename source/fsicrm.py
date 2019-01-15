# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 09:42:06 2018

@author: olivi
"""

import numpy as np
from tqdm import tqdm
import time

from utils import strategy_update, plot_traj, value_eval, mean_regret
from game import Game
from kuhn_game import KuhnGame, KuhnHistory

from goof_game import GoofGame, GoofHistory


# def fsicrm_eval(my_game: Game, mc_nb_iter, history):
#     """
#     Fixed-Strategy Iteration Counterfactual Regret Minimization as described in:
#     Neller, Todd W., and Marc Lanctot.,
#     "An Introduction to Counterfactual Regret Minimization." (2013).
#
#     :param my_game:
#     :param mc_nb_iter:
#     :return:
#     """
#     value_iter = []
#
#     for _ in range(mc_nb_iter):
#         history.reset()
#         for n in my_game.info_sets:
#             if n.is_reachable():
#                 action = None
#                 if n.is_initial:
#                     n.nb_visits = 1
#                     n.p_sum = np.ones(len(n.p_sum))
#                 if n.is_decision:
#                     # n.sigma = strategy_update(n.regrets)
#                     # n.sigma_sum += n.sigma * n.p_sum[n.player]
#
#                     for idx_a, a in enumerate(n.actions):
#                         child = my_game.get_child(starting_node=n,
#                                                   action=a,
#                                                   history=history.history)
#                         p_sum_update = n.p_sum
#                         p_sum_update[n.player] = n.sigma[idx_a] * p_sum_update[n.player]
#                         child.p_sum += p_sum_update
#                         child.nb_visits += n.nb_visits
#
#                 elif n.is_chance:
#                     a, action = n.compute_chance(game=my_game, history=history.history)
#                     a.nb_visits += n.nb_visits
#                     a.p_sum += n.p_sum
#
#                     # history[n.topological_idx] = action  # n.actions[idx]
#                 history.update(n, action)
#
#         for n in my_game.info_sets[::-1]:
#             if n.is_reachable():
#                 if n.is_decision:
#                     n.value = 0
#                     for index_a, a in enumerate(n.actions):
#                         c = my_game.get_child(starting_node=n, action=a, history=history.history)
#                         n.value_action[index_a] = c.value if c.player == n.player else -c.value
#                     n.value += (n.sigma * n.value_action).sum()
#                     cfp = n.p_sum[1 - n.player]
#                     # n.regrets = (1 / (n.nb_visits_total + n.nb_visits)) * \
#                     #             (n.nb_visits_total * n.regrets + n.nb_visits * cfp * (
#                     #                         n.value_action - n.value))
#                     n.nb_visits_total += n.nb_visits
#                 elif n.is_chance:
#                     n.player = n.chance_child.player
#                     n.value = n.chance_child.value
#                 else:
#                     n.value = n.utility
#
#                 if n.is_initial:
#                     value_iter.append(n.value)
#
#         for n in my_game.info_sets:
#             n.reset()
#     return np.mean(value_iter)


def fsicrm(my_game: Game, nb_iter, history, nb_mc_iter=10000, eval_every=10):
    """
    Fixed-Strategy Iteration Counterfactual Regret Minimization as described in:
    Neller, Todd W., and Marc Lanctot.,
    "An Introduction to Counterfactual Regret Minimization." (2013).

    :param my_game:
    :param nb_iter:
    :return:
    """
    mean_node_regrets = []
    value_iter = []
    times = []
    measure_time = 0.0
    start = time.time()

    for _iter in (range(nb_iter)):
        history.reset()
        for n in my_game.info_sets:
            if n.is_reachable():
                action = None
                if n.is_initial:
                    n.nb_visits = 1
                    n.p_sum = np.ones(len(n.p_sum))
                if n.is_decision:
                    n.sigma = strategy_update(n.regrets)
                    n.sigma_sum += n.sigma * n.p_sum[n.player]

                    for idx_a, a in enumerate(n.actions):
                        child = my_game.get_child(starting_node=n,
                                                  action=a,
                                                  history=history.history)
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
                
                # if n.is_initial:
                #     value_iter.append(n.value)
        # value_iter.append(fsicrm_eval(my_game, mc_nb_iter=1000, history=history))

        this_loop_time = time.time() - start - measure_time
        if _iter % eval_every == 0:
            start_measure = time.time()
            times.append(this_loop_time)
            value_iter.append(value_eval(my_game, nb_mc_iter=nb_mc_iter, history=history))
            mean_node_regrets.append(mean_regret(my_game))
            measure_time += time.time() - start_measure

        for n in my_game.info_sets:
            n.reset()
    return {
        'time': times,
        'values': value_iter,
        'regrets': mean_node_regrets
    }


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from math import sqrt
    from joblib import Parallel, delayed

    def train():
        game = KuhnGame()
        history = KuhnHistory()
        metrics = fsicrm(game, 500, history, nb_mc_iter=4000)
        return metrics['values']

    all_values = Parallel(n_jobs=8)(delayed(train)() for i in range(10))

    # all_values = []
    # for _ in range(10):
    #     game = KuhnGame()
    #     history = KuhnHistory()
    #     metrics = fsicrm(game, 500, history, nb_mc_iter=4000)
    #     all_values.append(metrics['values'])

    mean_value = np.mean(all_values, axis=0)
    plt.plot(mean_value)
    plt.show()
    
#    mean_node_regrets = fsicrm(game, 10000, history)
#    mean_node_regrets = mean_node_regrets[10:]

#    for node in game.info_sets:
#        print(node.available_information, node.sigma_sum / node.sigma_sum.sum())
#
#    plt.plot(mean_node_regrets[:, 0], label='Average node regrets Player 0')
#    plt.plot(mean_node_regrets[:, 1], label='Average node regrets Player 1')
#    plt.legend()
#    plt.xscale('log')
#    plt.show()

    plot_traj(all_values)
    plt.show()
