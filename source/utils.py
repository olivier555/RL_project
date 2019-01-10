import numpy as np


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