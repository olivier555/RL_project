import numpy as np
import matplotlib.pyplot as plt

from game import Game


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


def plot_value_iter(value_array):
    plt.plot(value_array, label='Value after t iterations')
    plt.legend()
    plt.show()


def value_eval(my_game: Game, nb_mc_iter, history):
    """
    Outcome sampling MCCFR as described in:
    Lanctot, Marc, et al. "Monte Carlo sampling for regret minimization in extensive games.",
    Advances in neural information processing systems. 2009.

    Using Uniform sampling

    Explorative (optimistical)

    :param my_game:
    :param nb_mc_iter:
    :return:
    """
    value_iter = []

    for t in range(nb_mc_iter):
        # Forward pass

        for n in my_game.info_sets:
            n.reset()
        history.reset()

        q_z = 1.0
        for n in my_game.info_sets:
            if n.is_reachable() and not n.is_terminal:
                if n.is_initial:
                    n.nb_visits = 1
                    n.p_sum = np.ones(len(n.p_sum))

                sigma = n.sigma if n.is_decision else None
                a, action = n.compute_chance(game=my_game, history=history.history, sigma=sigma)
                q_z *= 1.0 / len(n.actions)
                a.nb_visits += n.nb_visits
                a.p_sum += n.p_sum

                # if n.is_chance:
                #     history[n.topological_idx] = action  # n.actions[idx]
                history.update(n, action)

        # Backward pass
        for n in my_game.info_sets[::-1]:
            if n.is_reachable():
                if n.is_decision:
                    c = n.chance_child
                    sampled_action = n.idx_action_child
                    utility = c.value if c.player == n.player else -c.value

                    n.value = n.sigma[sampled_action] * utility

                    # cfp = n.p_sum[1 - n.player]
                    # regrets = -n.value*np.ones(len(n.actions))
                    # regrets[sampled_action] += utility
                    # regrets = (1.0/q_z)*cfp*regrets
                    # n.regrets += regrets

                    # n.sigma = strategy_update(n.regrets)
                    # n.sigma_sum += (t - n.last_visit) * n.sigma * n.p_sum[n.player]

                elif n.is_chance:
                    n.player = n.chance_child.player
                    n.value = n.chance_child.value
                else:
                    n.value = n.utility

                if n.is_initial:
                    value_iter.append(n.value)
                n.last_visit = t

    return np.mean(value_iter)


def mean_regret(my_game):
    expected_regrets_0 = []
    expected_regrets_1 = []
    for n in my_game.info_sets:
        if n.is_decision:
            if n.player == 0:
                expected_regrets_0.append((n.regrets * n.sigma).mean())
            else:
                expected_regrets_1.append((n.regrets * n.sigma).mean())
    mean_0 = np.mean(expected_regrets_0)
    mean_1 = np.mean(expected_regrets_1)
    return [mean_0, mean_1]


def plot_traj(history, x=None, **plot_params):
    # plot_params = {} if plot_params is None else plot_params
    history_np = np.array(history)
    theta_mean = np.mean(history_np, axis=0)
    theta_std = np.std(history_np, axis=0)
    n_iter = len(theta_mean)

    x = np.arange(n_iter) if x is None else x
    plt.plot(x, theta_mean, **plot_params)

    plt.fill_between(x=x,
                     y1=theta_mean - theta_std,
                     y2=theta_mean + theta_std, alpha=0.25)
