from tqdm import tqdm
import numpy as np
import time

from utils import strategy_update, value_eval, mean_regret, plot_traj
from game import Game

from goof_game import GoofGame, GoofHistory
from kuhn_game import KuhnGame, KuhnHistory


def oscrm_simulteneous(my_game: Game, nb_iter, history, nb_mc_iter=4000,
                       eval_every=10, verbose=False):
    """
    Outcome sampling MCCFR as described in:
    Lanctot, Marc, et al. "Monte Carlo sampling for regret minimization in extensive games.",
    Advances in neural information processing systems. 2009.

    Using Uniform sampling

    Explorative (optimistical)

    :param my_game:
    :param nb_iter:
    :return:
    """
    mean_node_regrets = []
    value_iter = []
    times = []
    measure_time = 0.0
    start = time.time()

    iter_range = range(nb_iter) if verbose is False else tqdm(range(nb_iter))
    for t in iter_range:
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

                a, action = n.compute_chance(game=my_game, history=history.history)
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

                    # utility = None
                    # sampled_action = None
                    # for index_a, a in enumerate(n.actions):
                    #     c = my_game.get_child(starting_node=n, action=a, history=history.history)
                    #     if c.is_reachable():
                    #         utility = c.value if c.player == n.player else -c.value
                    #         sampled_action = index_a
                    #         break

                    c = n.chance_child
                    sampled_action = n.idx_action_child
                    utility = c.value if c.player == n.player else -c.value

                    n.value = n.sigma[sampled_action] * utility
                    cfp = n.p_sum[1 - n.player]

                    regrets = -n.value*np.ones(len(n.actions))
                    regrets[sampled_action] += utility
                    regrets = (1.0/q_z)*cfp*regrets
                    n.regrets += regrets

                    n.sigma = strategy_update(n.regrets)
                    n.sigma_sum += (t - n.last_visit) * n.sigma * n.p_sum[n.player]

                elif n.is_chance:
                    n.player = n.chance_child.player
                    n.value = n.chance_child.value
                else:
                    n.value = n.utility

                # if n.is_initial:
                #     value_iter.append(n.value)
                # n.last_visit = t

        this_loop_time = time.time() - start - measure_time
        if t % eval_every == 0:
            start_measure = time.time()
            times.append(this_loop_time)
            value_iter.append(value_eval(my_game, nb_mc_iter=nb_mc_iter, history=history))

            regrets_norm = np.array(mean_regret(my_game))
            regrets_norm = 1.0/(t+1) * regrets_norm
            mean_node_regrets.append(regrets_norm)
            measure_time += time.time() - start_measure

    return {
        'time': times,
        'values': value_iter,
        'regrets': mean_node_regrets
    }


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # game = KuhnGame()
    # history = KuhnHistory()

    game = GoofGame(nb_cards=4)
    history = GoofHistory()

    values = oscrm_simulteneous(game, nb_iter=1000, history=history, eval_every=100,
                                nb_mc_iter=1,
                                verbose=True)

    times = values['time']
    regrets = np.array(values['regrets'])
    0
    plt.plot(times, regrets[:, 0])
    plt.plot(times, regrets[:, 1])
    plt.show()

    for node in game.info_sets:
       print(node.available_information)
       print(node.sigma_sum / node.sigma_sum.sum())

