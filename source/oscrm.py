import numpy as np

from utils import strategy_update
from game import Game
from kuhn_game import KuhnGame, KuhnHistory


def oscrm_simulteneous(my_game: Game, nb_iter, history):
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

    for t in range(nb_iter):
        # Forward pass
        history.reset()

        q_z = 1.0
        for n in my_game.info_sets:
            if n.is_reachable() and not n.is_terminal:
                action = None
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
                    utility = None
                    sampled_action = None
                    for index_a, a in enumerate(n.actions):
                        c = my_game.get_child(starting_node=n, action=a, history=history.history)
                        if c.is_reachable():
                            utility = c.value if c.player == n.player else -c.value
                            sampled_action = index_a
                            break

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

                n.last_visit = t

        for n in my_game.info_sets:
            n.reset()

    return np.array(mean_node_regrets)


if __name__ == '__main__':
    kuhn_game = KuhnGame()
    history = KuhnHistory()
    oscrm_simulteneous(kuhn_game, nb_iter=10000, history=history)

    for node in kuhn_game.info_sets:
        print(node.available_information, node.sigma_sum / node.sigma_sum.sum())
