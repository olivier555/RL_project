import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from fsicrm import fsicrm
from oscrm import oscrm_simulteneous
from dtcrm import dtcrm, dtoscrm

from utils import plot_traj
from goof_game import GoofGame, GoofHistory

NB_MC_ITER = 4000
EVAL_EVERY = 200
N_ITER = 8000
NB_CARDS = 3
NB_THREAD = 8


def os_train():
    game = GoofGame(nb_cards=NB_CARDS)
    history = GoofHistory()
    metrics = oscrm_simulteneous(game, N_ITER, history, nb_mc_iter=NB_MC_ITER,
                                 eval_every=EVAL_EVERY)
    return metrics


def dt_train(thresh=1.5):
    game = GoofGame(nb_cards=NB_CARDS)
    history = GoofHistory()
    metrics = dtoscrm(my_game=game, nb_iter=N_ITER, threshold_constant=thresh, history=history,
                      eval_every=EVAL_EVERY, nb_mc_iter=NB_MC_ITER)
    return metrics


if __name__ == '__main__':
    print('Training OS')
    os_trains = Parallel(n_jobs=NB_THREAD)(delayed(os_train)() for i in range(8))
    print('Training DT')
    dt_trains = Parallel(n_jobs=NB_THREAD)(delayed(dt_train)() for i in range(8))

    # Plots values
    os_times = np.array([metrics['time'] for metrics in os_trains]).mean(axis=0)
    dt_times = np.array([metrics['time'] for metrics in dt_trains]).mean(axis=0)

    os_values = [metrics['values'] for metrics in os_trains]
    dt_values = [metrics['values'] for metrics in dt_trains]

    # Averaging over all players
    os_regrets = np.array([metrics['regrets'] for metrics in os_trains])
    dt_regrets = np.array([metrics['regrets'] for metrics in dt_trains])

    # Plot value estimation
    plot_traj(os_values, os_times, label='OSCRM')
    plot_traj(dt_values, dt_times, label='DTOSCRM')
    plt.title('Goofspiel with {} cards'.format(NB_CARDS))
    plt.legend()
    plt.ylabel('Estimated Value')
    plt.xlabel('Time')
    # plt.xlim((0, 0.36))
    plt.savefig('goof_values.png')
    # plt.show()

    # Plot Regret estimation
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

    plt.sca(axes[1])
    plot_traj(dt_regrets[:,:,0], dt_times, label='player 0')
    plot_traj(dt_regrets[:,:,1], dt_times, label='player 1')
    plt.title('DTCRM')
    plt.legend()
    plt.ylabel('Mean Node Regret')
    plt.xlabel('Time')
    # plt.xscale('log')
    # plt.xlim((0, .55))

    plt.sca(axes[2])
    plot_traj(os_regrets[:, 2:, 0], os_times[2:], label='player 0')
    plot_traj(os_regrets[:, 2:, 1], os_times[2:], label='player 1')
    plt.title('OSCRM')
    plt.legend()
    plt.ylabel('Mean Node Regret')
    plt.xlabel('Time')
    # plt.xscale('log')
    # plt.xlim((0, .55))
    plt.savefig('goof_regrets.png')
    # plt.show()


    ###############
    # dt_trains_5 = Parallel(n_jobs=8)(delayed(dt_train)(thresh=5.) for i in range(16))
    # dt_trains_10 = Parallel(n_jobs=8)(delayed(dt_train)(thresh=10.) for i in range(16))
    #
    # dt_times_5 = np.array([metrics['time'] for metrics in dt_trains_5]).mean(axis=0)
    # dt_times_10 = np.array([metrics['time'] for metrics in dt_trains_10]).mean(axis=0)
    #
    # dt_values_5 = [metrics['values'] for metrics in dt_trains_5]
    # dt_values_10 = [metrics['values'] for metrics in dt_trains_10]
    #
    #
    # plot_traj(dt_values_10, dt_times_10, label='thresh=10')
    # plot_traj(dt_values_5, dt_times_5, label='thresh=5')
    # plot_traj(dt_values, dt_times, label='thresh=1')
    # plt.title('Kuhn Poker')
    # plt.legend()
    # plt.ylabel('Estimated Value')
    # plt.xlabel('Time')
    # # plt.xlim((0, 2))
    # plt.show()