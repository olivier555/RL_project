import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from fsicrm import fsicrm
from oscrm import oscrm_simulteneous
from dtcrm import dtcrm

from utils import plot_traj
from kuhn_game import KuhnGame, KuhnHistory

NB_MC_ITER = 5000
EVAL_EVERY = 100
N_ITER = 2000


def fs_train():
    game = KuhnGame()
    history = KuhnHistory()
    metrics = fsicrm(game, N_ITER, history, nb_mc_iter=NB_MC_ITER,
                     eval_every=EVAL_EVERY)
    return metrics


def os_train():
    game = KuhnGame()
    history = KuhnHistory()
    metrics = oscrm_simulteneous(game, 4000, history, nb_mc_iter=NB_MC_ITER,
                                 eval_every=EVAL_EVERY)
    return metrics


def dt_train(thresh=1.0):
    game = KuhnGame()
    history = KuhnHistory()
    metrics = dtcrm(game, N_ITER, threshold_constant=thresh, history=history,
                    nb_mc_iter=NB_MC_ITER, eval_every=EVAL_EVERY)
    return metrics


if __name__ == '__main__':
    print('Training FS')
    fs_trains = Parallel(n_jobs=8)(delayed(fs_train)() for i in range(16))
    print('Training OS')
    os_trains = Parallel(n_jobs=8)(delayed(os_train)() for i in range(16))
    print('Training DT')
    dt_trains = Parallel(n_jobs=8)(delayed(dt_train)() for i in range(16))

    # Plots values
    fs_times = np.array([metrics['time'] for metrics in fs_trains]).mean(axis=0)
    os_times = np.array([metrics['time'] for metrics in os_trains]).mean(axis=0)
    dt_times = np.array([metrics['time'] for metrics in dt_trains]).mean(axis=0)

    fs_values = [metrics['values'] for metrics in fs_trains]
    os_values = [metrics['values'] for metrics in os_trains]
    dt_values = [metrics['values'] for metrics in dt_trains]

    # Averaging over all players
    fs_regrets = np.array([metrics['regrets'] for metrics in fs_trains])
    os_regrets = np.array([metrics['regrets'] for metrics in os_trains])
    dt_regrets = np.array([metrics['regrets'] for metrics in dt_trains])

    # Plot value estimation
    plot_traj(fs_values, fs_times, label='FSICRM')
    plot_traj(os_values, os_times, label='OSCRM')
    plot_traj(dt_values, dt_times, label='DTCRM')
    plt.title('Kuhn Poker')
    plt.legend()
    plt.ylabel('Estimated Value')
    plt.xlabel('Time')
    plt.xlim((0, 2))
    plt.show()

    # Plot Regret estimation
    fig, axes = plt.subplots(ncols=3, figsize=(10, 4))

    plt.sca(axes[0])
    plot_traj(fs_regrets[:, :, 0], fs_times, label='player 0')
    plot_traj(fs_regrets[:, :, 1], fs_times, label='player 1')
    plt.title('FSICRM')
    plt.legend()
    plt.ylabel('Mean Node Regret')
    plt.xlabel('Time')
    # plt.xscale('log')
    # plt.xlim((0, 2))

    plt.sca(axes[1])
    plot_traj(dt_regrets[:,:,0], dt_times, label='player 0')
    plot_traj(dt_regrets[:,:,1], dt_times, label='player 1')
    plt.title('DTCRM')
    plt.legend()
    plt.ylabel('Mean Node Regret')
    plt.xlabel('Time')
    # plt.xscale('log')
    # plt.xlim((0, 2))


    plt.sca(axes[2])
    plot_traj(os_regrets[:, 2:, 0], os_times[2:], label='player 0')
    plot_traj(os_regrets[:, 2:, 1], os_times[2:], label='player 1')
    plt.title('OSCRM')
    plt.legend()
    plt.ylabel('Mean Node Regret')
    plt.xlabel('Time')
    # plt.xscale('log')
    # plt.xlim((0, .2))
    plt.show()


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