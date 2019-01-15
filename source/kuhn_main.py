import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from fsicrm import fsicrm
from oscrm import oscrm_simulteneous
from dtcrm import dtcrm

from utils import plot_traj
from kuhn_game import KuhnGame, KuhnHistory


def fs_train():
    game = KuhnGame()
    history = KuhnHistory()
    metrics = fsicrm(game, 500, history, nb_mc_iter=4000)
    return metrics


def os_train():
    game = KuhnGame()
    history = KuhnHistory()
    metrics = oscrm_simulteneous(game, 500, history, nb_mc_iter=4000)
    return metrics


def dt_train():
    game = KuhnGame()
    history = KuhnHistory()
    metrics = dtcrm(game, 500, threshold_constant=1.0, history=history, nb_mc_iter=4000)
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
    fs_regrets = np.array([metrics['regrets'] for metrics in fs_trains]).mean(axis=-1)
    os_regrets = np.array([metrics['regrets'] for metrics in os_trains]).mean(axis=-1)
    dt_regrets = np.array([metrics['regrets'] for metrics in dt_trains]).mean(axis=-1)

    # Plot value estimation
    plot_traj(fs_values, fs_times, label='FSICRM')
    plot_traj(os_values, os_times, label='OSCRM')
    plot_traj(dt_values, dt_times, label='DTCRM')
    plt.title('Kuhn Poker')
    plt.legend()
    plt.ylabel('Estimated Value')
    plt.xlabel('Time')
    plt.xlim((0, .2))
    plt.show()

    # Plot Regret estimation
    plot_traj(fs_regrets, fs_times, label='FSICRM')
    plot_traj(os_regrets, os_times, label='OSCRM')
    plot_traj(dt_regrets, dt_times, label='DTCRM')
    plt.title('Kuhn Poker')
    plt.legend()
    plt.ylabel('Mean Node Regret')
    plt.xlabel('Time')
    plt.xlim((0, .2))
    plt.show()


    fs_arr = np.array(fs_regrets)
