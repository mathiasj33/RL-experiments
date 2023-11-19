import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme()


def main():
    env = 'InvertedPendulum-v4'
    algorithm_and_experiments = {
        'vpg': ['default', 'relu', 'wider']
    }
    df = load_data(env, algorithm_and_experiments)
    sns.relplot(df, x='Episode', y='SmoothedReturn', kind='line', errorbar='sd', col='Experiment', col_wrap=2)
    plt.show()


def load_data(env: str, algorithm_and_experiments: dict[str, list[str]], smooth_radius=10) -> pd.DataFrame:
    dfs = []
    for alg, experiments in algorithm_and_experiments.items():
        for experiment in experiments:
            path = f'logs/{alg}/{env}/{experiment}'
            if not os.path.exists(path):
                raise ValueError(f'Invalid path: {path}')
            for file in os.listdir(path):
                if not file.startswith('log'):
                    continue
                df = pd.read_csv(f'{path}/{file}')
                df['Algorithm'] = alg
                df['Experiment'] = experiment
                df['Seed'] = int(file.split('_')[-1].replace('.csv', ''))
                df['SmoothedReturn'] = smooth(df['Return'], smooth_radius)
                dfs.append(df)
    return pd.concat(dfs)


def smooth(row, radius):
    """
    Computes the moving average over the given row of data. Returns an array of the same shape as the original row.
    """
    y = np.ones(radius)
    z = np.ones(len(row))
    return np.convolve(row, y, 'same') / np.convolve(z, y, 'same')


if __name__ == '__main__':
    main()
