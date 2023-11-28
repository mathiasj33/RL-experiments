import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme()


def main():
    env = 'HalfCheetah-v4'
    algorithm_and_experiments = {
        'ddpg': ['default']
    }
    df = load_data(env, algorithm_and_experiments, to_smooth=['TestEpisodeReturn'], log_dir='server_logs')
    sns.relplot(df, x='TotalSteps', y='TestEpisodeReturn_smooth', kind='line', errorbar='sd', col='Experiment')
    plt.show()


def load_data(env: str, algorithm_and_experiments: dict[str, list[str]], to_smooth: list[str], smooth_radius=10,
              log_dir='logs') -> pd.DataFrame:
    dfs = []
    for alg, experiments in algorithm_and_experiments.items():
        for experiment in experiments:
            path = f'{log_dir}/{alg}/{env}/{experiment}'
            if not os.path.exists(path):
                raise ValueError(f'Invalid path: {path}')
            for file in os.listdir(path):
                if not file.startswith('log'):
                    continue
                df = pd.read_csv(f'{path}/{file}')
                df['Algorithm'] = alg
                df['Experiment'] = experiment
                df['Seed'] = int(file.split('_')[-1].replace('.csv', ''))
                for col in to_smooth:
                    df[f'{col}_smooth'] = smooth(df[col], smooth_radius)
                dfs.append(df)
    result = pd.concat(dfs)
    if result.isna().any().any():
        print('Warning: data contains NaN values')
    return result


def smooth(row, radius):
    """
    Computes the moving average over the given row of data. Returns an array of the same shape as the original row.
    """
    y = np.ones(radius)
    z = np.ones(len(row))
    return np.convolve(row, y, 'same') / np.convolve(z, y, 'same')


if __name__ == '__main__':
    main()
