import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 14


def main():
    algorithm = 'vpg'
    env_name = 'InvertedPendulum-v4'
    x_limit = 1000
    y_limit = 1000
    plot_experiments(algorithm, env_name, ['default'], x_limit, y_limit, smooth_radius=10)


def plot_experiments(algorithm, env_name, experiments, x_limit, y_limit, smooth_radius=10):
    x = range(1, x_limit + 1)
    for experiment in experiments:
        results = []
        path = f'results/{algorithm}/{env_name}/{experiment}'
        for file in os.listdir(path):
            results.append(np.load(f'{path}/{file}'))
        results = np.row_stack(results)
        results = smooth(results, smooth_radius)
        mean = np.mean(results, axis=0)
        stddev = np.std(results, axis=0)

        plt.plot(x, mean, label=experiment)
        plt.fill_between(x, mean - stddev, mean + stddev, alpha=0.2)

    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.ylim(top=y_limit)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('result.pdf', bbox_inches='tight')
    plt.show()


def smooth(data, radius):
    result = np.zeros_like(data)

    def smooth_row(row):
        y = np.ones(radius)
        z = np.ones(len(row))
        return np.convolve(row, y, 'same') / np.convolve(z, y, 'same')

    if len(data.shape) == 1:
        return smooth_row(data)
    for i in range(len(data)):
        result[i, :] = smooth_row(data[i, :])
    return result


if __name__ == '__main__':
    main()