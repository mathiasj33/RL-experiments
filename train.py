import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from gaussian_policy_network import GaussianPolicyNetwork
from categorical_policy_network import CategoricalPolicyNetwork
from vanilla_policy_gradient import VanillaPolicyGradient

def train():
    num_episodes = 2000

    for seed in [1]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # env = gym.make('InvertedPendulum-v4')
        env = gym.make('Acrobot-v1')
        obs_space_dims = env.observation_space.shape[0]
        # action_space_dims = env.action_space.shape[0]
        action_space_dims = 3
        policy_net = CategoricalPolicyNetwork(obs_space_dims, action_space_dims, 16, 32)
        pg = VanillaPolicyGradient(policy_net)

        rewards = []
        pbar = tqdm(range(num_episodes))
        for _ in pbar:
            obs, _ = env.reset(seed=seed)
            reward = pg.train_episode(obs, env)
            rewards.append(reward)
            pbar.set_postfix({'reward': reward})

        env.close()
        torch.save(pg.policy_net.state_dict(), 'models/model.pth')

        plt.plot(rewards)
        plt.show()


if __name__ == '__main__':
    train()
