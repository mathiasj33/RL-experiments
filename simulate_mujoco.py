import torch
import gymnasium as gym

from vpg.gaussian_actor import GaussianActor as VPGGaussianActor
import torch.nn as nn

env_name = 'InvertedPendulum-v4'
env = gym.make(env_name, render_mode='human')
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]
obs, info = env.reset(seed=1)

policy_net = VPGGaussianActor(obs_space_dims, action_space_dims, [16, 32], nn.Tanh)
policy_net.load_state_dict(torch.load(f'models/vpg/{env_name}.pth'))
policy_net.eval()

with torch.no_grad():
    while True:
        dist = policy_net(torch.tensor(obs, dtype=torch.float32))
        action = dist.sample()
        # action = dist.mean
        # action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            print('Reset')
