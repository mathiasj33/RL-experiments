import torch
import gymnasium as gym
from gaussian_policy_network import GaussianPolicyNetwork

env = gym.make('InvertedPendulum-v4', render_mode='human')
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]
obs, info = env.reset(seed=1)

policy_net = GaussianPolicyNetwork(obs_space_dims, action_space_dims, 16, 32)
policy_net.load_state_dict(torch.load('models/model.pth'))
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
