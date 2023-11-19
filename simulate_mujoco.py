import torch
import gymnasium as gym

from vpg.gaussian_actor import GaussianActor as VPGGaussianActor
from vpg.config import inverted_pendulum_config

config = inverted_pendulum_config

env = gym.make(config.env_name, render_mode='human')
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]
obs, info = env.reset(seed=42)

policy_net = VPGGaussianActor(obs_space_dims, action_space_dims, config.layer_sizes, config.activation)
policy_net.load_state_dict(torch.load(f'models/vpg/{config.env_name}/del/model_seed_0.pth'))
policy_net.eval()

with torch.no_grad():
    try:
        while True:
            dist = policy_net(torch.tensor(obs, dtype=torch.float32))
            action = dist.sample()
            # action = dist.mean
            # action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()
                print('Reset')
    except KeyboardInterrupt:
        print('Stopped.')
