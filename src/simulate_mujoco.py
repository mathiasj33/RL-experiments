import torch
import gymnasium as gym

from algorithms.vpg.gaussian_actor import GaussianActor as VPGGaussianActor
from algorithms.vpg.config import inverted_pendulum_config as VPGInvertedPendulum
from algorithms.ddpg.actor import Actor as DDPGActor
from algorithms.ddpg.config import inverted_pendulum as DDPGInvertedPendulum, half_cheetah

config = half_cheetah
algorithm = 'ddpg'
experiment = 'default'
seed = 0
model_dir = 'server_models'

env = gym.make(config.env_name, render_mode='human')
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]
obs, info = env.reset(seed=42)

actor = DDPGActor(obs_space_dims, action_space_dims, config.actor_layers, config.actor_activation, limit=config.high_clip)
actor.load_state_dict(torch.load(f'{model_dir}/{algorithm}/{config.env_name}/{experiment}/seed_{seed}/actor.pth',
                                 map_location='cpu'))
actor.eval()

with torch.no_grad():
    try:
        ret = 0
        while True:
            dist = actor(torch.tensor(obs, dtype=torch.float32))
            # action = dist.sample()
            # action = dist.mean
            action = dist
            obs, reward, terminated, truncated, _ = env.step(action.cpu().detach().numpy())
            ret += float(reward)
            if terminated or truncated:
                observation, info = env.reset()
                print(f'Reset. Return achieved: {ret}')
                ret = 0
    except KeyboardInterrupt:
        print('Stopped.')
