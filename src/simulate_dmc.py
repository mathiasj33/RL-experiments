import numpy as np
import torch
from dm_control import suite, viewer

from algorithms.ddpg.actor import Actor as DDPGActor
from algorithms.ddpg.config import dmc_cartpole

config = dmc_cartpole
algorithm = 'ddpg'
experiment = 'long'
seed = 0

env = suite.load(domain_name="cartpole", task_name="balance_sparse", visualize_reward=False,
                 task_kwargs={'time_limit': float('inf')})
obs_space_dims = 5
action_space_dims = 1

actor = DDPGActor(obs_space_dims, action_space_dims, config.actor_layers, config.actor_activation, limit=config.high_clip)
actor.load_state_dict(torch.load(f'models/{algorithm}/{config.env_name}/{experiment}/seed_{seed}/actor.pth',
                                 map_location='cpu'))
actor.eval()

def policy(time_step):
    obs = np.concatenate([np.array(v) for k, v in time_step.observation.items()])
    action = actor(torch.tensor(obs, dtype=torch.float32))
    return action.cpu().detach().numpy()

viewer.launch(env, policy=policy)
