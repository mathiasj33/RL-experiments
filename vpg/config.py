from dataclasses import dataclass

import torch.nn as nn

@dataclass
class VPGConfig:
    env_name: str
    discrete: bool
    gamma: float
    num_episodes: int
    layer_sizes: list[int]
    activation: any  # torch module
    actor_lr: float


inverted_pendulum_config = VPGConfig(
    env_name='InvertedPendulum-v4',
    discrete=False,
    gamma=0.99,
    num_episodes=1000,
    layer_sizes=[32, 32],
    activation=nn.Tanh,
    actor_lr=1e-3
)

cartpole_config = VPGConfig(
    env_name='CartPole-v1',
    discrete=True,
    gamma=0.99,
    num_episodes=500,
    layer_sizes=[16, 32],
    activation=nn.Tanh,
    actor_lr=1e-3
)

acrobot_config = VPGConfig(
    env_name='Acrobot-v1',
    discrete=True,
    gamma=0.99,
    num_episodes=500,
    layer_sizes=[32, 32],
    activation=nn.Tanh,
    actor_lr=1e-3
)
