from dataclasses import dataclass
from typing import Optional

from torch import nn


@dataclass
class DDPGConfig:
    env_name: str
    max_episode_steps: Optional[int]
    gamma: float
    num_steps: int
    warmup_steps: int
    update_after: int
    update_every: int
    evaluate_every: int
    num_test_episodes: int
    noise_variance: float
    batch_size: int
    polyak: float
    actor_lr: float
    critic_lr: float
    actor_layers: list[int]
    critic_layers: list[int]
    actor_activation: any
    critic_activation: any


inverted_pendulum = DDPGConfig(
    env_name='InvertedPendulum-v4',
    max_episode_steps=None,
    gamma=0.99,
    num_steps=20_000,
    warmup_steps=2000,
    update_after=1000,
    update_every=50,
    evaluate_every=2000,
    num_test_episodes=10,
    noise_variance=0.1,
    batch_size=128,
    polyak=0.995,
    actor_lr=1e-3,
    critic_lr=1e-3,
    actor_layers=[256, 256],
    critic_layers=[256, 256],
    actor_activation=nn.ReLU,
    critic_activation=nn.ReLU
)

dmc_cartpole = DDPGConfig(
    env_name='dm_control/cartpole-balance_sparse-v0',
    max_episode_steps=None,
    gamma=0.99,
    num_steps=150_000,
    warmup_steps=4500,
    update_after=1000,
    update_every=50,
    evaluate_every=1500,
    num_test_episodes=10,
    noise_variance=0.3,
    batch_size=128,
    polyak=0.995,
    actor_lr=1e-3,
    critic_lr=1e-3,
    actor_layers=[256, 256],
    critic_layers=[256, 256],
    actor_activation=nn.ReLU,
    critic_activation=nn.ReLU
)

half_cheetah_small = DDPGConfig(
    env_name='HalfCheetah-v4',
    max_episode_steps=150,
    gamma=0.99,
    num_steps=20_000,
    warmup_steps=1000,
    update_after=1000,
    update_every=50,
    evaluate_every=1000,
    num_test_episodes=10,
    noise_variance=0.1,
    batch_size=100,
    polyak=0.995,
    actor_lr=1e-3,
    critic_lr=1e-3,
    actor_layers=[256, 256],
    critic_layers=[256, 256],
    actor_activation=nn.ReLU,
    critic_activation=nn.ReLU
)

half_cheetah = DDPGConfig(
    env_name='HalfCheetah-v4',
    max_episode_steps=None,
    gamma=0.99,
    num_steps=1_000_000,
    warmup_steps=10_000,
    update_after=1000,
    update_every=50,
    evaluate_every=10_000,
    num_test_episodes=10,
    noise_variance=0.1,
    batch_size=100,
    polyak=0.995,
    actor_lr=1e-3,
    critic_lr=1e-3,
    actor_layers=[256, 256],
    critic_layers=[256, 256],
    actor_activation=nn.ReLU,
    critic_activation=nn.ReLU
)
