import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from gymnasium.spaces import Box

from vpg.categorical_actor import CategoricalActor as VPGCategoricalActor
import torch.nn as nn

seed = 0
# torch.manual_seed(seed)

env_name = 'Acrobot-v1'
env = gym.make(env_name, render_mode='rgb_array')
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0] if isinstance(env.action_space, Box) else env.action_space.n
obs, _ = env.reset(seed=seed)

policy_net = VPGCategoricalActor(obs_space_dims, action_space_dims, [32, 32], nn.Tanh)
policy_net.load_state_dict(torch.load(f'models/vpg/{env_name}.pth'))
policy_net.eval()

with torch.no_grad():
    fig = plt.figure()

    def update(i):
        global obs

        dist = policy_net(torch.tensor(obs, dtype=torch.float32))
        action = dist.sample()
        # action = torch.argmax(dist.probs)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            print('Reset')

        frame = env.render()

        fig.clear()
        plt.imshow(frame)
        plt.draw()


    anim = animation.FuncAnimation(fig, update, frames=1, interval=0)
    plt.show()

env.close()
