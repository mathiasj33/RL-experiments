import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch

from policy.categorical_policy_network import CategoricalPolicyNetwork

with torch.no_grad():
    env = gym.make('Acrobot-v1', render_mode='rgb_array')
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = 3
    obs, _ = env.reset(seed=1)
    policy_net = CategoricalPolicyNetwork(obs_space_dims, action_space_dims, 16, 32)
    policy_net.load_state_dict(torch.load('models/model.pth'))
    policy_net.eval()

    fig = plt.figure()


    def update(i):
        global obs

        dist = policy_net(torch.tensor(obs, dtype=torch.float32))
        action = dist.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            print('Reset')

        frame = env.render()

        fig.clear()
        plt.imshow(frame)
        plt.draw()


    anim = animation.FuncAnimation(fig, update, frames=1, interval=10)
    plt.show()

env.close()
