import torch
from gymnasium.core import ObsType, Env

from policy_network import PolicyNetwork

class VanillaPolicyGradient:
    def __init__(self, policy_net: PolicyNetwork):
        self.policy_net = policy_net
        self.optimiser = torch.optim.AdamW(self.policy_net.parameters())
        self.gamma = 0.99

    def train_episode(self, obs: ObsType, env: Env) -> float:
        done = False
        log_probs = []
        rewards = []
        while not done:
            dist = self.policy_net(torch.tensor(obs, dtype=torch.float32))
            action = dist.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(dist.log_prob(action))
            rewards.append(float(reward))
            done = terminated or truncated

        running_reward = 0
        reward_sums = []
        for r in rewards[::-1]:
            running_reward = r + self.gamma * running_reward
            reward_sums.insert(0, running_reward)
        loss = 0
        for log_prob, delta in zip(log_probs, reward_sums):
            loss -= log_prob.mean() * delta

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return sum(rewards)
