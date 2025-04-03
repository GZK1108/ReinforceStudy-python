from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

"""
主要学习PPO思路与实现，目前效果差不收敛。
"""

class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_std = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x))
        std = self.softplus(self.fc_std(x)) + 1e-5
        return mean, std

    def select_action(self, state):
        with torch.no_grad():
            mean, std = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob


class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc_value(x)
        return value


class ReplayMemory:

    def __init__(self, batch_size):
        self.states = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int32)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size, device='cpu'):
        self.LR_ACTOR = 0.0001
        self.LR_CRITIC = 0.001
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.EPOCHS = 10
        self.EPSILON = 0.2
        self.BATCH_SIZE = batch_size
        self.device = device

        self.actor = ActorNet(state_dim, action_dim).to(device)
        self.old_actor = ActorNet(state_dim, action_dim).to(device)
        self.critic = CriticNet(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        self.memory = ReplayMemory(batch_size)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, log_prob = self.actor.select_action(state)
        value = self.critic(state)
        return action.detach().cpu().numpy()[0], np.float32(value.detach().cpu().numpy()[0])

    def learn(self):
        self.old_actor.load_state_dict(self.actor.state_dict())
        for epoch_i in range(self.EPOCHS):
            states, actions, vals, rewards, dones, batches = self.memory.generate_batches()
            T = len(rewards)
            advantages = np.zeros(T, dtype=np.float32)

            for t in range(T):
                discount = 1
                a_t = 0
                for k in range(t, T - 1):
                    a_t += discount * (rewards[k] + self.GAMMA * vals[k + 1] * (1 - int(dones[k])) - vals[k])
                    discount *= self.GAMMA * self.LAMBDA
                advantages[t] = a_t + vals[t]  # 广义优势函数

            with torch.no_grad():
                advantages = torch.tensor(advantages).unsqueeze(1).to(self.device)
                values = torch.tensor(vals).to(self.device)

            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions).to(self.device)

            for batch in batches:
                with torch.no_grad():
                    old_mu, old_sigma = self.old_actor(states[batch])
                    old_pi = Normal(old_mu, old_sigma)
                batch_old_probs = old_pi.log_prob(actions[batch])

                mu, sigma = self.actor(states[batch])
                pi = Normal(mu, sigma)
                batch_probs = pi.log_prob(actions[batch])

                ratio = torch.exp(batch_probs - batch_old_probs)
                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(ratio, 1 - self.EPSILON, 1 + self.EPSILON) * advantages[batch]
                actor_loss = -torch.min(surr1, surr2).mean()

                batch_returns = advantages[batch] + values[batch]
                batch_value = self.critic(states[batch])

                critic_loss = nn.MSELoss()(batch_returns, batch_value)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            self.memory.clear_memory()

    def save_policy(self):
        torch.save(self.actor.state_dict(), 'ppo_actor.pth')
        torch.save(self.critic.state_dict(), 'ppo_critic.pth')


env = Environment(5, 5)
NUM_EPISODE = 200
NUM_STEP = 200
BATCH_SIZE = 25
UPDATE_INTERVAL = 50
agent = PPOAgent(2, len(ACTIONS), BATCH_SIZE)

best_reward = -2000
REWARD_BUFFER = np.empty(NUM_EPISODE)
for episode_i in range(NUM_EPISODE):
    state = env.reset()
    done = False
    episode_reward = 0
    for step_i in range(NUM_STEP):
        # 选择动作
        action, value = agent.get_action(state)
        action_fixed = np.argmax(action)
        # 执行动作
        next_state, reward = env.step(state, action_fixed)
        episode_reward += reward
        done = True if (step_i == NUM_STEP - 1) else False

        # 存储经验
        agent.memory.store_memory(state, action, value, reward, done)

        # 更新状态
        state = next_state

        if (step_i + 1) % UPDATE_INTERVAL == 0 or (step_i + 1) == NUM_STEP:
            # 更新模型
            agent.learn()

    if episode_reward > -100 and episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy()
    print(f"Episode {episode_i + 1} / {NUM_EPISODE}, Reward: {episode_reward}, Best Reward: {best_reward}")

    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode {episode_i + 1} / {NUM_EPISODE}, Reward: {episode_reward}, Best Reward: {best_reward}")


