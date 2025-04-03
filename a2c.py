from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
    
class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Actor():
    def __init__(self, state_size, action_size, hidden_size=128):
        self.actor_network = ActorNetwork(state_size, action_size, hidden_size)
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=0.001)
    
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_prob = self.actor_network(state)
        action = torch.multinomial(action_prob, num_samples=1)
        return action.item()
    
    def learn(self, state, action, advantage):
        state = torch.tensor(state, dtype=torch.float32)
        action_prob = self.actor_network(state)
        # action_prob = action_prob[range(len(action_prob)), action]
        loss_pro = -torch.log(action_prob[action]) * advantage
        self.optimizer_actor.zero_grad()
        loss_pro.backward()
        self.optimizer_actor.step()

class Critic():
    def __init__(self, state_size, hidden_size=128, gamma=0.9):
        self.critic_network = CriticNetwork(state_size, hidden_size)
        self.optimizer = optim.Adam(self.critic_network.parameters(), lr=0.001)
        self.lossfunc=nn.MSELoss()#均方误差（MSE）
        self.gamma = gamma
    
    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.critic_network(state)
    
    def learn(self, state, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        with torch.no_grad():
            target = reward + self.gamma * self.critic_network(next_state)
        current = self.critic_network(state)
        loss = self.lossfunc(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        advantage = target - current
        return advantage.detach()  # 返回一个没有梯度的张量

class A2C:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.actor = Actor(state_size, action_size, hidden_size)
        self.critic = Critic(state_size, hidden_size)

        
    def train(self, env, num_episodes=1, max_steps=1000):
        loss_list = []
        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            for _ in range(max_steps):
                action = self.predict(state)
                next_state, reward = env.step(state, action)
                advantage = self.critic.learn(state, reward, next_state)
                self.actor.learn(state, action, advantage)
                # print(state, action, next_state, reward, advantage)
                state = next_state
            loss_list.append(advantage)
        return loss_list
    
    def predict(self, state):
        return self.actor.choose_action(state)

# 目前效果很差，需要改进
env = Environment(5, 5)
a2c = A2C(state_size=2, action_size=5)
loss_list = a2c.train(env, 10, 1000)
plt.plot(loss_list)
plt.show()
        
        
        
        
        
