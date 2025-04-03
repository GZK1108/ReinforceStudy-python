from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.position] = (state, action, reward, next_state)
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def clear(self):
        self.buffer.clear()
        self.position = 0
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, lr, gamma, epsilon, 
                        epsilon_min, epsilon_decay, batch_size, capacity, device):
        self.policy_net = DQN(input_size, hidden_size, output_size).to(device)
        self.target_net = DQN(input_size, hidden_size, output_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.capacity = capacity
        self.replay_buffer = ReplayBuffer(capacity)
        self.device = device

    # 经验存储
    def remember(self, state, action, reward, next_state):
        self.replay_buffer.push(state, action, reward, next_state)

    # 选择动作
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS) - 1)
        else:
            with torch.no_grad():
                return self.policy_net(torch.tensor(state, dtype=torch.float32)).argmax().item()
    
    # 更新模型
    def update_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    
    # 单轮训练
    def learn(self, epoch):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        transitions = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*transitions)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)

        for i in range(epoch):
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0].detach()

            target_q_values = (next_q_values * self.gamma) + rewards

            criterion = nn.MSELoss()

            loss = criterion(current_q_values, target_q_values.unsqueeze(1))
            
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        
            # 更新epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
        return loss.item()
        
    # 训练
    def train(self, env, policy,num_episodes, max_steps, update_interval):
        loss_list = []
        self.policy_net.train()
        for episode in range(num_episodes):
            print(f"Episode {episode+1} / {num_episodes}")
            state = env.reset()
            for _ in range(self.capacity):
                action = policy.get_action(state)
                next_state, reward = env.step(state,action)
                self.remember(normalize(state, 0, env.row-1), action, reward, normalize(next_state, 0, env.row-1))
                state = next_state

            for step in tqdm(range(max_steps)):
                # 每轮调用learn
                loss = self.learn(200)

                # 更新模型
                if (step+1) % update_interval == 0:
                    self.update_model()
                loss_list.append(loss)
        return loss_list
    
    def predict(self, env, policy):
        self.policy_net.eval()
        state = env.reset()
        for i in range(env.row):
            for j in range(env.col):
                state = [i,j]
                with torch.no_grad():
                    q_values = self.target_net(torch.tensor(normalize(state, 0, env.row-1), dtype=torch.float32))
                    
                q_argmax = np.argmax(q_values)
                policy.p[i,j] = np.eye(len(ACTIONS))[q_argmax]


env = Environment(5, 5)
policy = Policy(env)
agent = DQNAgent(input_size=2, hidden_size=128, output_size=len(ACTIONS), lr=0.001, gamma=0.9, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.999, batch_size=100, capacity=1000, device='cpu')
loss_list = agent.train(env, policy, 1, 1000, 20)
plt.plot(loss_list)
plt.show()
agent.predict(env, policy)
policy.render()

    
        
