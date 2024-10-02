import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
# 创建神经网络模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

state_size = 4  # CartPole-v1 的状态空间维度
action_size = 2  # 动作空间维度（左右推杆）
learning_rate = 0.001
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 256
memory_size = 2000
target_update = 10  # 每10步更新目标网络
episodes = 1000
max_steps = 500  # 每回合的最大步数
early_stop_threshold = 100
no_improvement_count = 0
best_reward = -float('inf')
best_model_path = 'best_dqn_model.pth'
# 创建环境
env = gym.make("CartPole-v1")

# 经验回放缓冲区
memory = deque(maxlen=memory_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建 Q 网络和目标网络
q_network = DQN(state_size, action_size).to(device)
target_network = DQN(state_size, action_size).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

# 优化器
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# 存储经验
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# 选择动作 (epsilon-greedy 策略)
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.choice(np.arange(action_size))  # 随机探索
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = q_network(state)
            return torch.argmax(q_values).item()  # 选择 Q 值最高的动作

# 经验回放训练
def replay():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    
    states, actions, rewards, next_states, dones = zip(*batch)
    # 转换为 PyTorch 张量
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # 当前 Q 网络预测的 Q 值
    q_values = q_network(states).gather(1, actions)

    # 目标 Q 值
    next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # 计算损失
    loss = nn.MSELoss()(q_values, target_q_values)

    # 反向传播更新网络
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 训练 DQN
for episode in range(episodes):
    state = env.reset()
    state = state[0]  # 提取实际的状态数组
    total_reward = 0

    for step in range(max_steps):
        # 选择动作
        action = select_action(state, epsilon)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # 只要其中一个为True，表示回合结束
        reward = reward if not terminated else -10  # 如果 terminated, 则惩罚

        total_reward += reward

        # 存储经验
        store_experience(state, action, reward, next_state, done)
        
        # 更新当前状态
        state = next_state
        # 执行经验回放训练
        replay()
        
        if done:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")
            break
    
    # 每隔 target_update 步更新目标网络
    if episode % target_update == 0:
        target_network.load_state_dict(q_network.state_dict())
    
    # 衰减 epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if total_reward > best_reward:
        best_reward = total_reward
        no_improvement_count = 0
        # 保存当前最优模型
        torch.save(q_network.state_dict(), best_model_path)
        print(f"New best model saved at episode {episode + 1}, reward: {best_reward}")
    else:
        no_improvement_count += 1

    # 如果超过 early_stop_threshold 回合没有提升，停止训练
    if no_improvement_count >= early_stop_threshold:
        print(f"Early stopping at episode {episode + 1}, best reward: {best_reward}")
        break


env.close()
# 测试 DQN
q_network.load_state_dict(torch.load(best_model_path))
print("Loaded the best model for testing.")
env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()
state = state[0]  # 提取实际的状态数组
env.render()
for step in range(max_steps):
    action = select_action(state, epsilon=0)  # 测试时不进行探索
    next_state, reward, terminated, truncated, _ = env.step(action)
    env.render()
    time.sleep(0.04)
    state = next_state
    if terminated or truncated:
        break

env.close()
