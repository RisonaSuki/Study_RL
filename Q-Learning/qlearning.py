import gym
import numpy as np
import random

# 创建环境并指定 render_mode
env = gym.make("FrozenLake-v1")

# 初始化 Q-Table
state_size = env.observation_space.n  # 状态数量
action_size = env.action_space.n  # 动作数量
q_table = np.zeros((state_size, action_size))

# 超参数
learning_rate = 1  # α，学习率
discount_rate = 0.95  # γ，折扣因子
epsilon = 1.0  # ε，探索率
epsilon_decay = 0.995  # 探索率衰减
min_epsilon = 0.01
episodes = 100000
max_steps = 1000  # 每个episode的最大步数

# 用于保存每个episode的总奖励
rewards = []

# 训练 Q-Learning 智能体
for episode in range(episodes):
    state = env.reset()  # 重置环境，初始化状态
    if isinstance(state, tuple):
        state = state[0]
    state = int(state)

    total_rewards = 0
    
    for step in range(max_steps):
        # 使用 epsilon-greedy 策略选择动作
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 探索
        else:
            action = np.argmax(q_table[state, :])  # 利用
        action = int(action)

        # 执行动作，得到下一个状态和奖励
        next_state, reward, done, truncated, info = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = int(next_state)
        
        # 更新Q值
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_rate * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        
        state = next_state
        total_rewards += reward
        
        if done or truncated:
            break
    # 逐渐减少探索率
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_rewards)
env.close()
env = gym.make("FrozenLake-v1", render_mode="human")
# 测试智能体表现
for episode in range(5):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = int(state)
    
    done = False
    print(f"Episode {episode + 1}")
    print(q_table)
    for step in range(max_steps):
        action = np.argmax(q_table[state, :])  # 选择最优动作
        action = int(action)
        next_state, reward, done, truncated, info = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = int(next_state)
        
        env.render()  # 展示智能体的动作
        state = next_state
        
        if done or truncated:
            break

env.close()
