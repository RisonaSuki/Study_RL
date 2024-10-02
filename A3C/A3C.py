import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# 设置环境变量以解决 OpenMP 错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置日志记录
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# 环境的超参数
ENV_NAME = "Ant-v4"
NUM_WORKERS = 8  # 并发的worker数量
MAX_EPISODE_LENGTH = 1000  # 每个episode的最大长度
GAMMA = 0.99  # 折扣因子
TAU = 1.0  # GAE中的tau
LR = 1e-5  # 学习率
ENTROPY_WEIGHT = 0.01  # 熵的权重
GLOBAL_MAX_EPISODES = 10000  # 全局训练episode数量

# 全局设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 策略网络和值网络
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 256)
        self.policy_mu = nn.Linear(256, action_dim)
        self.policy_sigma = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        mu = torch.tanh(self.policy_mu(x))  # tanh将动作限制在[-1, 1]
        sigma = torch.exp(self.policy_sigma(x))  # 保证 sigma 为正数
        value = self.value(x)
        return mu, sigma, value

# Worker线程
class Worker(mp.Process):
    def __init__(self, global_model, optimizer, global_episode_count, global_results, best_reward, best_reward_lock, result_lock, worker_id, input_dim, action_dim):
        super(Worker, self).__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.local_model = ActorCritic(input_dim, action_dim).to(device)
        self.env = gym.make(ENV_NAME)
        self.global_episode_count = global_episode_count
        self.global_results = global_results
        self.best_reward = best_reward
        self.best_reward_lock = best_reward_lock
        self.result_lock = result_lock
        self.worker_id = worker_id

    def run(self):
        while self.global_episode_count.value < GLOBAL_MAX_EPISODES:
            obs, info = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            episode_reward = 0
            done = False
            while not done:
                mu, sigma, _ = self.local_model(obs)
                dist = Normal(mu, sigma)
                action = dist.sample().squeeze()  # 去掉多余的维度，确保动作是(8,)
                action = torch.clamp(action, -1.0, 1.0)  # 限制动作在[-1, 1]之间
                action = action.cpu().numpy()

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)

                if done:
                    break

            # 更新全局网络
            with self.global_episode_count.get_lock():
                self.global_episode_count.value += 1
                current_episode = self.global_episode_count.value
                # 使用锁来保护对 global_results 的访问
                with self.result_lock:
                    self.global_results.append(episode_reward)

                # 保存最佳模型，使用锁保护 best_reward 更新
                with self.best_reward_lock:
                    if episode_reward > self.best_reward.value:
                        self.best_reward.value = episode_reward
                        save_model(self.global_model, 'best_model.pth')
                        print(f"New best model saved with reward: {self.best_reward.value}")

                # 打印训练进度
                if current_episode % 10 == 0:  # 每10个episode打印一次
                    avg_reward = np.mean(self.global_results[-10:])
                    logging.info(f"Worker {self.worker_id} - Episode {current_episode} - Reward: {episode_reward} - Avg Reward (last 10): {avg_reward}")
                    print(f"Worker {self.worker_id} - Episode {current_episode} - Reward: {episode_reward} - Avg Reward (last 10): {avg_reward}")

    def update_global(self):
        # 把局部网络更新到全局网络
        for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()
        self.local_model.load_state_dict(self.global_model.state_dict())

# 测试模型并展示训练结果
def test_model(global_model, input_dim, action_dim, render=False):
    env = gym.make(ENV_NAME,render_mode="rgb_array")
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < MAX_EPISODE_LENGTH:
        if render:
            env.render()
        mu, sigma, _ = global_model(obs)
        dist = Normal(mu, sigma)
        action = dist.sample().squeeze()  # 确保动作是(8,)维度的向量
        action = torch.clamp(action, -1.0, 1.0)  # 确保动作在[-1, 1]之间
        action = action.cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    env.close()
    print(f"Total reward in test: {total_reward}")

# 全局训练函数
def train(global_model, optimizer, input_dim, action_dim):
    global_episode_count = mp.Value('i', 0)
    global_results = mp.Manager().list()
    best_reward = mp.Value('d', float('-inf'))  # 初始化为负无穷大
    result_lock = mp.Lock()  # 创建一个锁来保护 global_results
    best_reward_lock = mp.Lock()  # 用于保护 best_reward 的更新
    workers = [Worker(global_model, optimizer, global_episode_count, global_results, best_reward, best_reward_lock, result_lock, i, input_dim, action_dim) for i in range(NUM_WORKERS)]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    return global_results

# 保存模型函数
def save_model(global_model, path):
    torch.save(global_model.state_dict(), path)

# 主函数
if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    input_dim = env.observation_space.shape[0]  # 获取正确的observation维度
    action_dim = env.action_space.shape[0]  # 获取正确的action维度
    global_model = ActorCritic(input_dim, action_dim).to(device)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=LR)

    results = train(global_model, optimizer, input_dim, action_dim)

    plt.plot(results)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('A3C Training Progress on Ant-v4')
    plt.show()

    # 训练结束后，测试模型并展示环境渲染
    print("Testing the trained model...")
    test_model(global_model, input_dim, action_dim, render=True)
