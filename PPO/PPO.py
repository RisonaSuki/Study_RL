import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

lr = 3e-4
gamma = 0.99
eps_clip = 0.2
update_timestep = 2000
K_epochs = 10
action_std = 0.5
early_stopping_threshold = 200 
best_reward = -float('inf')
no_improvement_steps = 0 
save_model_path = "best_ppo_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        action_std = self.log_std.exp().expand_as(action_mean.unsqueeze(0))
        action_dist = torch.distributions.Normal(action_mean, action_std.squeeze(0))
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action).sum(dim=-1)
        return action, action_logprob

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_std = self.log_std.exp().expand_as(action_mean)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        action_logprobs = action_dist.log_prob(action).sum(dim=-1)
        dist_entropy = action_dist.entropy().sum(dim=-1)
        state_value = self.critic(state)
        return action_logprobs, state_value, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, eps_clip):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        old_states = torch.tensor(np.array(memory.states), dtype=torch.float32).to(device)
        old_actions = torch.tensor(np.array(memory.actions), dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32).to(device)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(memory.dones, dtype=torch.float32).to(device)

        with torch.no_grad():
            returns = []
            discounted_reward = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)

            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            advantages = returns - self.policy.critic(old_states)

        for _ in range(K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]

def train_ppo(env, ppo, memory, max_episodes=5000):
    global best_reward, no_improvement_steps
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for t in range(1600): 
            state = torch.tensor(state, dtype=torch.float32).to(device)
            action, action_logprob = ppo.policy_old.act(state)
            action = action.cpu().detach().numpy()
            
            next_state, reward, terminate, truncated, _ = env.step(action)
            done = terminate or truncated
            episode_reward += reward
            
            memory.states.append(state.cpu().numpy())
            memory.actions.append(action)
            memory.logprobs.append(action_logprob.cpu().detach().numpy())
            memory.rewards.append(reward)
            memory.dones.append(done)

            state = next_state

            if done or t == 1600 - 1:
                ppo.update(memory)
                memory.clear_memory()
                print(f"Episode {episode}, Reward: {episode_reward}")
                
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    no_improvement_steps = 0
                    torch.save(ppo.policy.state_dict(), save_model_path)
                    print(f"New best model saved with reward: {best_reward}")
                else:
                    no_improvement_steps += 1

                if no_improvement_steps > early_stopping_threshold:
                    print(f"Early stopping at episode {episode}, best reward: {best_reward}")
                    env.close()
                    return

                break
    env.close()
    print("Training finished.")



if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(state_dim, action_dim, lr, gamma, eps_clip)
    memory = Memory()
    env.close()
    train_ppo(env, ppo, memory)
    print("Loading the best model for final rendering.")
    ppo.policy.load_state_dict(torch.load(save_model_path))
    env = gym.make('BipedalWalker-v3',render_mode='human')
    state, _ = env.reset()
    done = False
    while not done:
        env.render()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action, _ = ppo.policy_old.act(state)
        action = action.cpu().detach().numpy()
        state, _, terminate, truncated, _ = env.step(action)
        done = terminate or truncated