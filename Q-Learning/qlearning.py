import gym
import numpy as np
import random

env = gym.make("FrozenLake-v1")

state_size = env.observation_space.n
action_size = env.action_space.n 
q_table = np.zeros((state_size, action_size))

learning_rate = 0.8  # α
discount_rate = 0.95  # γ
epsilon = 1.0  # ε
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 10000
max_steps = 100

for episode in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = int(state)

    total_rewards = 0
    
    for step in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])
        action = int(action)

        next_state, reward, done, truncated, info = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = int(next_state)
        
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_rate * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        
        state = next_state
        total_rewards += reward
        
        if done or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()

env = gym.make("FrozenLake-v1", render_mode="human")

for episode in range(5):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = int(state)
    
    done = False
    print(f"Episode {episode + 1}")
    
    for step in range(max_steps):
        action = np.argmax(q_table[state, :])
        action = int(action)
        next_state, reward, done, truncated, info = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = int(next_state)
        
        env.render()
        state = next_state
        
        if done or truncated:
            break

env.close()