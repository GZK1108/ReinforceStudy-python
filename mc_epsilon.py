from utils import *

def mc_epsilon_greedy(env, policy, gamma=0.9, epsilon=0.1, num_episodes=1000, max_steps=100):
    
    G = np.zeros((env.row, env.col, len(ACTIONS)))
    G_count = np.zeros((env.row, env.col, len(ACTIONS)))
    for episode in range(num_episodes):
        episode_result = []
        
        state = [0,0]
        for step in range(max_steps):
            action = policy.get_action(state)
            next_state, reward = env.step(state, action)
            episode_result.insert(0, (state, action, reward))
            state = next_state
        
        for t, (state, action, reward) in enumerate(episode_result):
            if t > 1000:
                G[state[0], state[1], action] = reward + gamma * G[next_state[0], next_state[1], action]
                G_count[state[0], state[1], action] += 1

    q_table = G / (G_count + 1e-10)
    policy.update_policy_with_epsilon_greedy(q_table, epsilon)
    return policy


env = Environment(5, 5)
policy = Policy(env)
policy = mc_epsilon_greedy(env, policy, num_episodes=20, max_steps=100000)
print(policy.p)
policy.render()
