from utils import *

def sarsa(env, policy, gamma=0.9, epsilon=0.1, learning_rate=0.1, num_episodes=1000):
    q_table = np.zeros((env.row, env.col, len(ACTIONS)))
    for episode in range(num_episodes):
        state = env.reset()
        action = policy.get_action(state)
        while not env.done:
            next_state, reward = env.step(state, action)
            next_action = policy.get_action(next_state)
            q_table[state[0], state[1], action] -= learning_rate \
                                                * (q_table[state[0], state[1], action] 
                                                   - (reward + gamma * q_table[next_state[0], next_state[1], next_action]))
            policy.update_policy_segment_with_epsilon_greedy(q_table, state, epsilon)
            state = next_state
            action = next_action
    return q_table


def q_learning(env, policy_b, policy_T, gamma=0.9, learning_rate=0.1, num_episodes=1000, max_steps=1000):
    q_table = np.zeros((env.row, env.col, len(ACTIONS)))
    for episode in range(num_episodes):
        state = env.reset()
        for t in range(max_steps):
            action = policy_b.get_action(state)
            next_state, reward = env.step(state, action)
            q_table[state[0], state[1], action] -= learning_rate \
                                                * (q_table[state[0], state[1], action] 
                                                   - (reward + gamma * np.max(q_table[next_state[0], next_state[1]])))
            policy_T.update_policy_segment(q_table, state)
            state = next_state
    return q_table
            

env = Environment(5, 5)
policy_b = Policy(env)
policy_T = Policy(env)
q_table = q_learning(env, policy_b, policy_T, num_episodes=100, max_steps=10000)
policy_T.render()
