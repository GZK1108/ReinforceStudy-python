from utils import *
epsilon = 1e-6

# 计算动作值
def compute_action_value(env, V, gamma=0.9):
    q_table = np.zeros((env.row, env.col, len(ACTIONS)))
    for r in range(env.row):
        for c in range(env.col):
            for a in ACTIONS:
                next_state, reward = env.step([r, c], a)
                q_table[r, c, a] = reward + gamma * V[next_state[0], next_state[1]]
    return q_table


# 求解贝尔曼方程
def bellman_equation(env, policy, gamma=0.9):
    V = np.zeros((env.row, env.col))
    while True:
        V_old = V.copy()
        q_table = compute_action_value(env, V, gamma)
        assert q_table.shape == policy.p.shape, "Error"
        V = np.sum(policy.p * q_table, axis=-1)
        if L2_norm(V, V_old) < epsilon:
            break
    return np.round(V, 1)


# 值迭代
def value_iteration(env, policy, gamma=0.9):
    V = np.zeros((env.row, env.col))
    while True:
        V_old = V.copy()
        # 计算动作值
        q_table = compute_action_value(env, V, gamma)
        # 策略更新
        policy.update_policy(q_table)
        # 值更新
        V = np.max(q_table, axis=-1)
        # 收敛判断
        if L2_norm(V, V_old) < epsilon:
            break
    return np.round(V, 1)

# 策略迭代
def policy_iteration(env, policy, gamma=0.9):
    V_k1 = np.zeros((env.row, env.col))
    while True:
        # 策略评估
        V_k = V_k1.copy()
        while True:
            V_k_old = V_k.copy()
            q_table = compute_action_value(env, V_k, gamma)
            assert q_table.shape == policy.p.shape, "Error"
            V_k = np.sum(policy.p * q_table, axis=-1)
            if L2_norm(V_k, V_k_old) < epsilon:
                break
        # 策略改进
        policy.update_policy(q_table)
        V_k1 = np.max(q_table, axis=-1)
        if L2_norm(V_k1, V_k) < epsilon:
            break
    return np.round(V_k1, 1)
        



       
