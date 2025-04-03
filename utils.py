from matplotlib import pyplot as plt
import numpy as np

# 动作
STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
ACTIONS = [STAY, UP, DOWN, LEFT, RIGHT]

# 奖励
TARGET_REWARD = 10  # 到达目标奖励
BOUNDARY_PENALTY = -1  # 进入禁区惩罚
WALL_PENALTY = -1  # 撞墙惩罚
BASE_REWARD = 0  # 每步的基础奖励


class Environment:
    def __init__(self, len_length=5, len_wide=5):
        self.row = len_length
        self.col = len_wide
        self.states_num = self.row * self.col  # 状态数
        self.grid = [[0 for _ in range(self.col)] for _ in range(self.row)]
        self.agent_pos = [0, 0]  # 智能体初始位置
        self.target_pos = [3, 2]  # 目标位置
        self.done = False
        self.max_steps = 100  # 最大步数限制
        self.current_steps = 0

        # 初始化环境，指定障碍物和目标
        if self.row == 5 and self.col == 5:
            self.grid[1][1] = -1  # 障碍物
            self.grid[1][2] = -1
            self.grid[2][2] = -1
            self.grid[3][1] = -1
            self.grid[3][3] = -1
            self.grid[4][1] = -1
            self.grid[3][2] = 1  # 目标

    def reset(self):
        # 重置智能体位置和其他状态
        self.agent_pos = [0, 0]
        self.done = False
        self.current_steps = 0
        return self.agent_pos

    def step(self, state, action):
        """
        执行一步动作
        该函数也定义环境，动作对应的下一个状态和奖励
        参数：
            state: 当前状态
            action: 动作（STAY=0, UP=1, DOWN=2, LEFT=3, RIGHT=4）
        返回：
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # self.current_steps += 1
        reward = BASE_REWARD  # 每步的基础奖励
        # old_pos = self.agent_pos.copy()
        next_state = state.copy()

        # 根据动作更新位置
        if self.is_valid_actions(state, action):
            if action == UP:
                next_state[0] -= 1
            elif action == DOWN:
                next_state[0] += 1
            elif action == LEFT:
                next_state[1] -= 1
            elif action == RIGHT:
                next_state[1] += 1
        else:
            reward = WALL_PENALTY  # 撞墙惩罚

        # 检查是否进入禁区
        if self.grid[next_state[0]][next_state[1]] == -1:
            reward = BOUNDARY_PENALTY  # 进入禁区惩罚

        # 检查是否到达目标
        if next_state == self.target_pos:
            reward = TARGET_REWARD  # 到达目标奖励
            self.done = True

        # # 检查是否达到最大步数
        # if self.current_steps >= self.max_steps:
        #     self.done = True

        # info = {"steps": self.current_steps}
        return next_state, reward

    # def get_state(self):
    #     """
    #     返回当前状态（包括网格信息和智能体位置）
    #     """
    #     state = np.array(self.grid)
    #     state = state.copy()
    #     # 将智能体位置标记为2
    #     if not self.done:
    #         state[self.agent_pos[0]][self.agent_pos[1]] = 2
    #     return state

    def is_valid_actions(self, state, action):
        """
        判断当前动作是否有效，若有效则返回True，否则返回False
        """
        return not (
                (state[0] == 0 and action == UP) or
                (state[0] == self.row - 1 and action == DOWN) or
                (state[1] == 0 and action == LEFT) or
                (state[1] == self.col - 1 and action == RIGHT)
        )

    def render(self, show=True):
        # 创建颜色映射
        colors = []
        for i in range(self.row):
            row_colors = []
            for j in range(self.col):
                if self.grid[i][j] == -1:  # 障碍物
                    row_colors.append('orange')
                elif self.grid[i][j] == 1:  # 目标
                    row_colors.append('blue')
                else:  # 空白区域
                    row_colors.append('white')
            colors.append(row_colors)

        # 创建图形
        plt.figure(figsize=(8, 8))

        # 绘制网格
        for i in range(self.row):
            for j in range(self.col):
                plt.fill([j, j + 1, j + 1, j], [i, i, i + 1, i + 1], color=colors[i][j])
                plt.plot([j, j + 1, j + 1, j, j], [i, i, i + 1, i + 1, i], 'black')

        # 设置图形属性
        plt.xlim(0, self.col)
        plt.ylim(self.row, 0)
        plt.grid(True)
        if show:
            plt.show()


class Policy:
    def __init__(self, env):
        self.env = env
        self.row = env.row
        self.col = env.col
        self.p = np.ones((self.row, self.col, len(ACTIONS))) * (1 / len(ACTIONS))

    def get_action(self, state):
        return np.random.choice(ACTIONS, p=self.p[state[0], state[1]])

    def update_policy(self, q):
        self.p = np.eye(len(ACTIONS))[np.argmax(q, axis=-1)]

    def update_policy_with_epsilon_greedy(self, q, epsilon):
        self.p = (1 - epsilon) * np.eye(len(ACTIONS))[np.argmax(q, axis=-1)] \
                 + epsilon * np.ones((self.row, self.col, len(ACTIONS))) * (1 / len(ACTIONS))

    def update_policy_segment(self, q, s):
        # 当前操作的状态
        q_s_argmax = np.argmax(q[s[0], s[1]])
        # 更新
        self.p[s[0], s[1]] = np.eye(len(ACTIONS))[q_s_argmax]

    def update_policy_segment_with_epsilon_greedy(self, q, s, epsilon):
        # 当前操作的状态
        q_s_argmax = np.argmax(q[s[0], s[1]])
        # 更新
        self.p[s[0], s[1]] = (1 - epsilon) * np.eye(len(ACTIONS))[q_s_argmax] + epsilon * np.ones(len(ACTIONS)) * (
                    1 / len(ACTIONS))

    def render(self):
        self.env.render(show=False)
        # 利用p的前两维绘制一个网络，第三维绘制箭头，从位置1到4箭头方向分别为上，下，左，右，箭头大小为p的值，0表示不动；当不动时，画一个圈，圈直径等于p值

        for i in range(self.row):
            for j in range(self.col):
                plt.plot([j, j + 1, j + 1, j, j], [i, i, i + 1, i + 1, i], 'black')
                for a in range(len(ACTIONS)):
                    if self.p[i, j, a] > 0:
                        if a == 1:  # UP
                            plt.arrow(j + 0.5, i + 0.5, 0, -0.4 * self.p[i, j, a], head_width=0.05, head_length=0.05,
                                      fc='r', ec='r')
                        elif a == 2:  # DOWN
                            plt.arrow(j + 0.5, i + 0.5, 0, 0.4 * self.p[i, j, a], head_width=0.05, head_length=0.05,
                                      fc='r', ec='r')
                        elif a == 3:  # LEFT
                            plt.arrow(j + 0.5, i + 0.5, -0.4 * self.p[i, j, a], 0, head_width=0.05, head_length=0.05,
                                      fc='r', ec='r')
                        elif a == 4:  # RIGHT
                            plt.arrow(j + 0.5, i + 0.5, 0.4 * self.p[i, j, a], 0, head_width=0.05, head_length=0.05,
                                      fc='r', ec='r')
                        elif a == 0:  # STAY
                            circle = plt.Circle((j + 0.5, i + 0.5), 0.4 * self.p[i, j, a], color='r', fill=False)
                            plt.gca().add_patch(circle)

        plt.xlim(0, self.col)
        plt.ylim(self.row, 0)
        plt.grid(True)
        plt.show()


def L2_norm(p1, p2):
    assert p1.shape == p2.shape, "p1 and p2 must have the same shape"
    return np.sqrt(np.sum((p1 - p2) ** 2))


# epsilon贪心
def epsilon_greedy(p, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        return np.argmax(p)


# 归一化
def normalize(x, min_val, max_val):
    return (np.array(x) - min_val) / (max_val - min_val + 1e-10)


# rmse
def rmse(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    #plt.savefig(figure_file)
