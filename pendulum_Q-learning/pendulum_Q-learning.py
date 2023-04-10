import math
import self_pendulum_env
import numpy as np
import pandas
import pandas as pd

np.random.seed(2)  # 随机种子，便于代码复现
Angle_N = 60  # 代表角度等分成多少份
Angular_Velocity_N = 40  # 速度等分成多少份

N_Angle_STATES = np.linspace(-math.pi, math.pi, Angle_N)
N_Angle_STATES = np.round(N_Angle_STATES, decimals=4)
N_Angular_Velocity_STATES = np.linspace(-15 * math.pi, 15 * math.pi, Angular_Velocity_N)  # 离散化角速度
N_Angular_Velocity_STATES = np.round(N_Angular_Velocity_STATES, decimals=4)
U_Voltage_ACTIONS = [-3, 0, 3]  # 离散化可选电压的动作集
EPSILON = 0.9  # 贪心策略参数
Alp = 0.1  # 更新时的学习率
Lambda = 0.98  # 折扣因子
MAX_EPISODES = 100  # 最大轮数
FRESH_TIME = 0.0001  # 刷新时间
ONE_TURN_STEP = 50000

# pendulum参数
m = 0.055  # kg 重量
g = 9.81  # m/s2 重力加速度
l = 0.042  # m 重心到转子的距离
J = 1.91e-04  # kg · m2 转动惯量
b = 3.0e-06  # Nm · s/rad 粘滞阻尼
K = 0.0536  # Nm/A 转矩常数
R = 9.5  # Ω 转子电阻
T_s = 0.005  # 采样频率


def build_q_table(n_angle_states, n_angular_velocity_states, u_voltage_actions):
    index = pd.MultiIndex.from_product([n_angle_states, n_angular_velocity_states])  # 以(角度，角速度)为索引构建Q表
    columns = u_voltage_actions  # 列值为动作集

    # 构造Q表
    table = pd.DataFrame(
        np.zeros((len(n_angle_states) * len(n_angular_velocity_states), len(u_voltage_actions))), index=index,
        columns=columns
    )
    table.index.names = ['angle', 'angular_velocity']  # 标注索引名
    # print(table)
    return table


def choose_action(state: tuple, q_table_: pandas.DataFrame):
    state_actions = q_table_.loc[state]  # 获得当前state对应的action值
    random_rate = np.random.uniform()  # 获得一个随机数
    if (random_rate > EPSILON) or (state_actions.sum() == 0):
        action = np.random.choice(U_Voltage_ACTIONS)  # ε-greedy中探索性选择动作
    else:
        action = state_actions.idxmax()  # 贪心策略选择较大概率的动作

    return action


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def get_a__(a, a_, action):
    a = angle_normalize(a)
    a_pp = (1 / J) * (
            m * g * l * math.sin(a) - b * a_ - (K ** 2 / R) * a_ + (
            K / R) * action)  # 计算角加速度
    return a_pp


def get_approximate(val, states):
    idx = np.digitize(val, bins=states)
    if 0 < idx < len(states):
        left = states[idx - 1]
        right = states[idx]
        if np.abs(left - val) < np.abs(right - val):
            idx = idx - 1
        else:
            idx = idx
    elif idx <= 0:
        idx = 0
    else:
        idx = len(states) - 1
    return states[idx]


def round_state(state):
    a_k = state[0]
    a_p_k = state[1]

    temp = a_k
    x_ = np.arctan2(np.sin(temp), np.cos(temp))  # 映射到[-pi,pi]

    a_k = get_approximate(x_, states=N_Angle_STATES)

    a_p_k = get_approximate(a_p_k, states=N_Angular_Velocity_STATES)

    return a_k, a_p_k


def get_env_feedback_v2(env, action):
    action = [action]
    state, reward = env.step(action)
    state = tuple(state)
    return state, reward


def get_env_feedback(state: tuple, action):
    a_k = state[0]  # 角度
    a_p_k = state[1]  # 角速度/pi

    Reward = - (5 * (angle_normalize(a_k) ** 2) + 0.1 * (a_p_k) ** 2 + action ** 2)

    # 获得角加速度
    a_pp = get_a__(a_k, a_p_k, action)

    # 更新角速度
    a_p_k_1 = a_p_k + T_s * a_pp
    a_p_k_1 = np.clip(a_p_k_1, min(N_Angular_Velocity_STATES), max(N_Angular_Velocity_STATES))

    # 更新角度
    a_k_1 = a_k + T_s * a_p_k_1

    new_state = (a_k_1, a_p_k_1)

    return new_state, Reward


def update_env(state, episode, step_counter):
    if is_final_state(state):
        interaction = "Episode %s：total_step = %s" % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        # time.sleep(2)
        print('\n wowow finished!')
    else:

        state = round_state(state)
        interaction = "Episode %s：step = %s" % (episode + 1, step_counter)
        interaction += " angle：" + str(state[0] * 180 / math.pi) + ", angular_velocity：" + str(
            state[1] / math.pi) + "pi"
        print('\r{}'.format(interaction), end='')
        # time.sleep(FRESH_TIME)


def is_final_state(state):
    a = state[0]
    a_p = state[1]
    is_mid_a = is_mid(a, N_Angle_STATES)
    is_mid_a_p = is_mid(a_p, N_Angular_Velocity_STATES)
    return is_mid_a and is_mid_a_p


def is_mid(val, states):
    idx_mid = len(states) // 2
    if len(states) % 2 == 0:
        right = states[idx_mid]
        left = states[idx_mid - 1]
        return left < val < right
    else:
        mid = states[idx_mid]
        return val == mid


def run_pendulum_qlearning(env, is_view=False):
    # 构造Q表
    q_table = build_q_table(N_Angle_STATES, N_Angular_Velocity_STATES, U_Voltage_ACTIONS)

    # 训练MAX_EPISODES个回合
    for episode in range(MAX_EPISODES):
        step_counter = 0
        # 与环境交互
        if is_view:
            state, _ = env.reset()
            state = tuple(state)
        else:
            state, _ = env.reset(is_view=is_view)
            state = tuple(state)

        # 单个回合训练过程
        is_terminated = False
        update_env(state, episode, step_counter)
        while step_counter < ONE_TURN_STEP and (not is_terminated):
            # while not is_terminated:
            state_round = round_state(state)
            # 拿到最大Q值的动作
            action = choose_action(state_round, q_table)
            # 拿到最大Q值的动作对应的Q值
            q_predict = q_table.loc[state_round, action]

            # 与环境交互
            if not is_view:
                state_new, Reward = get_env_feedback(state, action)
            else:
                state_new, Reward = get_env_feedback_v2(env, action)

            # 计算Q-target
            if not is_final_state(state_new):
                q_target = Reward + Lambda * max(q_table.loc[round_state(state_new)])
            else:
                q_target = Reward
                update_env(state, episode, step_counter)
                is_terminated = True

                # 更新Q表
            q_table.loc[state_round, action] += Alp * (q_target - q_predict)
            state = state_new

            # 打印控制台
            update_env(state, episode, step_counter)
            step_counter += 1

    return q_table


def read_q_table(env):
    q_table = pd.read_csv('data/result_q_table.csv', index_col=[0, 1], header=0)
    is_view = True
    state, _ = env.reset(is_view=is_view)
    state = tuple(state)
    step_counter = 0
    is_terminated = False
    while step_counter < ONE_TURN_STEP and (not is_terminated):
        state_round = round_state(state)
        # 拿到最大Q值的动作
        action = choose_action(state_round, q_table)
        action = np.array([action], dtype='float64')
        next_state, reward = env.step(action)
        if is_final_state(next_state):
            is_terminated = True
            env.close()

        state = next_state
        step_counter += 1

    print(q_table)


if __name__ == '__main__':
    env = self_pendulum_env.PendulumEnv("human")
    # 训练时，最好把is_view关闭，因为动画渲染需要时间，影响了整体训练的速度
    # q_table = run_pendulum_qlearning(env, is_view=False)
    # q_table.to_csv('data/result_q_table.csv', index=True, header=True, date_format='%.4f')
    read_q_table(env)

#
# print(N_Angle_STATES)
# print(N_Angular_Velocity_STATES)
#
# a = -1.638
# idx = np.digitize(a, bins=N_Angle_STATES)
# val = N_Angle_STATES[idx]
# print(val)
