import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

reward_shaping = pickle.load(open('/Users/d/work/Human/CartPole_OLD/pic/feedback/reward_shaping/data.pkl', 'rb'))
baseq = pickle.load(open('/Users/d/work/Human/CartPole_OLD/pic/feedback/base_q/data.pkl', 'rb'))
action_biasing = pickle.load(open('/Users/d/work/Human/CartPole_OLD/pic/feedback/AB/data.pkl', 'rb'))
multi_bandit = pickle.load(open('/Users/d/work/Human/CartPole_OLD/pic/feedback/multi_bandit/data.pkl', 'rb'))
control_sharing = pickle.load(open('/Users/d/work/Human/CartPole_OLD/pic/feedback/control_sharing/data.pkl', 'rb'))
q_update = pickle.load(open('/Users/d/work/Human/CartPole_OLD/pic/feedback/qupdate/data.pkl', 'rb'))

def paint_smooth(data):
    for i in range(len(data)):
        if i < 20:
            data[i] -= 20
            data[i] = np.average(data[i: i + 100])
        else:
            data[i] = np.average(data[i:i + 100])
    return data[:850]

x = list(range(0, 850)) * 3
rs_y = paint_smooth(reward_shaping[1]) + paint_smooth(reward_shaping[1]) + paint_smooth(reward_shaping[2])
base_q_y = paint_smooth(baseq[0]) + paint_smooth(baseq[1]) + paint_smooth(baseq[2])
ab_y = paint_smooth(action_biasing[0]) + paint_smooth(action_biasing[1]) + paint_smooth(action_biasing[2])
cs_y = paint_smooth(control_sharing[0]) + paint_smooth(control_sharing[1]) + paint_smooth(control_sharing[2])
qu_y = paint_smooth(q_update[0]) + paint_smooth(q_update[1]) + paint_smooth(q_update[2])
mu_y = paint_smooth(multi_bandit[1]) + paint_smooth(multi_bandit[1]) + paint_smooth(multi_bandit[2])
print(1)
plt.figure()
rs = pd.DataFrame.from_dict({"episode": x, "reward": rs_y})
sns.lineplot("episode", "reward", data=rs)
baseq = pd.DataFrame.from_dict({"episode": x, "reward": base_q_y})
sns.lineplot("episode", "reward", data=baseq)
ab = pd.DataFrame.from_dict({"episode": x, "reward": ab_y})
sns.lineplot("episode", "reward", data=ab)
cs = pd.DataFrame.from_dict({"episode": x, "reward": cs_y})
sns.lineplot("episode", "reward", data=cs)
qu = pd.DataFrame.from_dict({"episode": x, "reward": qu_y})
sns.lineplot("episode", "reward", data=qu)
mu = pd.DataFrame.from_dict({"episode": x, "reward": mu_y})
sns.lineplot("episode", "reward", data=mu)
plt.legend(["Reward Shaping", "Q-learning", "Action Biasing", "Control Sharing", "QUpdate", "Adaptive Shaping"])
plt.show()


