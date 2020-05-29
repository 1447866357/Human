import gym, os, BaseQLearningAgents
import numpy as np
import ControlSharing as cs
import RewardShaping as rs
from tqdm import tqdm
import ActionBiasing as ab
import MultiBandit as mb
import pickle
MAX_EPISODE = 2000
ENV_NAME = 'CartPole-v0'

model_base_path = 'model'

# L = 0.base_q
# R = 10
# C = 0.8

# agent = BaseQLearning.BaseDeepQLearning(env, os.path.join(model_base_path, 'base', 'ckpt.ckpt'))
#
# agent = cs.ControlSharingAgent(env,
#                                os.path.join(model_base_path, 'base', 'ckpt.ckpt'),
#                                os.path.join(model_base_path, 'feedback', 'control_sharing', 'model.ckpt'), L, R, C)

# agent = rs.RewardShapingAgent(env,
#                               os.path.join(model_base_path, 'base', 'ckpt.ckpt'),
#                               os.path.join(model_base_path, 'feedback', 'reward_shaping', 'model.ckpt'), L, R, C)

# agent = ab.ActionBiasingAgent(env,
#                               os.path.join(model_base_path, 'base', 'ckpt.ckpt'),
#                               os.path.join(model_base_path, 'feedback', 'AB', 'model.ckpt'), L, R, C)

# agent = mb.MultiBanditAgent(env,
#                               os.path.join(model_base_path, 'base', 'ckpt.ckpt'),
#                               os.path.join(model_base_path, 'feedback', 'multi_bandit', 'model.ckpt'), L, R, C)

def main():
    episode_reward = []
    epsilon = 0.5
    for episode in tqdm(range(MAX_EPISODE)):
        state = env.reset()
        episode_reward.append(0)
        while True:
            action = agent.egreedy_action(state, epsilon)
            new_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, new_state, done)
            episode_reward[-1] += reward
            if done:
                epsilon -= 0.5 / MAX_EPISODE
                agent.B -= 1 / MAX_EPISODE
                if agent.B < 0:
                    agent.B = 0

                if epsilon < 0:
                    epsilon = 0
                break
            state = new_state

        # print(episode, episode_reward[-base_q])
        # agent.dump()

    return episode_reward

if __name__ == '__main__':



    L = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.,18]
    R = [10,100]
    C = list(np.arange(0.2, 1, 0.2))
    # C = [0.5,0.8,base_q]
    from CartPole_OLD.ave import acverRange
    import pandas as pd
    for l in L:
        for r in R:
            for c in C:
                print('base_q',l,r,c)



                env = gym.make(ENV_NAME)

                agent = mb.MultiBanditAgent(env,
                                            os.path.join(model_base_path, 'base', 'ckpt.ckpt'),
                                            os.path.join(model_base_path, 'feedback', 'control_sharing', 'model.ckpt'),
                                            l, r, c)

                print(l)
                runs = 3
                for i in range(runs):
                    print()
                    print('run',i)
                    total_reward = []
                    # reward_means = []
                    episode_reward = main()
                    agent.init_network()
                    print('base_q', episode_reward, len(episode_reward))

                    total_reward.append(episode_reward)
                    print("l: %.1f, r: %d, C: %.1f" % (l, r, c))
                    print("max reward: ", max(episode_reward))

                    # for i in range(0, MAX_EPISODE, MAX_EPISODE1 // 10):
                    #     reward_means.append(np.average(episode_reward[i: i+MAX_EPISODE // 10]).astype(np.int))
                    reward_means = acverRange(episode_reward)

                    print("mean reward: ", reward_means, len(reward_means))

                    dict = { 'Aver_reward': reward_means, 'Reward': episode_reward}
                    data = pd.DataFrame(dict)
                    pd.DataFrame(data)
                    pd.DataFrame(data).to_csv(
                        '/Users/d/work/Human/CartPole_OLD/data/Adaptive Shaping/data_' +  str(l) + '_'+ str(c) + '_'+ str(r) + '_' +  str(i) + '_'  + str(MAX_EPISODE) + '.csv', header=dict)

                agent.close()


# if __name__ == '__main__':
#     total_reward = []
#     # L = np.arange(0.base_q, base_q, 0.2)
#     # R = [10]
#     # C = np.arange(0.5, base_q, 0.base_q)
#     L = [1]
#     R = [10]
#     # C = np.arange(0.5, base_q, 0.base_q)
#     C = [0.05]
#     for l in L:
#         for r in R:
#             for c in C:
#                 env = gym.make(ENV_NAME)
#                 agent = mb.MultiBanditAgent(env,
#                                                os.path.join(model_base_path, 'base', 'ckpt.ckpt'),
#                                                os.path.join(model_base_path, 'feedback', 'control_sharing', 'model.ckpt'), l, r, c)
#
#                 for i in range(3):
#                     episode_reward = main()
#                     agent.init_network()
#                     total_reward.append(episode_reward)
#                     print("l: %.1f, r: %d, C: %.1f" % (l, r, c))
#                     print("max reward: ", max(episode_reward))
#                     reward_means = []
#                     for i in range(0, MAX_EPISODE, MAX_EPISODE // 10):
#                         reward_means.append(np.average(episode_reward[i: i+MAX_EPISODE // 10]).astype(np.int))
#
#                     print("mean reward: ", reward_means)
#                     pickle.dump(total_reward, open('pic/feedback/multi_bandit/data_%.1f_%.1f_%.2f.pkl' % (l, r, c), 'wb'))
#                 total_reward = []
#                 agent.close()

