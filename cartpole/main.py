import gym, os, BaseQLearningAgents
import numpy as np
import ControlSharing as cs
import RewardShaping as rs
from tqdm import tqdm
import ActionBiasing as ab
import MultiBandit as mb
import pickle
import pandas as pd
MAX_EPISODE = 2000
ENV_NAME = 'MountainCar-v0'

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
                # agent.B -= base_q / MAX_EPISODE
                if agent.B < 0:
                    agent.B = 0

                if epsilon < 0:
                    epsilon = 0
                break
            state = new_state

        # print(episode, episode_reward[-base_q])
        # agent.dump()
    # print('base_q',episode_reward,len(episode_reward))
    return episode_reward

def acverRange(Y):
    averageRange = 10
    nEpisodes = len(Y)

    smoothedRewards = np.copy(Y)

    for i in range(averageRange, nEpisodes):
        smoothedRewards[i] = np.mean(Y[i - averageRange:i + 1])
    return smoothedRewards


if __name__ == '__main__':


    L = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.,18]
    R = [10,100]
    C = list(np.arange(0.2, 1, 0.2))
    # C = [0.5,0.8,base_q]

    for l in L:
        for r in R:
            for c in C:
                print('base_q',l,r,c)



                env = gym.make(ENV_NAME)
                agent = cs.ControlSharingAgent(env,
                                               os.path.join(model_base_path, 'base', 'ckpt.ckpt'),
                                               os.path.join(model_base_path, 'feedback', 'control_sharing', 'model.ckpt'), l, r, c)
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

                    # pickle.dump(total_reward, open('pic/feedback/control_sharing/data_%.1f_%.1f_%.1f.pkl' % (l, r, c), 'wb'))
                    # save = pd.DataFrame(reward_means, episode_reward, columns=['Reward', 'Aver_reward'])
                    # save.to_csv('\\Users\\d\\work\\Human\\CartPole_OLD\\pic\\feedback\\control_sharing\\data.csv')

                    dict = { 'Aver_reward': reward_means, 'Reward': episode_reward}
                    data = pd.DataFrame(dict)
                    pd.DataFrame(data)
                    pd.DataFrame(data).to_csv(
                        '/Users/d/work/Human/CartPole_OLD/data/control_sharing/data_' +  str(l) + '_'+ str(c) + '_'+ str(r) + '_' +  str(i)  + '_' + str(MAX_EPISODE) + '.csv', header=dict)

                agent.close()

