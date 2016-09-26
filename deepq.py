"""
Solve OpenAI Gym Cartpole V1 with DQN.
"""
import gym

import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    action = 1
    env = gym.make('CartPole-v1')
    env.monitor.start('training_dir', force=True)

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            # action is either 1 or -1
            if action == 1:
               action = 0
            else:
               action = 1
            #action = env.action_space.sample()
            print action
            observation, reward, done, info = env.step(action)
            print("OBSERVED")
            print(observation)
            print("REWARD")
            print(reward)
            print("DONE")
            print(done)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.monitor.close()
