"""
Solve OpenAI Gym Cartpole V1 with DQN.
"""
import gym

import numpy as np
import tensorflow as tf
   

if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    env.monitor.start('training_dir', force=True)

    D=[]
    explore = 1.0

    max_episodes = 2000
    max_steps = 500


    #..........................
    # Q-Learner Setup
    w1 = tf.random_uniform([env.observation_space.shape[0], 200], -1.0, 1.0)
    w1 = tf.Variable(w1)
    b1 = tf.random_uniform([200], -1.0, 1.0)
    b1 = tf.Variable(b1)

    w2 = tf.random_uniform([200, env.action_space.n], -1.0, 1.0)
    w2 = tf.Variable(w2)
    b2 = tf.random_uniform([env.action_space.n], -1.0, 1.0)
    b2 = tf.variable(b2)

    # Connecting the neurons together
    prev_states = tf.placeholder(tf.float32, [None, self._dim_state])
    hidden_1 = tf.nn.relu(tf.matmul(prev_states, w1) + b1)
    prev_action_values = tf.squeeze(tf.matmul(hidden_1, w2) + b2)

    #Setup Q
    

    #Setup Qprime
    

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()

            action = DQL.take_action(); env.action_space.sample()
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
