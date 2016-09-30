"""
Solve OpenAI Gym Cartpole V1 with DQN.
"""
import gym

import numpy as np
import tensorflow as tf
   

#Hyperparameters
H = 50 #number of neurons in hidden layer
batch_number = 50 # size of batches for training
learn_rate = .01
gamma = 0.99


def reduced_rewards(r):
    reduced_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        reduced_r[t] = running_add
    return reduced_r

if __name__ == '__main__':

    
    env = gym.make('CartPole-v1')
    env.monitor.start('training_dir', force=True)
    #Setup tensorflow
    tf.reset_default_graph()

    observations = tf.placeholder(tf.float43, [None, env.observation_space.shape[0]] , name="input_x")
    w1 = tf.get_variable("w1", shape[env.observation_space.shape[0], H],
                         initializer=tf.contrib.layers.xavier_initializer())
    hidden_layer_1 = tf.nn.relu(tf.matmul(observations, w1))
    w2 = tf.get_variable("w2", shape[env.observation_space.shape[0], 1],
                         initializer=tf.contrib.layers.xavier_initializer())
    result_score = tf.matmul(hidden_layer_1, w2)
    probablility = tf.nn.sigmoid(result_score)

    training_variables = tf.trainable_variables()
    input_y = tf.placeholder(tf.float32, [None, 1], name "input_y")
    advantage = tf.placeholder(tf.float32,name="reward_signal")

    #Loss Function
    loss = -tf.reduce_mean((tf.log(input_y - probablility)) * advantage)

    new_gradients = tf.gradients(loss, training_variables)

    # Training

    adam = tf.train.AdamOptimizer(learning_rate=learn_rate)

    w1_gradent = tf.placeholder(tf.float32,name="batch_gradent1")

    w2_gradent = tf.placeholder(tf.float32,name="batch_gradent2")

    batch_gradent = [w1_gradent, w2_gradent]
    update_gradent = adam.apply_gradients(zip(batch_gradent, training_variables))
    
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
    action_values = tf.nn.softmax(tf.matmul(hidden_1, w2) + b2)
    

    #Setup Q
    
    #Setup Qprime
    
    for episode in xrange(max_episodes):
        observation = env.reset()
        for step in xrange(max_steps):
            env.render()
            #need to retrieve actions
            if (action_value[1] > action_value[2])
                action = 0
            else
                action = 1
            print action
            #Run the program for one step
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
