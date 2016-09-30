"""
Solve OpenAI Gym Cartpole V1 with DQN.
"""
import gym

import numpy as np
import tensorflow as tf
import math


#Hyperparameters
envSize = 4
H = 100 #number of neurons in hidden layer
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

    observations = tf.placeholder(tf.float32, [None, envSize] , name="input_x")
    w1 = tf.get_variable("w1", shape=[envSize, H],
                         initializer=tf.contrib.layers.xavier_initializer())
    hidden_layer_1 = tf.nn.relu(tf.matmul(observations, w1))
    w2 = tf.get_variable("w2", shape=[H, 1],
                         initializer=tf.contrib.layers.xavier_initializer())
    result_score = tf.matmul(hidden_layer_1, w2)
    probablility = tf.nn.sigmoid(result_score)

    training_variables = tf.trainable_variables()
    input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
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
    
    #D=[]
    #explore = 1.0

    max_episodes = 2000
    max_steps = 500


    #..........................
    # Q-Learner Setup
    #w1 = tf.random_uniform([env.observation_space.shape[0], 200], -1.0, 1.0)
    #w1 = tf.Variable(w1)
    #b1 = tf.random_uniform([200], -1.0, 1.0)
    #b1 = tf.Variable(b1)

    #w2 = tf.random_uniform([200, env.action_space.n], -1.0, 1.0)
    #w2 = tf.Variable(w2)
    #b2 = tf.random_uniform([env.action_space.n], -1.0, 1.0)
    #b2 = tf.variable(b2)

    # Connecting the neurons together
    #prev_states = tf.placeholder(tf.float32, [None, self._dim_state])
    #hidden_1 = tf.nn.relu(tf.matmul(prev_states, w1) + b1)
    #action_values = tf.nn.softmax(tf.matmul(hidden_1, w2) + b2)
    

    #Setup Q
    
    #Setup Qprime

    xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 1

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        #setting up the training variables
        gradBuffer = sess.run(training_variables)
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
    
        for episode in xrange(max_episodes):
            observation = env.reset()
            for step in xrange(max_steps):
                if(step == (max_steps-1)):
                    print 'Made 500 steps!'
                env.render()
                x = np.reshape(observation,[1,envSize])

                #get action from policy
                tfprob = sess.run(probablility,feed_dict={observations: x})
                action = 1 if np.random.uniform() < tfprob else 0
                #will need to rework action to be more generic, not just 1 or 0
            
                xs.append(x) # observation
                y = 1 if action == 0 else 0 # something about fake lables, need to investigate
                ys.append(y)

                #run an action
                observation, reward, done, info = env.step(action)
                reward_sum += reward

                drs.append(reward)

                if done:
                    episode_number +=1
                    print 'Episode %f: Reward: %f'  %(episode_number, reward_sum)
                    #putting together all inputs, is there a better way to do this?
                    epx = np.vstack(xs)
                    epy = np.vstack(ys)
                    epr = np.vstack(drs)
                    tfp = tfps
                    xs,hs,dlogpr,drs,ys,tfps = [],[],[],[],[],[] #reset for next episode

                    #compute reward
                    discounted_epr = reduced_rewards(epr)
                    discounted_epr -= np.mean(discounted_epr)
                    discounted_epr /= np.std(discounted_epr)

                    #get gradient, save in gradent_buffer
                    tGrad = sess.run(new_gradients,feed_dict={observations: epx, input_y: epy, advantage: discounted_epr})
                    for ix,grad in enumerate(tGrad):
                        gradBuffer[ix] += grad

                    if episode_number % batch_number == 0:
                        sess.run(update_gradent,feed_dict={w1_gradent: gradBuffer[0],w2_gradent: gradBuffer[1]})
                        for ix,grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0

                        running_reward = reward_sum if running_reward is None else (((running_reward * episode_number - 50) + (reward_sum * 50))/episode_number)
                        print 'Average reward for episode %f. total average reward %f' %(reward_sum/batch_number, running_reward/batch_number)

                        if reward_sum/batch_number > 475:
                                print 'Task solved in', episode_number, 'episodes!'
                                break
                        reward_sum = 0
                    break
                
    env.monitor.close()
