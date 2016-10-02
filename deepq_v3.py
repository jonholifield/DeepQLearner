# Deep Q network

import gym
import numpy as np
import tensorflow as tf
import math
import random

# HYPERPARMETERS
H = 80
H2 = 40
batch_number = 50
gamma = 0.99
explore = 1
length_of_slope = 10.0


# idea here is the last 10 will have less than 1
def future_reward_gen(instant_reward):
	future_included_reward = np.zeros_like(instant_reward)
	for t in reversed(xrange(0, instant_reward.size)):
		if t == instant_reward.size-1:
			future_included_reward[t] = 0;
		else:
			future_included_reward[t] = future_included_reward[t+1] + (instant_reward[t]/10)
			if (future_included_reward[t] > 1):
				future_included_reward[t] = 1;
	return future_included_reward
	
def train_batch(D):
    samples = D.sample(50)
    
    
if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    env.monitor.start('training_dir', force=True)
    #Setup tensorflow
    
    tf.reset_default_graph()

    #First Q Network
    w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0],H], -1.0, 1.0))
    bias1 = tf.Variable(tf.random_uniform([H], -1.0, 1.0))
    
    w2 = tf.Variable(tf.random_uniform([H,H2], -1.0, 1.0))
    bias2 = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))
    
    w3 = tf.Variable(tf.random_uniform([H2,env.action_space.n], -1.0, 1.0))
    bias3 = tf.Variable(tf.random_uniform([env.action_space.n], -1.0, 1.0))
    
    states = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="states")  # This is the list of matrixes that hold all observations
    actions = tf.placeholder(tf.float32, [None, env.action_space.n], name="actions")
    
    hidden_1 = tf.nn.relu(tf.matmul(states, w1) + bias1)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + bias2)
    action_values = tf.nn.softmax(tf.matmul(hidden_2, w3) + bias3)
    
    Q = tf.reduce_sum(tf.mul(action_values, actions), reduction_indices=1) 
    
    #previous_action_masks = tf.placeholder(tf.float32, [None, env.action_space.n], name="p_a_m") # This holds all actions taken 
    #previous_values = tf.reduce_sum(tf.mul(previous_action_values, previous_action_masks), reduction_indices=1) #Combination of action taken and resulting q
    
    #Is there a better way to do this?
    #w1_prime = tf.Variable(tf.random_uniform([env.observation_space.shape[0],H], -1.0, 1.0))
    #bias1_prime = tf.Variable(tf.random_uniform([H], -1.0, 1.0))
    
    #w2_prime = tf.Variable(tf.random_uniform([H,H2], -1.0, 1.0))
    #bias2_prime = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))
    
    #w3_prime = tf.Variable(tf.random_uniform([H2,env.action_space.n], -1.0, 1.0))
    #bias3_prime = tf.Variable(tf.random_uniform([env.action_space.n], -1.0, 1.0))
    
    #w1_prime= w1_prime.assign(w1)
    #bias1_prime= bias1_prime.assign(bias1)
    #w2_prime= w2_prime.assign(w2)
    #bias2_prime= bias2_prime.assign(bias2)
    #w3_prime= w3_prime.assign(w3)
    #bias3_prime= bias3_prime.assign(bias3)
    
    #Second Q network
    #rewards = tf.placeholder(tf.float32, [None, 1], name="rewards") # This holds all the rewards that are real
    #next_states = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="n_s") # This is the list of matrixes that hold all observations
    #hidden_1_prime = tf.nn.relu(tf.matmul(next_states, w1_prime) + bias1_prime)
    #hidden_2_prime = tf.nn.relu(tf.matmul(hidden_1_prime, w2_prime) + bias2_prime)
    #next_action_values =  tf.matmul(hidden_2_prime, w3_prime) + bias3_prime
    #next_values = prev_rewards + gamma * tf.reduce_max(next_action_values, reduction_indices=1)   
    
    #Q_prime = rewards + gamma * tf.reduce_max(next_action_values, reduction_indices=1)
    corrected_actions =  actions = tf.placeholder(tf.float32, [None, env.action_space.n], name="Corrected_actions")
    
    loss = (corrected_actions - action_values)**2 #* actions , Note, only send in one at a time to train
    
    #tp = tf.transpose(actions)
    #log_prob = tf.log(tf.diag_part(tf.matmul(action_values, tp)))
    #log_prob = tf.reshape(log_prob, (1,-1))
    #loss = tf.matmul(log_prob, rewards)
    #loss = -tf.reshape(loss, [-1])
    
    train = tf.train.AdamOptimizer(.01).minimize(loss)
    
    #Setting up the enviroment
    
    max_episodes = 2000
    max_steps = 600

    D = []
    rewardList = []
    past_actions = []
    
    episode_number = 0
    episode_reward = 0
    reward_sum = 0
    
    init = tf.initialize_all_variables()
   
    with tf.Session() as sess:
        sess.run(init)
    
        for episode in xrange(max_episodes):
            observation = env.reset()
            #D.append(observation) # observation
            #rewardList.append(1.0)
            
            for step in xrange(max_steps):
                
                if(step == (max_steps-1)):
                    print 'Made 600 steps!'
                
                if episode_number % batch_number == 0:
                    env.render()
                x = observation #np.reshape(observation,[1,envSize])


                if explore > random.random():
                    action = env.action_space.sample()
                    if(action == 1):
                        past_actions.append([0,1])
                    else:
                        past_actions.append([1,0])
                else:
                    #get action from policy
                    results = sess.run(action_values, feed_dict={states: np.array([observation])})
                    #print results
                    action = (np.argmax(results))
                    #print action
                    if(action == 1):
                        past_actions.append([0,1])
                    else:
                        past_actions.append([1,0])
                #print 'action is', action
                #will need to rework action to be more generic, not just 1 or 0
            
                
                #y = 1 if action == 0 else 0 # something about fake lables, need to investigate
                #.append(y)
                
                #run an action
                observation, reward, done, info = env.step(action)
                
                D.append(observation)
                rewardList.append(reward)
                episode_reward += reward
                if (len(D) > 50):
                    D.pop(0)
                    rewardList.pop(0)
                    past_actions.pop(0)

                if done:
                    episode_number +=1
                    #print 'Episode %f: Reward: %f'  %(episode_number, episode_reward)
                    reward_sum += episode_reward;
                    episode_reward = 0;

                    #if episode_number % 150 == 0: # copy q to q_prime 
                    #    w1_prime= tf.identity(w1)
                    #    bias1_prime= tf.identity(bias1)
                    #    w2_prime= tf.identity(w2)
                    #    bias2_prime= tf.identity(bias2)
                    #    w3_prime= tf.identity(w3)
                    #    bias3_prime= tf.identity(bias3)
                    if 0 == 0: #episode_number % batch_number == 0:
                        #compute modified rewards
                        rewardListMod = future_reward_gen(np.vstack(rewardList))
                        #print rewardListMod
                        rewardList = []
                        #print rewardListMod
                        #print 'states:'
                        #print D[1]
                        #print 'rewards:'
                        #print rewardListMod
                        #print 'actions:'
                        #print len(past_actions)
                        for i in xrange(20): # take a sample of action/rewards to train with
                            if step < 50:
                                stepSelected = random.randint(0,step)
                            else:
                                stepSelected = random.randint(0,49)
                            
                            #if reward > .8 then keep action as correct action, else flip each one
                            if rewardListMod[stepSelected] > .6:
                                correct_action = list(past_actions[stepSelected])
                            else:
                                correct_action = list(past_actions[stepSelected])
                                for x in xrange(len(correct_action)):
                                    #print 'correct_action ', correct_action[x] 
                                    if correct_action[x] == 1:
                                        correct_action[x] = 0
                                    else:
                                        correct_action[x] = 1
                                        
                            #print [D[stepSelected]]
                            #print [rewardListMod[stepSelected]]
                            #print [past_actions[stepSelected]]
                            #print [correct_action]
                            #print (rewardListMod[stepSelected] - past_actions[stepSelected])**2
                            sess.run(train, feed_dict={states: [D[stepSelected]],corrected_actions: [correct_action],actions: [past_actions[stepSelected]] }) # next_states: D
                        if episode_number % batch_number == 0:
                            print 'Average reward for episode %f is %f.' %(episode_number,reward_sum/batch_number)
                            if reward_sum/batch_number > 475:
                                print 'Task solved in', episode_number, 'episodes!'
                                reward_sum = 0
                                break
                            reward_sum = 0
                            print 'explore is ', explore
                        explore = explore * .996
                        D = []
                        rerwardListMod = []
                        past_actions = []
                        
                    break
                
    env.monitor.close()
    
    
    
    
    
    