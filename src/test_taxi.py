import numpy as np
import gym
# import pygame
import random

# create Taxi environment
env = gym.make('Taxi-v3')
env.reset()
env.step(1)
env.render(mode='human')

# get the basic size fo the action/state space
state_size = env.observation_space.n
action_size = env.action_space.n

# testing variables
num_episodes = 2000
max_steps = 100  # per episode

# load in teh saved qtable from the training
qtable = np.load("../data/taxi_qtable.npy")

q_est = np.zeros((state_size, action_size))

# Challenge:
#  1. we don't know the state numbering
#  2. we do know the actions, but not the correct order
#

# watch trained agent
for idx in range(num_episodes):

    state = env.reset()
    done = False
    rewards = 0

    print("Episode {}".format(idx))

    for s in range(max_steps):

        action = np.argmax(qtable[state,:])

        q_est[state, action] += 1

        new_state, reward, done, info = env.step(action)

        rewards += reward
        # env.render(mode='human')
        state = new_state

        if done == True:
            print("  Step {}".format(s+1))
            print("  Score: {}".format(rewards))
            break


env.close()

