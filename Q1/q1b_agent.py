#!/usr/bin/env python3
#%%
'''
Created on 24 Jan 2022

NOTE: This code has been adapted from the week 1 lab of COMP 0037
'''

import numpy as np
import matplotlib.pyplot as plt
import gym

from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.random_action_agent import RandomActionAgent

def run_bandits(environment, number_of_steps):
    
    action_and_reward = np.zeros([environment.number_of_bandits(),number_of_steps])
    for b in range(0, environment.number_of_bandits()):
        rewards = np.zeros(number_of_steps)
        for s in range(0, number_of_steps):
            obs, reward, done, info = environment.step(b)
            rewards[s] = reward
        action_and_reward[b] = rewards
    return action_and_reward
    
if __name__ == '__main__':
    # Create bandit
    environment = BanditEnvironment(4)
    
    # Add some bandits
    environment.set_bandit(0, Bandit(4, 1))    
    environment.set_bandit(1, Bandit(4.1, 1))
    environment.set_bandit(2, Bandit(3.9, 1))
    environment.set_bandit(3, Bandit(4.2, 1))
    
    #Use run bandit function to simulate
    runs = 4000
    results = run_bandits(environment, runs)
    results = results.reshape(runs, 4)
    
    #Create violin plot figure, axes and labels
    agent_plt = plt.figure()
    ax = agent_plt.add_axes([0,0,1,1])
    ax.violinplot(results)
    plt.xlabel('Charging Location')
    plt.ylabel('Mean Charging Rate (Amps)')
    plt.title('Violin Plot of Robot Charging Data')
    plt.grid(True)
# %%
