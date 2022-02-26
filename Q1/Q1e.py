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
from bandits.performance_measures import compute_percentage_of_optimal_actions_selected

if __name__ == '__main__':
    environment = BanditEnvironment(4)
    
    environment.set_bandit(0, Bandit(4, 1))    
    environment.set_bandit(1, Bandit(4.1, 1))
    environment.set_bandit(2, Bandit(3.9, 1))
    environment.set_bandit(3, Bandit(4.2, 1))
    
    agent = RandomActionAgent(environment)
    runs = 4000
    reward_history = np.zeros(runs)
    action_history = np.zeros(runs)
    for p in range(0, runs):
        action_history[p], reward_history[p] = agent.step()
    
    percentage_correct_actions = compute_percentage_of_optimal_actions_selected(environment, action_history)
    # #Create violin plot figure, axes and labels
    # agent_plt,ax = plt.subplots()
    # # ax = agent_plt.add_axes([0,0,1,1])
    # ax.violinplot(reward_history)
    # plt.xlabel('Charging Location')
    # plt.ylabel('Mean Charging Rate (Amps)')
    # plt.title('Violin Plot of Robot Charging Data')
    # plt.grid(True)
    # agent_plt.savefig('violin1e.jpg')
    plt.figure()
    plt.plot(percentage_correct_actions, color = 'red', label = '% optimal actions')
    plt.legend()
    plt.xlabel('Sample number')
    plt.ylabel('Percentage optimal action')
    plt.grid(True)
    plt.show()
# %%
