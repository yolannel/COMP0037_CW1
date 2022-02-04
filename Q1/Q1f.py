#!/usr/bin/env python3
#%%
'''
Created on 24 Jan 2022

NOTE: This code has been adapted from the week 1 lab of COMP 0037
'''

import numpy as np
import matplotlib.pyplot as plt

from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.damped_epsilon_greedy_agent import DampedEpsilonGreedyAgent
from bandits.performance_measures import compute_percentage_of_optimal_actions_selected
from bandits.performance_measures import compute_regret

if __name__ == '__main__':
    environment = BanditEnvironment(4)
    
    environment.set_bandit(0, Bandit(4, 1))    
    environment.set_bandit(1, Bandit(4.1, 1))
    environment.set_bandit(2, Bandit(3.9, 1))
    environment.set_bandit(3, Bandit(4.2, 1))
    
    epsilon = 0.05
    agent = DampedEpsilonGreedyAgent(environment,epsilon)
    runs = 4000
    reward_history = np.zeros(runs)
    action_history = np.zeros(runs)
    for p in range(0, runs):
        action_history[p], reward_history[p] = agent.step()
    
    percentage_correct_actions = compute_percentage_of_optimal_actions_selected(environment, action_history)
    plt.ion()
    plt.plot(percentage_correct_actions, color = 'red', label = '% optimal actions')
    plt.legend()
    plt.grid()
    plt.xlabel('Sample number')
    plt.ylabel('Percentage optimal action')

    # Plot the regret curves
    regret = compute_regret(environment, reward_history)
    plt.figure()
    plt.plot(regret, color = 'red', label = 'Regret')
    plt.legend()
    plt.grid()
    plt.xlabel('Sample number')
    plt.ylabel('Regret')
    
    plt.show()
    plt.pause(0.001)
    input()
