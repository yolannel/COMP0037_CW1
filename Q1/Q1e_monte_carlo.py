#!/usr/bin/env python3
#%%
'''
Created on 24 Jan 2022

NOTE: This code has been adapted from the week 1 lab of COMP 0037
'''

import numpy as np
import matplotlib.pyplot as plt
import gym
from bandits.epsilon_greedy_agent import EpsilonGreedyAgent

from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.random_action_agent import RandomActionAgent
from bandits.try_them_all_agent import TryThemAllAgent
from bandits.performance_measures import compute_percentage_of_optimal_actions_selected
from bandits.performance_measures import compute_regret

if __name__ == '__main__':
    environment = BanditEnvironment(4)
    
    environment.set_bandit(0, Bandit(4, 1))    
    environment.set_bandit(1, Bandit(4.1, 1))
    environment.set_bandit(2, Bandit(3.9, 1))
    environment.set_bandit(3, Bandit(4.2, 1))
    

    number_pulls_list = [1, 10, 100, 1000]
    number_of_steps = 100000
    for i in range(len(number_pulls_list)):

        number_pulls = number_pulls_list[i]
        agent = TryThemAllAgent(environment, number_pulls)

        reward_history = np.zeros(number_of_steps)
        action_history = np.zeros(number_of_steps)
        for p in range(0, number_of_steps):
            action_history[p], reward_history[p] = agent.step()
        
        
        # percentage_correct_actions = compute_percentage_of_optimal_actions_selected(environment, action_history)
        # plt.plot(percentage_correct_actions)
        # plt.title("Percentage optimal actions for diffent number of Monte Carlo runs")
        # plt.ylim(0,1.05)
        # plt.xlim(0,number_of_steps)
        # plt.xlabel('Sample number')
        # plt.ylabel('Percentage optimal action')
        # plt.legend(['Runs=1','Runs=10','Runs=100','Runs=1000'], prop={'size': 6})

        regret = compute_regret(environment, reward_history)
        plt.plot(regret)
        plt.title("Cumulative regret for diffent number of Monte Carlo runs")
        plt.xlabel("Number of steps")
        plt.ylabel("Cumulative regret")
        plt.xlim(0,number_of_steps)
        plt.ylim(0,15000)
        plt.legend(['Runs=1','Runs=10','Runs=100','Runs=1000'], prop={'size': 6})

    plt.grid()
    plt.show()

# %%
