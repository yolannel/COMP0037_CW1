#!/usr/bin/env python3

'''
Created on 14 Jan 2022

NOTE: This code has been adapted from the week 1 lab of COMP 0037
'''

import matplotlib.pyplot as plt
import numpy as np

from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.epsilon_greedy_agent import EpsilonGreedyAgent
from bandits.damped_epsilon_greedy_agent import DampedEpsilonGreedyAgent
from bandits.performance_measures import compute_percentage_of_optimal_actions_selected
from bandits.performance_measures import compute_regret

if __name__ == '__main__':
       # Create bandit
    environment = BanditEnvironment(4)
    
    # Add some bandits
    environment.set_bandit(0, Bandit(0, 1))    
    environment.set_bandit(1, Bandit(0, 1))    
    environment.set_bandit(2, Bandit(0.1, 1))
    environment.set_bandit(3, Bandit(0, 1))
    
    number_of_steps = 100000
    # epsilon = 0.01
    epsilonList = [0.01,0.1,0.3,0.5]

    for i in range(len(epsilonList)):
        epsilon = epsilonList[i]

    # Step-by-step store of rewards
        reward_history = np.zeros(number_of_steps)
        action_history = np.zeros(number_of_steps)
        agent = DampedEpsilonGreedyAgent(environment, epsilon)

        # Step through the agent and let it do its business
        for p in range(0, number_of_steps):
            action_history[p], reward_history[p] = agent.step()
        
        # Plot percentage correct action curves
        percentage_correct_actions = compute_percentage_of_optimal_actions_selected(environment, action_history)
        plt.plot(percentage_correct_actions)
        plt.xlabel('Sample number')
        plt.ylabel('Percentage optimal action')
        plt.title('Percentage Optimal Pulls for varying values of epsilon - e-Greedy Strategy')
        plt.xlim(0,number_of_steps)
        plt.legend(['e=0.01','e=0.1','e=0.3','e=0.5'], prop={'size': 6})
    
        # Plot the regret curves
        # regret = compute_regret(environment, reward_history)
        # plt.title('Cumulative regret for varying values of epsilon - e-Greedy Strategy')
        # plt.plot(regret)
        # plt.xlabel('Sample number')
        # plt.ylabel('Cumulative regret')
        # #plt.ylim(0,16000)
        # plt.xlim(0,number_of_steps)
        # plt.grid()
        # plt.legend(['e=0.01','e=0.05','e=0.1','e=0.2'], prop={'size': 6})
    plt.grid()   
    plt.show()