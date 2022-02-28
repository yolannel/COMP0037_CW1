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
from bandits.performance_measures import compute_percentage_of_optimal_actions_selected
from bandits.performance_measures import compute_regret

if __name__ == '__main__':
    # Create bandit
    environment = BanditEnvironment(4)
    
    # Add some bandits
    environment.set_bandit(0, Bandit(4, 1))    
    environment.set_bandit(1, Bandit(4.1, 1))
    environment.set_bandit(2, Bandit(3.9, 1))
    environment.set_bandit(3, Bandit(4.2, 1))
    
    number_of_steps = 100000
    epsilonlist = [0.01, 0.1, 0.2, 0.5, 1]
    y_label = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    for i in range(len(epsilonlist)):
        epsilon = epsilonlist[i]
        agent = EpsilonGreedyAgent(environment, epsilon)
        
        # Step-by-step store of rewards
        reward_history = np.zeros(number_of_steps)
        action_history = np.zeros(number_of_steps)
        
        # Step through the agent and let it do its business
        for p in range(0, number_of_steps):
            action_history[p], reward_history[p] = agent.step()
        
        
    ##### UNCOMMENT FOR DIFFERENT PLOTS #####

        # Plot percentage correct action curves
        percentage_correct_actions = compute_percentage_of_optimal_actions_selected(environment, action_history)
        plt.plot(percentage_correct_actions)
        plt.xlabel('Sample number')
        plt.ylabel('Percentage optimal action')
        plt.title('Percentage Optimal Pulls for varying values of epsilon - e-Greedy Strategy')
        plt.grid()
        plt.ylim(0,1.05)
        plt.xlim(0,number_of_steps)
        plt.yticks(y_label)
        #plt.axhline(y=longtermave, color='y', linestyle='--')
        plt.legend(['e=0.01','e=0.1','e=0.2','e=0.5','e=1'], prop={'size': 6})

        # # Plot the regret curves
        # regret = compute_regret(environment, reward_history)
        # plt.title('Cumulative regret for varying values of epsilon - e-Greedy Strategy')
        # plt.plot(regret)
        # plt.xlabel('Sample number')
        # plt.ylabel('Cumulative regret')
        # plt.ylim(0,16000)
        # plt.xlim(0,number_of_steps)
        # plt.grid()
        # plt.legend(['e=0.01','e=0.1','e=0.2','e=0.5','e=1'], prop={'size': 6})

        longtermave = 1 + (1/4-1)*epsilon
        print(f"Long term average probability when e={epsilon}:", longtermave)
    
    plt.show()