#!/usr/bin/env python3

'''
Created on 14 Jan 2022

NOTE: This code has been adapted from the week 1 lab of COMP 0037
'''

from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.upper_confidence_bound_agent import UpperConfidenceBoundAgent
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

    y_label = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    x_label = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0]
    plotmean_reward=[]

    ##Uncomment for all other plots
    #i = [0.1,0.5,1,2,3,4,5,10]


    ##Uncomment to plot total mean reward vs confidence level
    i = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.3,1.6,2,3]


    for c in i:
        agent = UpperConfidenceBoundAgent(environment, c)

        # Step-by-step store of rewards
        reward_history = np.zeros(number_of_steps)
        action_history = np.zeros(number_of_steps)
        
        # Step through the agent and let it do its business
        for p in range(0, number_of_steps):
            action_history[p], reward_history[p] = agent.step()
            mean_reward = np.mean(reward_history)
        plotmean_reward.append(mean_reward)

    # Plot UCB with different  degrees of exploration
        ### Plot percentage correct
        # percentage_correct_actions = compute_percentage_of_optimal_actions_selected(environment, action_history)
        # plt.plot(percentage_correct_actions)
        # plt.title("Performance of UCB for different degrees of exploration")
        # plt.xlabel("Number of steps")
        # plt.ylabel("Percentage correct action")
        # plt.legend(['c=0.1','c=0.2','c=0.3','c=0.4','c=0.5','c=0.6','c=0.7','c=0.8','c=0.9','c=1'], prop={'size': 6})
        # plt.xlim(0,number_of_steps)
        # plt.ylim(0,1.05)
        # plt.yticks(y_label)

        ### Plot cumulative regret
        # regret = compute_regret(environment, reward_history)
        # plt.plot(regret)
        # plt.title("Cumulative regret of UCB for different degrees of exploration")
        # plt.xlabel("Number of steps")
        # plt.legend(['c=0.1','c=0.5','c=1','c=2','c=3','c=4','c=5','c=10'], prop={'size': 6})
        # plt.ylabel("Regret")
        # plt.xlim(0,number_of_steps)
        # #plt.ylim(0,8000)


        ### Plot history of arms pulled 
        # #change to plot : number of pulls vs arm
        # y_tick=[0,1,2,3]
        # plt.plot(action_history)
        # plt.title("History of charging stations used for different degrees of exploration")
        # plt.xlabel("Number of steps")
        # plt.yticks(y_tick)
        # plt.legend(['c=0.1','c=0.5','c=1','c=2','c=3','c=4','c=5','c=10'], prop={'size': 6})
        # plt.ylabel("Charging Station")

    plt.plot(i,plotmean_reward,'o-')
    plt.title("Total Mean Reward vs Confidence Level")
    plt.xlabel("Confidence Level")
    plt.ylabel("Total Mean Reward")
    plt.xlim(0,2)
    plt.xticks(x_label)

    plt.grid()
    plt.show()
    