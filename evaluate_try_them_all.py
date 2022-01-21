#!/usr/bin/env python3

'''
Created on 14 Jan 2022

@author: ucacsjj
'''

import matplotlib.pyplot as plt
import numpy as np

from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.try_them_all_agent import TryThemAllAgent
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
    
    number_of_steps = 10000
    
    number_of_tries = 5
    
    agent = TryThemAllAgent(environment, number_of_tries)
    
    # Step-by-step store of rewards
    reward_history = np.zeros(number_of_steps)
    action_history = np.zeros(number_of_steps)
    
    # Step through the agent and let it do its business
    for p in range(0, number_of_steps):
        action_history[p], reward_history[p] = agent.step()
        
    print(f'Mean reward={np.mean(reward_history)}')
    
    # Q4a:
    # Plot the % optimal actions
    percentage_correct_actions = compute_percentage_of_optimal_actions_selected(environment, action_history)
    
    # Q4c:
    # Plot the regret curves
    regret = compute_regret(environment, reward_history)
    