#!/usr/bin/env python3

'''
Created on 27 Jan 2022

@author: ucacsjj
'''

from airport.scenarios import *
from airport.airport_environment import PlannerType
from airport.airport_environment import AirportBatteryChargingEnvironment
from airport.actions import ActionType

if __name__ == '__main__':
    
    # Create the scenario
    airport_map = full_scenario()
    
    # Create the gym environment
    airport_environment = AirportBatteryChargingEnvironment(airport_map, PlannerType.BREADTH_FIRST)
    
    # Set the graphics debugging to full
    airport_environment.enable_verbose_graphics(True)
    
    # First specify the start location of the robot
    action = (ActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (0, 0))
    observation, reward, done, info = airport_environment.step(action)
    
    if reward is -float('inf'):
        print('Unable to teleport to (1, 1)')
        
    # Get all the rubbish bins and toilets; these are places which need cleaning
    all_rubbish_bins = airport_map.all_rubbish_bins()
        
    # Q2b
    # Modify this code to collect the data needed to assess the different algorithms
    
    # Now go through them and plan a path sequentially
    for rubbish_bin in all_rubbish_bins:
            action = (ActionType.DRIVE_ROBOT_TO_NEW_POSITION, rubbish_bin.coords())
            observation, reward, done, info = airport_environment.step(action)
    
    try:
        input("Press enter in the command window to continue.....")
    except SyntaxError:
        pass  
    