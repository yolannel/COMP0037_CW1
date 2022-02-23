#!/usr/bin/env python3

"""
Created on 26 Jan 2022

@author: ucacsjj
"""

# This script is needed for Q2i-l

from airport.scenarios import *
from airport.airport_environment import AirportBatteryChargingEnvironment
from airport.actions import ActionType
from airport.charging_policy import ChargingPolicy
from airport.charging_policy_drawer import ChargingPolicyDrawer

if __name__ == '__main__':
    # Get the map
    airport_map = full_scenario()
    
    charging_policy = ChargingPolicy(airport_map, set_random=True)
    
    # Create the environment
    airport_environment = AirportBatteryChargingEnvironment(airport_map)

    # Q2j, k:
    # Implement your algorithm here to use the airport_environment
    # to work out the optimal. Modify the heuristic of the planner and run again.
    
    # Plot the resulting policy
    charging_policy_drawer = ChargingPolicyDrawer(charging_policy, 200)
    charging_policy_drawer.update()
    #charging_policy_drawer.wait_for_key_press()
    
    try:
        input("Press enter in the command window to continue.....")
    except SyntaxError:
        pass
   