#!/usr/bin/env python3

'''
Created on 26 Jan 2022

@author: ucacsjj
'''

# This script is needed for Q2i-l

from airport.scenarios import *
from airport.airport_environment import AirportBatteryChargingEnvironment
from airport.actions import ActionType
from airport.charging_policy import ChargingPolicy
from airport.charging_policy_drawer import ChargingPolicyDrawer

if __name__ == '__main__':
    # Get the map
    airport_map = full_scenario()
    
    charging_policy = ChargingPolicy(airport_map, set_random = False)
    
    # Create the gym environment
    airport_environment = AirportBatteryChargingEnvironment(airport_map)

    # Q2j, k:
    # Implement your algorithm here to use the airport_environment
    # to work out the optimal. Modify the heuristic of the planner and run again.

    # IDEA: For each valid state, calculate q_pi and return the action with the best return, repeat for each valid state.
    #       then save to charging policy.

    for x in range(airport_map.width()):
        for y in range(airport_map.height()):
            # Skip if cell is obstructed
            if not airport_map.is_obstructed(x,y):
                print("Cell Coords:" + str(x) + ',' +str(y))
                best_reward = -float('inf')
                # Check cost of each charging station and return optimal action
                for count, charging_station in enumerate(airport_map.all_charging_stations()):
                    # Teleport robot to start position
                    action = (ActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (x, y))
                    observation, reward, done, info = airport_environment.step(action)
                    action = (ActionType.DRIVE_ROBOT_TO_NEW_POSITION, charging_station.coords())
                    observation, reward, done, info = airport_environment.step(action)
                    action_func = charging_station.params()[0] + reward
                    print(action_func)
                    # Higher action_func is better
                    if action_func > best_reward:
                        charging_policy.set_action(x,y,count)
                        best_reward = action_func
                        print("Best Reward:" + str(best_reward))
            
    
    # IDEA: For each valid state, calculate q_pi and return the action with the best return, repeat for each valid state.
    #       then save to charging policy.

    # Extract cell type from airport_map
    # If valid: Use A* to calculate distance to each cell
    # Compare and return best charging point

    for i in range(len(airport_map.all_charging_stations())):
        print(i)
        print(airport_map.charging_station(i).params())



    # Plot the resulting policy
    charging_policy_drawer = ChargingPolicyDrawer(charging_policy, 200)
    charging_policy_drawer.update()
    #charging_policy_drawer.wait_for_key_press()
    
    try:
        input("Press enter in the command window to continue.....")
    except SyntaxError:
        pass
   