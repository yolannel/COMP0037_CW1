'''
Created on 29 Jan 2022

@author: ucacsjj
'''

from .dynamic_programming_base import DynamicProgrammingBase

# This class ipmlements the value iteration algorithm

class ValueIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the value iteration
        # algorithm is carried out is carried out.
        self._max_optimal_value_function_iterations = 2000
   
    # Method to change the maximum number of iterations
    def set_max_optimal_value_function_iterations(self, max_optimal_value_function_iterations):
        self._max_optimal_value_function_iterations = max_optimal_value_function_iterations

    #    
    def solve_policy(self):

        # Initialize the drawers
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        self._compute_optimal_value_function()
 
        self._extract_policy()
        
        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        return self._v, self._pi

    # Q3h: Finish implementation of the value iterator
    
    def _compute_optimal_value_function(self):
        # Get the environment and map
        environment = self._environment
        map = environment.map()
        
        # Execute the loop at least once
        
        iteration = 0
        
        while True:
            
            delta = 0
            # Sweep systematically over all the states            
            for x in range(map.width()):
                for y in range(map.height()):
                    
                    # We skip obstructions and terminals. If a cell is obstructed,
                    # there's no action the robot can take to access it, so it doesn't
                    # count. If the cell is terminal, it executes the terminal action
                    # state. The value of the value of the terminal cell is the reward.
                    # The reward itself was set up as part of the initial conditions for the
                    # value function.
                    if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                        continue
                                       
                    # Unfortunately the need to use coordinates is a bit inefficient, due
                    # to legacy code
                    cell = (x, y)
                    
                    # Set up a max holder
                    max_v = 0

                    # Get the previous value function
                    old_v = self._v.value(x, y)

                    for i, movement in enumerate(environment._driving_deltas):
                        try:
                            # Compute p(s',r|s,a)
                            s_prime, r, p = environment.next_state_and_reward_distribution(cell, \
                                                                            i, True)
                            new_v = 0
                            for t in range(len(p)):
                                sc = s_prime[t].coords()
                                new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1])) 
                            
                            # print("old: " + str(old_v) + " new: " + str(new_v))
                            max_v = max(max_v,new_v)

                        except IndexError:
                            # If index error means on boundary, ignore and continue
                            pass
                    
                    new_v = max_v
                    # Set the new value in the value function
                    # print(x,y,new_v)
                    self._v.set_value(x, y, new_v)
                    
                    # Update the maximum deviation
                    delta = max(delta, abs(old_v-new_v))
                    # Sum over the rewards

            # Increment the policy evaluation counter        
            iteration += 1

            # Terminate the loop if either the change was very small, or we exceeded
            # the maximum number of iterations.
            if (delta < self._theta) or (iteration >= self._max_optimal_value_function_iterations):
                # self._steps_per_iteration.append(iteration)
                print("iterations: " + str(iteration))
                break


    def _extract_policy(self):
        # Get environment and map
        environment = self._environment
        map = environment.map()

        # Assume policy is stable
        # policy_stable = True

        # For every state go through each cell and choose the neighbouring cell with the highest score
        for x in range(map.width()):
            for y in range(map.height()):
                # Skip irrelevant cells
                if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                        continue

                # Current best action
                try:
                    current_a = self._pi.action(x,y)
                    # print(current_a.value)
                    # print(environment._driving_deltas[current_a.value])
                    current_arrow = environment._driving_deltas[current_a.value]
                    # print(current_arrow)
                    # Check if pointing at obstruction and stop loopback
                    if map.cell(x+current_arrow[0], y+current_arrow[1]).is_obstruction() or x+current_arrow[0] < 0 or y+current_arrow[1] < 0:
                        current_v = -10000000
                    else:
                        current_v = self._v.value(x+current_arrow[0], y+current_arrow[1])
                except IndexError:
                    current_v = -10000000

                for i, movement in enumerate(environment._driving_deltas):
                    try:
                        # Possible v values
                        if x+movement[0] >= 0 and y+movement[1] >= 0:
                            possible_v = self._v.value(x+movement[0], y+movement[1])
                            # print(x,y,current_v,possible_v)

                        # Compare possible v values, write to cell if better
                        if possible_v > current_v:
                            # If changes have to be made policy not stable
                            policy_stable = False
                            self._pi.set_action(x,y,i)
                    except IndexError:
                        # If index error means on boundary, ignore and continue
                        pass
