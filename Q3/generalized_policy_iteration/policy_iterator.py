'''
Created on 29 Jan 2022

@author: ucacsjj
'''

# This class implements the policy iterator algorithm.

from cmath import nan
from .dynamic_programming_base import DynamicProgrammingBase

class PolicyIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the policy evaluation algorithm
        # will be run before the for loop is exited.
        self._max_policy_evaluation_steps_per_iteration = 100
        
        
        # The maximum number of times the policy evaluation iteration
        # is carried out.
        self._max_policy_iteration_steps = 1000

        # List storing evaluation iterations before improvement
        self._evaluation_steps_per_iteration = []
        

    def solve_policy(self):
                            
        # Initialize the drawers if defined
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()

        # Reset termination indicators       
        policy_iteration_step = 0        
        policy_stable = False
        
        # Loop until either the policy converges or we ran out of steps        
        while (policy_stable is False) and \
            (policy_iteration_step < self._max_policy_iteration_steps):
            
            # Evaluate the policy
            self._evaluate_policy()

            # Improve the policy            
            policy_stable = self._improve_policy()
            
            # Update the drawers if needed
            if self._policy_drawer is not None:
                self._policy_drawer.update()
                
            if self._value_drawer is not None:
                self._value_drawer.update()
                
            policy_iteration_step += 1

            # print(f'Finished policy iteration {policy_iteration_step}')

        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()

        # Return the value function and policy of the solution
        return self._v, self._pi, self._evaluation_steps_per_iteration, policy_iteration_step

        
    def _evaluate_policy(self):
        
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
                    
                    # Get the previous value function
                    old_v = self._v.value(x, y)

                    # Compute p(s',r|s,a)
                    s_prime, r, p = environment.next_state_and_reward_distribution(cell, \
                                                                                     self._pi.action(x, y))
                    
                    # Sum over the rewards
                    new_v = 0
                    for t in range(len(p)):
                        sc = s_prime[t].coords()
                        new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))                        
                        
                    # Set the new value in the value function
                    self._v.set_value(x, y, new_v)
                                        
                    # Update the maximum deviation
                    delta = max(delta, abs(old_v-new_v))
 
            # Increment the policy evaluation counter        
            iteration += 1
                       
            # print(f'Finished policy evaluation iteration {iteration}')
            
            # Terminate the loop if either the change was very small, or we exceeded
            # the maximum number of iterations.
            if (delta < self._theta) or (iteration >= self._max_policy_evaluation_steps_per_iteration):
                self._evaluation_steps_per_iteration.append(iteration)
                break

    def _improve_policy(self):

        # Q3d: Finish implementing policy iteration

        # Get environment and map
        environment = self._environment
        map = environment.map()

        # Assume policy is stable
        policy_stable = True

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
                            # print(x,y,current_v,'changed')
                            current_v = possible_v

                    except IndexError:
                        # If index error means on boundary, ignore and continue
                        pass

                # Choose the nearby cell with the highest v

        return True if policy_stable else False
                
                
                
            
