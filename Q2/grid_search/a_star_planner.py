'''
Created on 2 Jan 2022

@author: ucacsjj
'''

import math

from .dijkstra_planner import DijkstraPlanner

# This class implements the A* search algorithm

class AStarPlanner(DijkstraPlanner):
    
    def __init__(self, occupancyGrid):
        DijkstraPlanner.__init__(self, occupancyGrid)

    # Q2h:
    # Complete implementation of A*.
    def push_cell_onto_queue(self, cell):
        cellCoords = cell.coords()
        goalCoords = self.goal.coords()
        
        # Estimate Cost to Come
        dXp = cellCoords[0] - goalCoords[0]
        dYp = cellCoords[1] - goalCoords[1]

        # Euclidean Heuristic
        estCost = math.sqrt(dXp * dXp + dYp * dYp)

        # Squared Euclidean Heuristic (INADMISSIBLE)
        # estCost = dXp * dXp + dYp * dYp

        if (cell.parent is None):
            cell.path_cost = 0
        else:
            # Calculate current path cost using parent and penalty
            parentCoords = cell.parent.coords()
            dp = self._environment_map.compute_transition_cost(parentCoords, cellCoords)
            cell.path_cost = cell.parent.path_cost + dp

        self.priorityQueue.put((cell.path_cost + estCost, cell))
