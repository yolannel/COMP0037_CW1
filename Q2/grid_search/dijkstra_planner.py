"""
Created on 2 Jan 2022

@author: ucacsjj
"""

from math import sqrt
from queue import PriorityQueue

from .planner_base import PlannerBase


class DijkstraPlanner(PlannerBase):

    # This implements Dijkstra. The priority queue is the path length
    # to the current position.
    
    def __init__(self, occupancyGrid):
        PlannerBase.__init__(self, occupancyGrid)
        self.priorityQueue = PriorityQueue()

    def push_cell_onto_queue(self, cell):
        cellCoords = cell.coords()
        # startCoords = self.start.coords()

        # Also init. path cost to cell
        # Distance from startCoords to current coords
        # startCoords cost is 0

        if (cell.parent is None):
            cell.path_cost = 0
        else:
            # Calculate current path cost using parent
            parentCoords = cell.parent.coords()
            # # Without pentalty
            # dXp = cellCoords[0] - parentCoords[0]
            # dYp = cellCoords[1] - parentCoords[1]
            # dp = sqrt(dXp * dXp + dYp * dYp)
            # Evaluate cell type to determin cost multiplier
            dp = self._environment_map.compute_transition_cost(parentCoords, cellCoords)
            # print(self._environment_map.compute_transition_cost(parentCoords, cellCoords))
            cell.path_cost = cell.parent.path_cost + dp

        # Calculate distance from start for prio. q
        # dX = cellCoords[0] - startCoords[0]
        # dY = cellCoords[1] - startCoords[1]
        # d = sqrt(dX * dX + dY * dY)
        self.priorityQueue.put((cell.path_cost, cell))

    # Check the queue size is zero
    def is_queue_empty(self):
        return self.priorityQueue.empty()

    # Simply pull from the front of the list
    def pop_cell_from_queue(self):
        cell = self.priorityQueue.get()
        return cell[1]

    def resolve_duplicate(self, cell, parent_cell):
        cellCoords = cell.coords()
        cellPathCost = cell.path_cost
        parentCoords = parent_cell.coords()
        parentPathCost = parent_cell.path_cost

        # Extract current path cost from each cell
        cellPathCost = cell.path_cost
        # Compare path cost of moving from parent to cell
            # NewCost: Parent Cost + Euclid. Dist
        dXnc = cellCoords[0] - parentCoords[0]
        dYnc = cellCoords[1] - parentCoords[1]
        dnc = sqrt(dXnc * dXnc + dYnc * dYnc)
        newCost = parentPathCost + dnc
        # If NewCost < CurrentCost
            # Set parent_cell as new parent for cell
            # Write NewCost as path_cost
        if newCost < cellPathCost:
            cell.path_cost = newCost
            cell.set_parent(parent_cell)
        pass
        
