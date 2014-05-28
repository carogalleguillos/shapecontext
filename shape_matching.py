import numpy as np
from shape_context import ShapeContext
from munkres import Munkres

class ShapeMatching(object):

    def __init__(self):
        self.eps = np.finfo(float).eps 

    def match_shapes(self, s1, s2):
        cost = self.compute_cost(s1, s2)
        return cost

    def compute_assignment(self):
        """ Hungarian assigment. It returns the best matches between 2 sets of points """
        cost = self.cost
        rows = self.cost.shape[0] - self.cost.shape[1]
        if rows > 0:
            extra = np.zeros((self.cost.shape[0] , rows))
            cost = np.hstack((self.cost, extra))
        m = Munkres()
        indexes = m.compute(cost)
        return  indexes

    def compute_cost(self, s1, s2):
        """ Compute cost of matching shape context features between shapes.
            The smaller the cost the most similar they are.
            Cost is the chi square distance between two features"""

        cost = np.zeros((s1.num_points, s2.num_points))
        a1, b1 = s1.features.shape
        a2, b2 = s2.features.shape
        if a1 > a2:
            sc1 = s1.features
            sc2 = s2.features
        else:
            sc1 = s2.features
            sc2 = s1.features
            a1, a2 = a2, a1 

        feat1 = np.tile(sc2, (a1, 1, 1))
        feat2 = np.tile(sc1, (a2, 1, 1))
        feat2 = np.transpose(feat2, (2, 0, 1))
        feat1 = np.transpose(feat1)
        cost = 0.5 * np.sum(np.power(feat1 - feat2, 2) / (feat1 + feat2 + self.eps), axis=0)  
        sc_cost = np.mean(np.amin(cost, axis=1)) + np.mean(np.amin(cost, axis=0))
        return sc_cost
