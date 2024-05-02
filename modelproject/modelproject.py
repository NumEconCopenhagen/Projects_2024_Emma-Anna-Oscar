from scipy import optimize
from types import SimpleNamespace


class CoutnotOligopoly:
    """ This class implements the Cournot oligopoly model"""
    def __init__(self):
        par = self.par = SimpleNamespace()
        par.a = a
        par.b = b
        par.c = c


    def invdemand(self,q1,q2):
        par = self.par
        p = a - b*(q1+q2)
        return p
    
    def cost(self,qi):
        par = self.par
        cost = c*qi
        return cost

    def profit1(self,q1):
        par = self.par
        p = self.invdemand()
        cost1 = self.cost(q1)
        profit1 = p*q1 - cost1
        return profit1

    def profit2(self,q2):
        par = self.par
        p = self.invdemand()
        cost2 = self.cost(q2)
        profit2 = p*q2 - cost2
        return profit2


    
