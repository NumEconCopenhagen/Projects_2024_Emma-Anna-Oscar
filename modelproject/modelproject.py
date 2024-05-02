%matplotlib inline
from scipy import optimize
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


class CournotDuopoly:
    """ This class implements the Cournot Duopoly model"""
    def __init__(self,a,b,c):
        par = self.par = SimpleNamespace()
        par.a = a
        par.b = b
        par.c = c


    def invdemand(self,q1,q2):
        par = self.par
        p = par.a - par.b*(q1+q2)
        return p
    
    def cost(self,qi):
        par = self.par
        cost = par.c*qi
        return cost

    def profit1(self,q1,q2):
        par = self.par
        profit1 = self.invdemand(q1,q2)*q1 - self.cost(q1)
        return profit1

    def profit2(self,q1,q2):
        par = self.par
        profit2 = self.invdemand(q1,q2)*q2 - self.cost(q2)
        return profit2

    def BR1(self,q2):
        par = self.par
        value_of_choice = lambda q1: -self.profit1(q1,q2)
        q1_opt = optimize.minimize(value_of_choice, method='SLSQP', x0=0)
        return q1_opt.x[0]
    
    def BR2(self,q1):
        par = self.par
        value_of_choice = lambda q2: -self.profit2(q1,q2)
        q2_opt = optimize.minimize(value_of_choice, method='SLSQP', x0=0)
        return q2_opt.x[0]
    
    def q_eval(self,q):
        par = self.par
        q_eval = np.array([q[0] - self.BR1(q[1]),
                           q[1] - self.BR2(q[0])])
        return q_eval

    def nash_equilibrium(self):
        par = self.par
        q_init = np.array([0, 0])
        sol = optimize.fsolve(lambda q: self.q_eval(q), q_init)
        return sol
    
    def ne_plot(self):
        par = self.par

        q_ne = self.nash_equilibrium()
        q_val = np.linspace(0, max(q_ne)*1.5, 100)
        br1_val = [self.BR1(q2) for q2 in q_val]
        br2_val = [self.BR2(q1) for q1 in q_val]


        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1,1,1)
        ax.plot(q_val, br1_val, label='BR for firm 1')
        ax.plot(q_val, br2_val, label='BR for firm 2')
        ax.plot(q_ne[1], q_ne[0], 'o')
        
    