from scipy import optimize
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

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
        @interact(a = (20,50,1), b = (0.1,1,0.1), c = (0,20,1))
        def plot(a,b,c):
            dp_cournot = CournotDuopoly(a,b,c)

            q_ne = dp_cournot.nash_equilibrium()
            q_val = np.linspace(0, max(q_ne)*1.5, 100)
            br1_val = [dp_cournot.BR1(q2) for q2 in q_val]
            br2_val = [dp_cournot.BR2(q1) for q1 in q_val]


            fig = plt.figure(dpi=100)
            ax = fig.add_subplot(1,1,1)
            ax.plot(q_val, br1_val, label='BR for firm 1', color='grey', linestyle='--')
            ax.plot(br2_val, q_val, label='BR for firm 2', color='grey')
            ax.plot(q_ne[1], q_ne[0], 'bo')
            ax.annotate(f'NE: ({q_ne[0]:.1f}, {q_ne[1]:.1f})', xy=q_ne, xytext=(10,10), textcoords='offset points',  
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            ax.set_xlabel('Quantity for firm 1')
            ax.set_ylabel('Quantity for firm 2')
            ax.set_title('Nash Equilibria in Cournot Duopoly')
            ax.legend()


class BertrandDuopoly:
    """ This class implements the Bertrand Duopoly model"""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    
    def demand(self, p):
        return (self.a - p) / self.b
    
    def profit_firm1(self, p1, p2):
        q1 = self.demand(p1) 
        if p1 < p2:
            return (p1-self.c)*q1
        elif p1 == p2:
            return (p1-self.c)*q1 / 2
        else:
            return 0

    def profit_firm2(self, p1, p2):
        q2 = self.demand(p2)
        if p2 < p1:
            return (p2-self.c)*q2
        elif p2 == p1:
            return (p2-self.c)*q2 / 2
        else:
            return 0

    def BR1(self, p2):
        def objective(p1):
            return -self.profit_firm1(p1, p2)
        result = minimize(objective, self.c, bounds=[(self.c, None)])
        return result.x[0]

    def BR2(self, p1):
        def objective(p2):
            return -self.profit_firm2(p1, p2)
        result = minimize(objective, self.c, bounds=[(self.c, None)])
        return result.x[0]
    
    def p_eval(self,p):
        p_eval = np.array([p[0] - self.BR1(p[1]),
                           p[1] - self.BR2(p[0])])
        return p_eval

    def nash_equilibrium(self):
        p_init = np.array([0, 0])
        sol = optimize.fsolve(lambda p: self.p_eval(p), p_init)
        return sol

    def ne_plot(self):
        @interact(a = (20,50,1), b = (0.1,1,0.1), c = (0,20,1))
        def plot(a,b,c):
            dp_bertrand = BertrandDuopoly(a,b,c)

            p_ne = dp_bertrand.nash_equilibrium()
            p_val = np.linspace(0, max(p_ne)*1.5, 100)
            br1_val = [dp_bertrand.BR1(p2) for p2 in p_val]
            br2_val = [dp_bertrand.BR2(p1) for p1 in p_val]


            fig = plt.figure(dpi=100)
            ax = fig.add_subplot(1,1,1)
            ax.plot(p_val, br1_val, label='BR for firm 1', color='grey', linestyle='--')
            ax.plot(br2_val, p_val, label='BR for firm 2', color='grey')
            ax.plot(p_ne[1], p_ne[0], 'bo')
            ax.annotate(f'NE: ({p_ne[0]:.1f}, {p_ne[1]:.1f})', xy=p_ne, xytext=(10,10), textcoords='offset points',  
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            ax.set_xlabel('Price for firm 1')
            ax.set_ylabel('Price for firm 2')
            ax.set_title('Nash Equilibria in Bertrand Duopoly')
            ax.legend()

class CournotOligopoly:
    """ This class implements the Cournot Oligopoly model with n firms"""
    def __init__(self,c,i,n):
        par = self.par = SimpleNamespace()
        par.c = c
        par.i = i
        par.n = n
        
    def cost(self,qi):
        par = self.par
        cost = par.c*qi
        return cost

    def invdemand(self,qi):
        par = self.par
        for i in range(1,par.n,20):
            p = 20 - 2*()
            return p

    def profiti(self,qi):
        par = self.par
        profit1 = self.invdemand(qi*par.i)*qi - self.cost(qi)
        return profit1

    def BRi(self):
        par = self.par
        value_of_choice = lambda qi: -self.profit1(qi*(par.i-1))
        qi_opt = optimize.minimize(value_of_choice, method='SLSQP', x0=0)
        return qi_opt.x[0]
            
    def q_eval(self,q):
        par = self.par
        q_eval = np.array(q[0] - self.BRi(q[1]))
        return q_eval

    def nash_equilibrium(self):
        par = self.par
        q_init = np.array([0, 0])
        sol = optimize.fsolve(lambda q: self.q_eval(q), q_init)
        return sol
    
class BertrandOligopoly:
    """This class shows the analytical solution to the Bertrand oligopoly for N firms"""
    def __init__(self,c):
        par = self.par = SimpleNamespace()
        par.c = c
    
    def nash_price(self):
        par = self.par
        p = par.c
        return p
    
    def nash_profit(self):
        par = self.par
        profits = []
        i = 1
        while i <= 50:
            profit = (self.nash_price()-par.c)/2
            profits.append(profit)
            i+=1
        return profits


class CournotOligopoly:
    def __init__(self,c):
        par = self.par = SimpleNamespace()
        par.c = c
    
    def nash_profit(self):
        par = self.par
        profits = []
        for n in range(1,51):
            q = (20-par.c)/((n+1)*2)
            profit = (20-2*q*n)*q - par.c*q
            profits.append(profit)
        return profits
