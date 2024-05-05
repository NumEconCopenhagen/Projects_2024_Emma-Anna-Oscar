from scipy import optimize
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
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

class BertrandOligopoly:
    """ This class implements the Bertrand oligopoly model"""
    def __init__(self,a,b,c):
        par = self.par = SimpleNamespace()
        par.a = a
        par.b = b
        par.c = c

    def demand(self, p1, p2):
        par = self.par
        q1 = (par.a-p1)/par.b
        q2 = (par.a-p2)/par.b
        return q1, q2
    
    def profit1(self, p1, p2):
        par = self.par
        q1 = self.demand(p1,p2)[0]
        if p1 < p2:
            return (p1-par.c)*q1
        elif p1 == p2:
            return (p1-par.c)*q1 / 2
        else:
            return 0
        
    def profit2(self, p1, p2):
        par = self.par
        q2 = self.demand(p1,p2)[1]
        if p2 < p1:
            return (p2-par.c)*q2
        elif p2 == p1:
            return (p2-par.c)*q2 / 2
        else:
            return 0

    def BR1(self, p2):
        par = self.par
        value_of_choice = lambda p1: -self.profit1(p1, p2)
        p1_opt = minimize_scalar(value_of_choice, bounds=(par.c, par.a), method='bounded')
        return p1_opt.x
    
    def BR2(self, p1):
        par = self.par
        value_of_choice = lambda p2: -self.profit2(p1, p2)
        p2_opt = minimize_scalar(value_of_choice, bounds=(par.c, par.a), method='bounded')
        return p2_opt.x
    
    def nash_equilibrium(self):
        par = self.par
        p1 = minimize_scalar(lambda p1: -self.profit1(p1, self.BR2(p1)), bounds=(par.c, par.a), method='bounded').x
        p2 = minimize_scalar(lambda p2: -self.profit2(self.BR1(p2), p2), bounds=(par.c, par.a), method='bounded').x
        return p1, p2
    
    def plot_nash_equilibrium(self, p_range):
        p1_ne, p2_ne = self.nash_equilibrium()
        br1_values = [self.BR1(p2) for p2 in p_range]
        br2_values = [self.BR2(p1) for p1 in p_range]
        plt.plot(p_range, br1_values, label='Best Response of Firm 1', color='blue')
        plt.plot(br2_values, p_range, label='Best Response of Firm 2', color='red')
        plt.scatter(p1_ne, p2_ne, color='green', label='Nash Equilibrium')
        plt.xlabel('Price for Firm 1')
        plt.ylabel('Price for Firm 2')
        plt.title('Bertrand Oligopoly: Nash Equilibrium')
        plt.legend()
        plt.grid(True)
        plt.show()


class CournotOligopoly:
    """ This class implements the Cournot Oligopoly model with i firms"""
    def __init__(self,a,b,c):
        par = self.par = SimpleNamespace()
        par.a = a
        par.b = b
        par.c = c

    def complete_competition(self):
        p = BertrandOligopoly.nash_equilibrium[0]
        return p

    def calculating_market(self):
        i = 0 # counter
        N = np.linspace(1,None,1) # number of firms, starts at 1 firm, no upper bound, only integers allowed (full firms)
        q = qi*N # q = total production of all firms, qi = individual production
        while i <= N:
            def cost(self,qi):
                par = self.par
                cost = par.c*qi
                return cost
        
            def invdemand(self,q):
                par = self.par
                p = par.a - par.b*(q)
                return p

            def profiti(self,qi,q):
                par = self.par
                profit1 = self.invdemand(q)*qi - self.cost(q)
                return profit1

            def BRi(self,qi,q):
                par = self.par
                value_of_choice = lambda qi: -self.profit1(q)
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
        
            i += 1                
            q.append()
            
            return

