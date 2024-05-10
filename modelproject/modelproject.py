from scipy import optimize
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

class CournotDuopoly:
    """
    This class is able to find the Nash Equilibrium in a Cournot Duopoly model, given parameter values
    for the demand at cost zero, a, the slope of the demand curve, b, and the marginal cost, c.
    After finding the Nash Equilibrium, the class is able to plot the best response functions and the Nash 
    Equilibria with sliders for the three parameters.
    """
    def __init__(self,a,b,c):
        # Defining the parameters in order to refer to them as par.a, par.b, par.c later.
        par = self.par = SimpleNamespace()
        par.a = a
        par.b = b
        par.c = c

    def invdemand(self,q1,q2):
        #This function is for the inverse demand.
        #We create a reference to the parameters from the initialization.
        par = self.par

        #We define the inverse demand as a function of the two quantities, returning the price that the
        #demand results in.
        p = par.a - par.b*(q1+q2)
        return p
    
    def cost(self,qi):
        #This function is for the costs the firms face.
        #We create a reference to the parameters from the initialization.
        par = self.par

        #The cost is simply the quantity produced times the marginal cost, and we return the cost.
        cost = par.c*qi
        return cost

    def profit1(self,q1,q2):
        #This function defines profits for firm 1.
        #We call the inverse demand as a function of both quantities and the cost as a function of firm 1's own quantity.
        #We return the profits of firm 1.
        profit1 = self.invdemand(q1,q2)*q1 - self.cost(q1)
        return profit1

    def profit2(self,q1,q2):
        #This function defines profits for firm 2.
        #We call the inverse demand as a function of both quantities and the cost as a function of firm 2's own quantity.
        #We return the profits of firm 2.
        profit2 = self.invdemand(q1,q2)*q2 - self.cost(q2)
        return profit2

    def BR1(self,q2):
        #This function defines the best response for firm 1.
        #We define a value of choice as the negative profits of firm 1, so the minimization below results in a maximization.
        #The value of choice function is evaluated in q1, since this is what we want to maximize with regards to.
        value_of_choice = lambda q1: -self.profit1(q1,q2)
        
        #We define the optimal q1 as the minimized value of the value of choice function. We use the SLSQP method with a initial
        #guess of 0.
        q1_opt = optimize.minimize(value_of_choice, method='SLSQP', x0=0)
        #We return the first element of q1_opt to get the optimal quantity.
        return q1_opt.x[0]
    
    def BR2(self,q1):
        #The best response of firm 2 is exactly symmetrical to BR1. See the above description.
        value_of_choice = lambda q2: -self.profit2(q1,q2)
        q2_opt = optimize.minimize(value_of_choice, method='SLSQP', x0=0)
        return q2_opt.x[0]
    
    def q_eval(self,q):
        #This function allows us to use a root finding algorithm to find the Nash Equilibrium. 
        #q_eval is a function of q, which is a vector with 2 elements that will be varied in the nash_equilibrium function.
        #We need this q to find the exact point where BOTH firms are best-responding to each other.
        #We evaluate the best response functions of both firms in q.
        #For firm 1, we return the difference between the FIRST element of q (the q1-value that firm 2 is best-responding to) 
        #and the BR1-function evaluated in the SECOND element of q (corresponding to a given q2-value).
        #For firm 2, we return the difference between the SECOND element of q (the q2-value that firm 1 is best-responding to)
        #and the BR2-function evaluated in the FIRST element of q (corresponding to a given q1-value).
        #Thus, we get out a 2x1 numpy-array.
        q_eval = np.array([q[0] - self.BR1(q[1]),
                           q[1] - self.BR2(q[0])])
        return q_eval

    def nash_equilibrium(self):
        #This function uses a root finding algorithm to find the value of q (q1,q2) that makes q_eval equal to zero. 
        #When q_eval is zero, both firms are best-responding to the other firm's quantity. This is our Nash Equilibrium.
        #We start with an initial guess of [0,0].
        q_init = np.array([0, 0])
        
        #The fsolve funtion now tries different values of q until both equations in q_eval are equal to zero.
        sol = optimize.fsolve(lambda q: self.q_eval(q), q_init)
        #We return the solution, which is a 2x1 numpy-array containing the q1- and q2-values of the Nash Equilibrium.
        return sol
    
    def ne_plot(self):
        #The ne_plot function is a method that allows us to create an interactive plot the best response functions and 
        #the Nash Equilibrium for different values of a, b and c.
        #First we use the @interact decorator to create sliders for the three parameters. It is important that a starts 
        #where c ends, so that a ≥ c always holds and we are not able to get negative quantities.
        @interact(a = (20,50,1), b = (0.1,1,0.1), c = (0,20,1))
        def plot(a,b,c):
            #Now, we call the CournotDuopoly class with the interactive parameter values. 
            dp_cournot = CournotDuopoly(a,b,c)

            #We find the Nash Equilibrium for the given parameter values.
            q_ne = dp_cournot.nash_equilibrium()
            #We create a range of q-values to plot the best response functions against. The lower bound is 0 and the upper
            #bound is 50% above the maximum q-value in the Nash Equilibrium, so that the plot is not too squeezed, regardless
            #of the parameter values.
            q_val = np.linspace(0, max(q_ne)*1.5, 100)
            #We create a list of the best response values in the range of q-values for firm 1 and firm 2, respectively.
            br1_val = [dp_cournot.BR1(q2) for q2 in q_val]
            br2_val = [dp_cournot.BR2(q1) for q1 in q_val]

            #We initialize our figure and axis.
            fig = plt.figure(dpi=100)
            ax = fig.add_subplot(1,1,1)
            #We plot the best response functions against the range of q-values for firms 1 and firm 2, respectively.
            ax.plot(q_val, br1_val, label='BR for firm 1', color='grey', linestyle='--')
            ax.plot(br2_val, q_val, label='BR for firm 2', color='grey')
            #We plot the Nash Equilibrium as a blue dot and annotate the point with the q1- and q2-values.
            ax.plot(q_ne[1], q_ne[0], 'bo')
            ax.annotate(f'NE: ({q_ne[0]:.1f}, {q_ne[1]:.1f})', xy=q_ne, xytext=(10,10), textcoords='offset points',  
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            #We set the labels and title of the plot and add a legend.
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
            ax.plot(p_ne[1], p_ne[0], 'o', color='mediumpurple')
            ax.annotate(f'NE: ({p_ne[0]:.1f}, {p_ne[1]:.1f})', xy=p_ne, xytext=(10,10), textcoords='offset points',  
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            ax.set_xlabel('Price for firm 1')
            ax.set_ylabel('Price for firm 2')
            ax.set_title('Figure 2: Nash Equilibria in Bertrand Duopoly')
            ax.legend()

    
class Oligopoly:
    """This class shows the analytical solution to the Bertrand oligopoly for N firms"""
    def __init__(self,c):
        par = self.par = SimpleNamespace()
        par.c = c
    
    def nash_price_bertrand(self):
        par = self.par
        p = par.c
        return p
    
    def nash_profit_bertrand(self):
        par = self.par
        profits = []
        i = 1
        while i <= 50:
            profit = (self.nash_price_bertrand()-par.c)/2
            profits.append(profit)
            i+=1
        return profits
    
    def nash_profit_cournot(self):
        par = self.par
        profits = []
        for n in range(1,51):
            q = (20-par.c)/((n+1)*2)
            profit = (20-2*q*n)*q - par.c*q
            profits.append(profit)
        return profits

    def plot_convergence(self):
        oligopoly1 = Oligopoly(1)
        oligopoly2 = Oligopoly(5)
        oligopoly3 = Oligopoly(10)
        oligopoly4 = Oligopoly(15)

        firms = np.linspace(0,20,50)

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), dpi=100, constrained_layout=True)
        fig.suptitle('Figure 3: Convergence to zero-profits as number of firms increases, Cournot & Bertrand Oligopoly', fontsize=16)
        axs[0][0].set_title('Marginal cost = 1')
        axs[0][0].set_xlim([0,20])
        axs[0][0].set_ylim([-1,45])
        axs[0][0].set_xlabel('# of firms')
        axs[0][0].set_ylabel('$\pi$')
        axs[0][0].plot(firms, oligopoly1.nash_profit_cournot(), label='Cournot', color='indigo')
        axs[0][0].plot(firms, oligopoly1.nash_profit_bertrand(), label='Bertrand', color='mediumpurple', linestyle='--')
        axs[0][0].legend()
        axs[0][0].grid('on', linestyle = '--')

        axs[0][1].set_title('Marginal cost = 5')
        axs[0][1].set_xlim([0,20])
        axs[0][1].set_ylim([-1,45])
        axs[0][1].set_xlabel('# of firms')
        axs[0][1].set_ylabel('$\pi$')
        axs[0][1].plot(firms, oligopoly2.nash_profit_cournot(), label='Cournot', color='indigo')
        axs[0][1].plot(firms, oligopoly2.nash_profit_bertrand(), label='Bertrand', color='mediumpurple', linestyle='--')
        axs[0][1].legend()
        axs[0][1].grid('on', linestyle = '--')

        axs[1][0].set_title('Marginal cost = 10')
        axs[1][0].set_xlim([0,20])
        axs[1][0].set_ylim([-1,45])
        axs[1][0].set_xlabel('# of firms')
        axs[1][0].set_ylabel('$\pi$')
        axs[1][0].plot(firms, oligopoly3.nash_profit_cournot(), label='Cournot', color='indigo')
        axs[1][0].plot(firms, oligopoly3.nash_profit_bertrand(), label='Bertrand', color='mediumpurple', linestyle='--')
        axs[1][0].legend()
        axs[1][0].grid('on', linestyle = '--')


        axs[1][1].set_title('Marginal cost = 15')
        axs[1][1].set_xlim([0,20])
        axs[1][1].set_ylim([-1,45])
        axs[1][1].set_xlabel('# of firms')
        axs[1][1].set_ylabel('$\pi$')
        axs[1][1].plot(firms, oligopoly4.nash_profit_cournot(), label='Cournot', color='indigo')
        axs[1][1].plot(firms, oligopoly4.nash_profit_bertrand(), label='Bertrand', color='mediumpurple', linestyle='--')
        axs[1][1].legend()
        axs[1][1].grid('on', linestyle = '--')

        plt.show()
