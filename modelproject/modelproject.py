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
        '''Defining the parameters in order to refer to them as par.a, par.b, par.c later.'''
        par = self.par = SimpleNamespace()
        par.a = a
        par.b = b
        par.c = c

    def invdemand(self,q1,q2):
        '''This function is for the inverse demand.'''
        #We create a reference to the parameters from the initialization.
        par = self.par

        #We define the inverse demand as a function of the two quantities, returning the price that the
        #demand results in.
        p = par.a - par.b*(q1+q2)
        return p
    
    def cost(self,qi):
        '''This function is for the costs the firms face.'''
        #We create a reference to the parameters from the initialization.
        par = self.par

        # qi refers to the quantity produced by firm i, so can be used to define the costs of either firm 1 or 2.
        # The cost is simply the quantity produced times the marginal cost, and we return the cost.
        cost = par.c*qi
        return cost

    def profit(self,qi,qj):
        '''This function is for the profits of firm i.'''
        #We call the inverse demand as a function of the given firm's own quantity, qi, and the other firm's quantity, qj.
        #The cost is just called as a function of firm i's own quantity.
        #We return the profits of firm i.
        profit = self.invdemand(qi,qj)*qi - self.cost(qi)
        return profit

    def BR(self,qj):
        '''This function defines the best response for firm i.'''
        #We define a value of choice as the negative profits of firm i, so the minimization below results in a maximization.
        #qj refers to the quantity produced by the other firm. So for the best response of firm 1, qj would refer to firm 2's quantity.
        #The value of choice function is evaluated in qj, since the best response of the firms depend on the quentity of the other firm.
        value_of_choice = lambda qi: -self.profit(qi,qj)
        
        #We define the optimal qi as the minimized value of the value of choice function. We use the SLSQP method with an initial
        #guess of 0.
        qi_opt = optimize.minimize(value_of_choice, method='SLSQP', x0=0)
        #We return the first element of q_opt to get the optimal quantity.
        return qi_opt.x[0]
 
    def q_eval(self,q):
        '''This function allows us to use a root finding algorithm to find the Nash Equilibrium. '''
        #q_eval is a function of q, which is a vector with 2 elements that will be varied in the nash_equilibrium function.
        #We need this q to find the exact point where BOTH firms are best-responding to each other.
        #We evaluate the best response functions of both firms in q.
        #For firm 1, we return the difference between the FIRST element of q (the q1-value that firm 2 is best-responding to) 
        #and the BR-function evaluated in the SECOND element of q (corresponding to a given q2-value).
        #For firm 2, we return the difference between the SECOND element of q (the q2-value that firm 1 is best-responding to)
        #and the BR-function evaluated in the FIRST element of q (corresponding to a given q1-value).
        #Since the firms are identical, their best response functions are the same, but this evaluation function still finds a vector
        #that is evaluated so that firm 1 best responds to firm 2 and vice versa.
        #Thus, we get out a 2x1 numpy-array.
        q_eval = np.array([q[0] - self.BR(q[1]),
                           q[1] - self.BR(q[0])])
        return q_eval

    def nash_equilibrium(self):
        '''This function uses a root finding algorithm to find the value of q (q1,q2) that makes q_eval equal to zero.''' 
        #When q_eval is zero, both firms are best-responding to the other firm's quantity. This is our Nash Equilibrium.
        #We start with an initial guess of [0,0].
        q_init = np.array([0, 0])
        
        #The fsolve funtion now tries different values of q until both equations in q_eval are equal to zero.
        sol = optimize.fsolve(lambda q: self.q_eval(q), q_init)
        #We return the solution, which is a 2x1 numpy-array containing the q1- and q2-values of the Nash Equilibrium.
        return sol
    
    def ne_plot(self):
        '''The ne_plot function is a method that allows us to create an interactive plot the best response functions and 
        the Nash Equilibrium for different values of a, b and c.'''
        #First we use the @interact decorator to create sliders for the three parameters. It is important that a starts 
        #where c ends, so that a ≥ c always holds and we are not able to get negative quantities.
        @interact(a = (20,50,1), b = (0.1,2,0.1), c = (0,20,1))
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
            br1_val = [dp_cournot.BR(q2) for q2 in q_val]
            br2_val = [dp_cournot.BR(q1) for q1 in q_val]

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
    """ 
    This class simulates a Bertrand duopoly where two firms strategically set prices for their identical products.
    The class is able to find the Nash Equilibrium, given parameter values for the demand at cost zero, a, 
    the slope of the demand curve, b, and the marginal cost, c.
    After finding the Nash Equilibrium, the class is able to plot the best response functions and the Nash 
    Equilibria with sliders for the three parameters.
    """
    def __init__(self, a, b, c):
        '''Initializing the Bertrand duopoly by defining the parameters a, b and c. Using self as a reference later on.'''
        self.a = a
        self.b = b
        self.c = c
    
    def demand(self, p):
        '''The demand function returns the quantity demanded with a given price p.'''
        return (self.a - p) / self.b
    
    def profit(self, pi, pj):
        '''The profit function returns the profit of firm i, given the price set by firm i and j respectively.'''
        #We define the quantity demanded by firm i, using the demand function.
        qi = self.demand(pi)
        #We create the binding conditions for the demand, as described in section 2.3.
        #This way, we are able to capture the effects from the assumptions made in the Bertrand model.
        if pi < pj:
            return (pi-self.c)*qi
        elif pi == pj:
            return (pi-self.c)*qi / 2
        else:
            return 0
    
    def BR(self, pj):
        '''The best response function for firm i, given the price set by firm j.'''
        #First, we define an objective function for us to maximize.
        def objective(pi):
            #The objective function is the negative profit of firm i, so the minimization below results in a maximization.
            return -self.profit(pi, pj)
        #Using the maximize function from scipy to find the optimal price in regards to maximizing firm i's profit.
        #We use bounds to specify that the price never goes below the marginal cost, c.
        result = minimize(objective, self.c, bounds=[(self.c, None)])
        #result.x provides the optimal solution for firm i.
        return result.x[0]


    def p_eval(self,p):
        '''This function allows us to use a root finding algorithm to find the Nash Equilibrium.'''
        #It operates on a vector, p, with two elements, representing the prices set by both firms.
        #Evaluating the best response functions of both firms based on p helps identify the exact point 
        #where each firm optimally responds the the others strategy
        #For firm 1, it computes the difference between its price (p1) and the best response to the price
        #set by firm 2 (p2).
        #Similarly, for firm 2, it provides the difference between its optimal price (p2), which firm 1 is best responding to,
        #and the best response to the price set by firm 1 (p1).
        #The result is a 2x1 numpy array capturing the deviations from optimality for both firms.
        p_eval = np.array([p[0] - self.BR(p[1]),
                           p[1] - self.BR(p[0])])
        return p_eval

    def nash_equilibrium(self):
        #This function uses a root finding algorithm to find the value of p (p1,p2) that makes p_eval equal to zero. 
        #When p_eval is zero, both firms are best-responding to the other firm's price. This is our Nash Equilibrium.
        #We start with an initial guess of [0,0].
        p_init = np.array([0, 0])

        #The fsolve funtion now tries different values of p until both equations in p_eval are equal to zero.
        sol = optimize.fsolve(lambda p: self.p_eval(p), p_init)
        #We return the solution, which is a 2x1 numpy-array containing the p1- and p2-values of the Nash Equilibrium.
        return sol

    def ne_plot(self):
        #The ne_plot function is a method that allows us to create an interactive plot the best response functions and 
        #the Nash Equilibrium for different values of a, b and c.
        #First we use the @interact decorator to create sliders for the three parameters. It is important that a starts 
        #where c ends, so that a ≥ c always holds and we are not able to get negative prices.
        @interact(a = (20,50,1), b = (0.1,1,0.1), c = (0,20,1))
        def plot(a,b,c):
            #We call the Bertrand class
            dp_bertrand = BertrandDuopoly(a,b,c)

            #We create a range of p-values to plot the best response functions against. The lower bound is 0 and the upper
            #bound is 50% above the maximum p-value in the Nash Equilibrium, so that the plot is not too squeezed, regardless
            #of the parameter values.
            p_ne = dp_bertrand.nash_equilibrium()
            p_val = np.linspace(0, max(p_ne)*1.5, 100)
            br1_val = [dp_bertrand.BR(p2) for p2 in p_val]
            br2_val = [dp_bertrand.BR(p1) for p1 in p_val]

            #Initializing our figure and axis
            fig = plt.figure(dpi=100)
            ax = fig.add_subplot(1,1,1)
            #We plot the best response functions against the range of q-values for firms 1 and firm 2, respectively.
            ax.plot(p_val, br1_val, label='BR for firm 1', color='grey', linestyle='--')
            ax.plot(br2_val, p_val, label='BR for firm 2', color='grey')
            #We plot the Nash Equilibrium as a mediumpurple dot.
            ax.plot(p_ne[1], p_ne[0], 'o', color='mediumpurple')
            ax.annotate(f'NE: ({p_ne[0]:.1f}, {p_ne[1]:.1f})', xy=p_ne, xytext=(10,10), textcoords='offset points',  
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            #We set the labels and title of the plot and add a legend.
            ax.set_xlabel('Price for firm 1')
            ax.set_ylabel('Price for firm 2')
            ax.set_title('Figure 2: Nash Equilibria in Bertrand Duopoly')
            ax.legend()

    
class Oligopoly:
    """This class shows the analytical solution to the Bertrand oligopoly for N firms"""
    def __init__(self,c):
        par = self.par = SimpleNamespace()
        # We only wish to analyise what happens for given values of c. We therefor keep a and b constant at a=20 and b=2.
        par.c = c
    
    def nash_price_bertrand(self):
        par = self.par
        # We know the price will be set equal to marginal cost, ergo:
        p = par.c
        return p
    
    def nash_profit_bertrand(self):
        par = self.par
        # The profts of all firms will be equal to zero, no matter how many firms are present. 
        # We will use this as what the Cournot model converges towards, meaning this method essentially allows for plotting a straight line = 0 for n firms.
        profits = []
        i = 1
        while i <= 50:
            profit = (self.nash_price_bertrand()-par.c)/2
            profits.append(profit)
            i+=1
        return profits
    
    def nash_profit_cournot(self):
        # We would like to analysie the convergence of a Cournot Oligopoly towards complete compettion, ergo profit equal to zero.
        # This method calculates the NE of the Cournot model for a given value of c and n firms. 
        par = self.par
        profits = []
        for n in range(1,51):
            q = (20-par.c)/((n+1)*2)
            profit = (20-2*q*n)*q - par.c*q
            profits.append(profit)
        return profits

    def plot_convergence(self):
        # Calling the methods above for given values of c (1, 5, 10, 15) to analyse their respective convergence. 
        oligopoly1 = Oligopoly(1)
        oligopoly2 = Oligopoly(5)
        oligopoly3 = Oligopoly(10)
        oligopoly4 = Oligopoly(15)

        # Set the possible number of firms within the market to be between 0 and 20, with 50 steps in between.
        firms = np.linspace(0,20,50)

        # plotting the first convergence for c=1
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), dpi=100, constrained_layout=True)
        fig.suptitle('Figure 3: Convergence to zero-profits as number of firms increases, Cournot & Bertrand Oligopoly', fontsize=16)
        axs[0][0].set_title('Marginal cost = 1')
        axs[0][0].set_xlim([0,20])
        axs[0][0].set_ylim([-1,45])
        axs[0][0].set_xlabel('# of firms')
        axs[0][0].set_ylabel('$\pi$')
        # We use the Bertrand model as the baseline end goal where the firms profit is equal to zero, just like in a market of complete competition.
        # We do this because in the Bertrand model, even two firms set their price equal to marginal cost, which is the same outcome as complete competition markets. 
        axs[0][0].plot(firms, oligopoly1.nash_profit_cournot(), label='Cournot', color='indigo')
        axs[0][0].plot(firms, oligopoly1.nash_profit_bertrand(), label='Bertrand', color='mediumpurple', linestyle='--')
        axs[0][0].legend()
        axs[0][0].grid('on', linestyle = '--')

        # same plot for c=5
        axs[0][1].set_title('Marginal cost = 5')
        axs[0][1].set_xlim([0,20])
        axs[0][1].set_ylim([-1,45])
        axs[0][1].set_xlabel('# of firms')
        axs[0][1].set_ylabel('$\pi$')
        axs[0][1].plot(firms, oligopoly2.nash_profit_cournot(), label='Cournot', color='indigo')
        axs[0][1].plot(firms, oligopoly2.nash_profit_bertrand(), label='Bertrand', color='mediumpurple', linestyle='--')
        axs[0][1].legend()
        axs[0][1].grid('on', linestyle = '--')

        # for c=10
        axs[1][0].set_title('Marginal cost = 10')
        axs[1][0].set_xlim([0,20])
        axs[1][0].set_ylim([-1,45])
        axs[1][0].set_xlabel('# of firms')
        axs[1][0].set_ylabel('$\pi$')
        axs[1][0].plot(firms, oligopoly3.nash_profit_cournot(), label='Cournot', color='indigo')
        axs[1][0].plot(firms, oligopoly3.nash_profit_bertrand(), label='Bertrand', color='mediumpurple', linestyle='--')
        axs[1][0].legend()
        axs[1][0].grid('on', linestyle = '--')

        # for c=15
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
