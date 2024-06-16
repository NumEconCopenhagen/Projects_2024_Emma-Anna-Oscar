from types import SimpleNamespace

import numpy as np
from scipy import optimize
from scipy.optimize import fsolve

import pandas as pd
import matplotlib.pyplot as plt


class ExchangeEconomyClass:

    def __init__(self):
        '''Initialize the class and define the parameter values'''
        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

    def utility_A(self,x1A,x2A):
        '''Calculate the utility for a given level of x1 and x2 for agent B'''
        par = self.par
        return x1A**par.alpha * x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        '''Calculate the utility for a given level of x1 and x2 for agent B'''
        par = self.par
        return x1B**par.beta * x2B**(1-par.beta)

    def demand_A(self,p1):
        '''Calculate the demand of x1 and x2 for agent A'''
        par = self.par
        x1A = par.alpha*(p1*par.w1A+par.w2A)/p1
        x2A = (1-par.alpha)*(p1*par.w1A+par.w2A)
        return x1A, x2A
    
    def demand_B(self,p1):
        '''Calculate the demand of x1 and x2 for agent B'''
        par = self.par
        x1B = par.beta*((p1*(1-par.w1A))+(1-par.w2A))/p1
        x2B = (1-par.beta)*(p1*(1-par.w1A)+(1-par.w2A))
        return x1B, x2B
       
    def check_market_clearing(self,p1):
        '''Calculate the market clearing errors for a given p1'''
        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2

    def market_clearing_error(self):
        '''Loop over values of p1 to calculate the corresponding market clearing errors'''
        N = 75
        p1_vec = np.linspace(0.5,2.5,N)
        eps1_values, eps2_values = self.check_market_clearing(p1_vec)
        
        return eps1_values, eps2_values

    def market_clearing_price(self):
        '''Find the price that clears the markets'''
        N = 75
        p1_vec = np.linspace(0.5,2.5,N)
        
        market_clearing_price = []

        i = 0
        for p1 in p1_vec:
            eps1_values, eps2_values = self.market_clearing_error()
            if abs(eps1_values[i]) < 0.009:
                market_clearing_price.append(p1)
            i = i+1
        return market_clearing_price
    
    def pareto_improvement(self):
        '''Loop over values of x1 and x2 to see when they yield a pareto improvement from the starting point'''
        par = self.par
        N = 75

        pareto_pairs = []

        x1A_vec = np.linspace(0,1,N)
        x2A_vec = np.linspace(0,1,N)

        for x1A in (x1A_vec):
            for x2A in (x2A_vec):
                if (self.utility_A(x1A, x2A) >= self.utility_A(par.w1A, par.w2A)) and \
                    (self.utility_B(1-x1A, 1-x2A)) >= self.utility_B(1-par.w1A, 1-par.w2A):
                    pareto_pairs.append((x1A, x2A))
        return pareto_pairs
    
    def price_setter_1(self):
        '''Objective function for maximizing the utility of agent A when setting the price within P1'''
        N = 75
        p1_vec = np.linspace(0.5,2.5,N)
        utility_best = -np.inf # initial maximum
        p1_best = np.nan # not-a-number 

        # We loop over the values in the vector of prices
        for p1 in p1_vec:
            # We find the demand of agent B for the prices we loop over
            x1B, x2B = self.demand_B(p1)

            # We define the current utility for agent A by inputting the demand
            # for A when B's demand depends on the prices we loop over
            utility_now = self.utility_A(1-x1B, 1-x2B)

            # We create an if-statement that updates the optimal whenever it is greater
            # than the previous optimal utility. Furthermore, we update the optimal price,
            # the optimal quantities for A and the corresponding quantities for B.
            if utility_now > utility_best:
                utility_best = utility_now
                p1_best = p1
                x1A_best = 1-x1B
                x2A_best = 1-x2B

        return p1_best, utility_best, x1A_best, x2A_best

    def price_setter_2(self):
        '''Objective function for maximizing the utility of agent A when setting any positive price'''
        # Setting the bounds of the price, which can be any positive number
        bounds = [(0,None)]

        # We make an initial guess of 2 for the price, since we saw in 4.1 that within the range of P1, this would be the optimal price
        initial_guess = [2]

        # We create an objective function that is the negative of the utility, sincne the optimizer minimizes.
        # The utility is a function of demand evaulated in p1, so we get x1A and x1B that are variable in the p1-parameter.
        objective = lambda p1: -self.utility_A(1-self.demand_B(p1)[0], 1-self.demand_B(p1)[1])

        # We use the optimization algorithm to find the optimal price
        result = optimize.minimize(objective, initial_guess, bounds=bounds)

        return result.x[0]

    def market_maker(self,p1):
        '''objective function to use for question 5'''
        par = self.par
        x1A,x2A = self.demand_A(p1)
        return -self.utility_A(x1A,x2A)

    
    def utilitarian_planner(self):
        '''objective function to find the Utalitarian social planner's best allocation'''
        # We define the aggregate utility by summing over A's and B's utility at the variable x.
        def agri_utility(x):
            # Defining x as x1A and x2A.
            x1A, x2A = x
            return self.utility_A(x1A, x2A) + self.utility_B(1-x1A, 1 - x2A)
        # We build the constraints, making sure that the quantities are positive and less than 1.
        constraints = ({'type': 'ineq', 'fun': lambda x: x[0]},
                       {'type': 'ineq', 'fun': lambda x: x[1]},
                       {'type': 'ineq', 'fun': lambda x: 1 - x[0]},
                       {'type': 'ineq', 'fun': lambda x: 1 - x[1]})
        
        #Our initial guess, x0, we set to 0.5 for both agents.
        x0 = [0.5, 0.5]

        # We use the optimization algorithm to find the optimal allocation where the total utility is highest.
        result = optimize.minimize(lambda x: -agri_utility(x), x0, constraints=constraints)
        optimal_allocation = result.x
        return optimal_allocation

    def constraint_C(self,x):
        x1A, x2A = x
        return [
            self.utility_A(x1A, x2A) - self.utility_A(self.par.w1A, self.par.w2A),
            self.utility_B(1 - x1A, 1 - x2A) - self.utility_B(1 - self.par.w1A, 1 - self.par.w2A),
            x1A - 1,
            x2A - 1]

    def CreateEndowments(self, n=50):
        #Creating random endowments for A
        return np.random.uniform(0,1,(n,2))
    
    def equilibrium_price(self, wA):
        #Finding equlibrium price for given endowment
        self.par.w1a, self.par.w2a = wA
        p1_optimal = fsolve(lambda p1: self.check_market_clearing(p1)[0], 1)[0]
        return p1_optimal
    
    def OptimalAllocation(self, endowments):
        allocations = []
        for wA in endowments:
            p1_optimal = self.equilibrium_price(wA)
            x1A, x2A = self.demand_A(p1_optimal)
            allocations.append((wA, (x1A, x2A)))
        return allocations 
     
    def W_edgeworthbox(self, endowments, allocations):
          fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
          ax_A = fig.add_subplot(1, 1, 1)

          ax_A.set_xlabel("$x_1^A$")
          ax_A.set_ylabel("$x_2^A$")

          temp = ax_A.twinx()
          temp.set_ylabel("$x_2^B$")
          ax_B = temp.twiny()
          ax_B.set_xlabel("$x_1^B$")
          ax_B.invert_xaxis()
          ax_B.invert_yaxis()
        
          #plotting the pairs
          equilibrium_plots = np.array([alloc[1] for alloc in allocations]).T
          ax_A.scatter(equilibrium_plots[0], equilibrium_plots[1], marker='o', color='orange')
        
          #limits      
          ax_A.plot([0, 1], [0, 0], 1w=2, color='black')
          ax_A.plot([0, 1], [1, 1], 1w=2, color='black')
          ax_A.plot([0, 0], [0, 1], 1w=2, color='black')
          ax_A.plot([1, 1], [0, 1], 1w=2, color='black')

          ax_A.set_xlim([-0.1, 1.1])
          ax_A.set_ylim([-0.1, 1.1])
          ax_B.set_xlim([1.1, -0.1])
          ax_B.set_ylim([1.1, -0.1])

          plt.show()

    

