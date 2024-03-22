from types import SimpleNamespace

import numpy as np
from scipy import optimize

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
        eps_values = []

        for p1 in p1_vec:
            eps_values.append(self.check_market_clearing(p1))
        return eps_values

    def market_clearing_price(self):
        '''Find the price that clears the markets'''
        N = 75
        p1_vec = np.linspace(0.5,2.5,N)
        
        market_clearing_price = []

        i = 0
        for p1 in p1_vec:
            eps_values = self.market_clearing_error()
            eps1 = [pair[0] for pair in eps_values]
            if abs(eps1[i]) < 0.009:
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
                    (self.utility_B(1-x1A, 1-x1A)) >= self.utility_B(1-par.w1A, 1-par.w2A):
                    pareto_pairs.append((x1A, x2A))
        return pareto_pairs
    
    def price_setter(self,p1):
        '''objective function to use for maximizing (minimizing)'''
        par = self.par
        return -par.utility_A(1-par.x1B, 1-par.x2B)
        
    
