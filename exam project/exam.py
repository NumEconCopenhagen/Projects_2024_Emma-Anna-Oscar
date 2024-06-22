import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

class CareerChoice:
    def __init__(self, seed=None):
        '''Initializes the parameters of the model'''
        par = self.par = SimpleNamespace()
        par.J = 3
        par.N = 10
        par.K = 10000

        par.F = np.arange(1,par.N+1)
        par.sigma = 2

        par.v = np.array([1,2,3])
        par.c = 1

        if seed is not None:
            np.random.seed(seed)
        self.seed = seed        

    def epsdraw(self, size):
        '''Draws epsilon from a normal distribution with mean 0 and standard deviation sigma.
        Changes seed based on input.'''
        par = self.par
        eps= np.random.normal(loc=0, scale=par.sigma, size=size)
        return eps

    def utility(self):
        '''Calculates the expected utility of each career choice given the seed for the normal
        distribution for each career, v_j.'''
        par = self.par
        utility = []
        for v in par.v:
            utility.append(v + np.mean(self.epsdraw(par.K)))
        return utility
    
    def v1(self):
        par = self.par
        EU = []
        for i in range(1,par.N+1):
            eps = np.random.normal(loc=0, scale=par.sigma, size=i)
            eu = 1/i*(1*i + eps.sum())
            EU.append(eu)
        return EU
    
    def v2(self):
        par = self.par
        EU = []
        for i in range(1,par.N+1):
            eps = np.random.normal(loc=0, scale=par.sigma, size=i)
            eu = 1/i*(2*i + eps.sum())
            EU.append(eu)
        return EU
    
    def v3(self):
        par = self.par
        EU = []
        for i in range(1,par.N+1):
            eps = np.random.normal(loc=0, scale=par.sigma, size=i)
            eu = 1/i*(3*i + eps.sum())
            EU.append(eu)
        return EU
    
    def career(self):
        par = self.par
        EUv1 = self.v1()
        EUv2 = self.v2()
        EUv3 = self.v3()
        career = []
        EV = []
        noiseterm = []
        RV = []
        for i in range(0,par.N):
            choice = np.max([EUv1[i], EUv2[i], EUv3[i]])
            if choice == EUv1[i]:
                career.append(1)
                EV.append(EUv1[i])
                noise = np.random.normal(loc=0, scale=par.sigma, size=1)
                noiseterm.append(noise[0])
            elif choice == EUv2[i]:
                career.append(2)
                EV.append(EUv2[i])
                noise = np.random.normal(loc=0, scale=par.sigma, size=1)
                noiseterm.append(noise[0])
            else:
                career.append(3)
                EV.append(EUv3[i])
                noise = np.random.normal(loc=0, scale=par.sigma, size=1)
                noiseterm.append(noise[0])
        
        for i in range(0,par.N):
            RV.append(career[i] + noiseterm[i])
        return career, EV, RV
    
    def sort(self):
        par = self.par
        career, EV, RV = self.career()

        careerdict = {}
        for i in range(1,par.N+1):
            careerdict[i] = career[i-1]

        EVdict = {}
        for i in range(1,par.N+1):
            EVdict[i] = EV[i-1]
        
        RVdict = {}
        for i in range(1,par.N+1):
            RVdict[i] = RV[i-1]
        
        return careerdict, EVdict, RVdict

