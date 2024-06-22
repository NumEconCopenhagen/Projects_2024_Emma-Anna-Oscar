import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize

class ProductionEconomy:

    def __init__(self):
        par = self.par = SimpleNamespace()

        # firms
        par.A = 1.0
        par.gamma = 0.5

        # households
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0

        # government
        par.tau = 0.0
        par.T = 0.0

        # Question 3
        par.kappa = 0.1

    def opt_labor1(self, p1, w):
        par = self.par
        labor1 = (p1*par.A * par.gamma/w) ** (1/ (1-par.gamma))
        return labor1
    
    def opt_labor2(self, p2, w):
        par = self.par
        labor2 = (p2*par.A * par.gamma/w) ** (1/(1-par.gamma))
        return labor2

    def opt_output1(self,p1,w):
        par = self.par
        output1 = par.A * (self.opt_labor1(p1,w) ** par.gamma)
        return output1
    
    def opt_output2(self,p2,w):
        par = self.par
        output2 = par.A * (self.opt_labor2(p2,w) ** par.gamma)
        return output2
    
    def profit1(self, p1, w):
        par = self.par
        profits1 = ((1 - par.gamma)*w / par.gamma) * (p1 * par.A * par.gamma/w)**(1/(1-par.gamma))
        return profits1

    def firm1(self):
        par = self.par
        obj_f1 = lambda p1,w: p1*self.opt_output1(p1,w) - w*self.opt_labor1(p1,w)
        constraint = self.opt_output1() - par.A*(self.opt_labor1()**par.gamma)

        sol_firm1 = optimize.minimize(obj_f1, x0=0, bounds=(0,None), constraints=constraint, method='bounded')

        return sol_firm1
    
    def profit2(self, p2, w):
        par = self.par
        profits2 = ((1-par.gamma)*w/par.gamma) * (p2*par.A *par.gamma/w)**(1 / (1-par.gamma))
        return profits2
    
    def firm2(self,p2,w):
        par = self.par
        obj_f2 = lambda p2,w: p2*self.opt_output2(p2,w) - w*self.opt_labor2(p2,w)
        constraint = self.opt_output2(p2,w) - par.A*(self.opt_labor2(p2,w)**par.gamma)

        sol_firm2 = optimize.minimize(obj_f2, x0=0, bounds=(0,None), constraints=constraint, method='bounded')

        return sol_firm2
    
    
    def optimal_behavior_c1(self, p1, p2, w):
        par = self.par
        c1 = par.alpha * (w*(self.opt_labor1(p1,w)+self.opt_labor2(p2,w))+par.T+self.profit1(p1,w)+self.profit2(p2,w)) /p1
        return c1
    
    def optimal_behavior_c2(self, p1, p2, w):
        par = self.par
        c2 = (1-par.alpha) * (w*(self.opt_labor1(p1,w)+self.opt_labor2(p2,w))+par.T+self.profit1(p1,w)+self.profit2(p2,w)) / (p2 + par.tau)
        return c2

    def utility(self, p1, p2, w):
        par = self.par
        constraint = p1*self.optimal_behavior_c1(p1,p2,w) + (p2+par.tau)*self.optimal_behavior_c2(p1,p2,w) - w*(self.opt_labor1(p1,w)+self.opt_labor2(p2,w)) - par.T - self.profit1(p1,w) - self.profit2(p2,w)
        obj_u = np.log(self.optimal_behavior_c1(p1,p2,w)**par.alpha * self.optimal_behavior_c2(p1,p2,w)**(1-par.alpha)) - par.nu*((self.opt_labor1(p1,w)+self.opt_labor2(p2,w))**(1+par.epsilon)) / (1+par.epsilon)

        sol_u = optimize.minimize(obj_u, x0=0, constraints=constraint, method='SLSQP')

        return sol_u
    
    def optimal_behavior_labor(self,p1,p2,w):
        par = self.par
        obj_l = np.log((self.optimal_behavior_c1(p1,p2,w)**par.alpha) * (self.optimal_behavior_c2(p1,p2,w)**(1-par.alpha))) - par.nu * ((self.opt_labor1(p1,w)+self.opt_labor2(p2,w))**(1+par.epsilon)) / (1+par.epsilon)
        sol_l = optimize.minimize(obj_l, x0=0, method='SLSQP')
        return sol_l
    
    def labor_market_clearing(self,p1,p2,w):
        l = self.optimal_behavior_labor(p1,p2,w)
        l1 = self.opt_labor1(p1,w)
        l2 = self.opt_labor2(p2,w)
        labor_market = l - (l1 + l2)
        return labor_market
    
    def goods_market_clearing(self,p1,p2,w):
        c1 = self.optimal_behavior_c1(p1,p2,w)
        y1 = self.opt_output1(p1,w)
        good_market1 = c1 - y1
        return good_market1
    
    def goods_market_clearing2(self,p1,p2,w):
        c2 = self.optimal_behavior_c2(p1,p2,w)
        y2 = self.opt_output2(p2,w)
        good_market2 = c2 - y2
        return good_market2


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
    
    def simulate(self):
        par = self.par

        careerdict = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: []
        }
        EVdict = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: []
        }
        RVdict = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: []
        }

        for i in range(0,par.N):
            n = 0
            while n <= par.K:
                career, EV, RV = self.career()
                careerdict[i+1].append(career[i])
                EVdict[i+1].append(EV[i])
                RVdict[i+1].append(RV[i])
                n += 1

        return careerdict, EVdict, RVdict
    

def count(list):
    count1 = 0
    count2 = 0
    count3 = 0
    for element in list:
        if element == 1:
            count1 += 1
        elif element == 2:
            count2 += 1
        else:
            count3 += 1

    v1 = count1/len(list)
    v2 = count2/len(list)
    v3 = count3/len(list)
    
    h = [v1, v2, v3]

    return h

def plot_career(C1,C2,C3,C4,C5,C6,C7,C8,C9,C10):
        '''This function plots the shares of the different career choices for i = 1,2,...,10'''

        # plotting the 10 graphs for the shares of the different career choices
        fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(12, 20), dpi=100, constrained_layout=True)
        fig.suptitle('Figure 2.1: Distribution of career choices for i = 1,...,10', fontsize=16)
        labels = ['v1', 'v2' , 'v3']

        # then, creating the individual plots for each i
        # for i = 1:
        axs[0][0].set_ylabel('percentage')
        axs[0][0].bar(labels, C1, color='mediumpurple')
        axs[0][0].set_title('i = 1')
        axs[0][0].set_ylim(0,1)

        # for i = 2:
        axs[0][1].set_ylabel('percentage')
        axs[0][1].bar(labels,C2, color='mediumpurple')
        axs[0][1].set_title('i = 2')
        axs[0][1].set_ylim(0,1)

        # for i = 3:
        axs[1][0].set_ylabel('percentage')
        axs[1][0].bar(labels,C3, color='mediumpurple')
        axs[1][0].set_title('i = 3')
        axs[1][0].set_ylim(0,1)

        # for i = 4:
        axs[1][1].set_ylabel('percentage')
        axs[1][1].bar(labels,C4, color='mediumpurple')
        axs[1][1].set_title('i = 4')
        axs[1][1].set_ylim(0,1)

        # for i = 5:
        axs[2][0].set_ylabel('percentage')
        axs[2][0].bar(labels,C5, color='mediumpurple')
        axs[2][0].set_title('i = 5')
        axs[2][0].set_ylim(0,1)
        
        # for i = 6:
        axs[2][1].set_ylabel('percentage')
        axs[2][1].bar(labels,C6, color='mediumpurple')
        axs[2][1].set_title('i = 6')
        axs[2][1].set_ylim(0,1)

        # for i = 7:
        axs[3][0].set_ylabel('percentage')
        axs[3][0].bar(labels,C7, color='mediumpurple')
        axs[3][0].set_title('i = 7')
        axs[3][0].set_ylim(0,1)
        
        # for i = 8:
        axs[3][1].set_ylabel('percentage')
        axs[3][1].bar(labels,C8, color='mediumpurple')
        axs[3][1].set_title('i = 8')
        axs[3][1].set_ylim(0,1)

        # for i = 9:
        axs[4][0].set_ylabel('percentage')
        axs[4][0].bar(labels,C9, color='mediumpurple')
        axs[4][0].set_title('i = 9')
        axs[4][0].set_ylim(0,1)

        # for i = 10:
        axs[4][1].set_ylabel('percentage')
        axs[4][1].bar(labels,C10, color='mediumpurple')
        axs[4][1].set_title('i = 10')
        axs[4][1].set_ylim(0,1)

        plt.show()

def plot_exp_utility(list):
    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot(1,1,1)
    ax.set_title('Figure 2.2: Expected utility of career choices')
    labels = ['i = 1', 'i = 2', 'i = 3', 'i = 4', 'i = 5', 'i = 6', 'i = 7', 'i = 8', 'i = 9', 'i = 10']
    ax.bar(labels, list, color='mediumpurple', alpha=0.7, label='v1')
    ax.set_ylim(2.5,4.0)
    ax.set_xlabel('Type of graduate')
    ax.set_ylabel('Expected utility')
    plt.show()

def plot_realized_utility(list):
    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot(1,1,1)
    ax.set_title('Figure 2.3: Realized utility of career choices')
    labels = ['i = 1', 'i = 2', 'i = 3', 'i = 4', 'i = 5', 'i = 6', 'i = 7', 'i = 8', 'i = 9', 'i = 10']
    ax.bar(labels, list, color='mediumpurple', alpha=0.7, label='v1')
    ax.set_ylim(2.0,3.0)
    ax.set_xlabel('Type of graduate')
    ax.set_ylabel('Realized utility')
    plt.show()