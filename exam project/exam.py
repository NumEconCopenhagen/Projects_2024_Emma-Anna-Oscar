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

    
    def firm1(self, p1, w):
        par = self.par
        # Defining labor for firm 1 with inputs p1 and w and given parameters
        l1 = (p1*par.A * par.gamma/w) ** (1/(1-par.gamma))
        y1 = par.A * (l1 ** par.gamma)
        
        return l1, y1
    
    def imp_profit1(self, p1, w):
        par = self.par
        # Defining the implied profits for firm 1 with inputs p1 and w and given parameters
        pi1 = ((1-par.gamma)*w/par.gamma) * (p1*par.A*par.gamma/w)**(1/(1-par.gamma))

        return pi1
    
    def firm2(self, p2, w):
        par = self.par
        # Defining labor for firm 2 with inputs p2 and w and given parameters
        l2 = (p2*par.A * par.gamma/w) ** (1/(1-par.gamma))
        y2 = par.A * (l2 ** par.gamma)
        
        return l2, y2
    
    def imp_profit2(self, p2, w):
        par = self.par
        # Defining the implied profits for firm 2 with inputs p2 and w and given parameters
        pi2 = ((1-par.gamma)*w/par.gamma) * (p2*par.A*par.gamma/w)**(1/(1-par.gamma))

        return pi2

    
    def consumer_behavior(self,p1,p2,w):
        ''' Defining the consumer's behavior, given prices p1, p2, and wage w '''
        par = self.par
        pi1 = self.imp_profit1(p1,w)
        pi2 = self.imp_profit2(p2,w)

        def utility(l):
            ''' Defining the utility function '''
            c1 = par.alpha * (w*l + par.T + pi1 + pi2) / p1
            c2 = (1-par.alpha) * (w*l + par.T + pi1 + pi2) / (p2 + par.tau)
            return np.log(c1**par.alpha * c2**(1-par.alpha)) - par.nu * l**(1+par.epsilon) / (1+par.epsilon)
        
        sol = optimize.minimize(lambda l: -utility(l), x0=0, method='SLSQP', bounds=[(0,None)])

        l_star = sol.x[0]
        c1_star = par.alpha * (w*l_star + par.T + pi1 + pi2) / p1
        c2_star = (1-par.alpha) * (w*l_star + par.T + pi1 + pi2) / (p2 + par.tau)

        return l_star, c1_star, c2_star

    
    def market_error(self,p1,p2,w):
        l_star, c1_star, c2_star = self.consumer_behavior(p1,p2,w)
        l1_star, y1_star = self.firm1(p1,w)
        l2_star, y2_star = self.firm2(p2,w)

        exc_labor = l_star - l1_star - l2_star
        exc_good1 = c1_star - y1_star
        exc_good2 = c2_star - y2_star

        return exc_labor, exc_good1, exc_good2
    
    def check_market_clearing(self, p1_values, p2_values, w):

        for p1 in p1_values:
            for p2 in p2_values:
                try:
                    l_star, c1_star, c2_star = self.consumer_behavior(p1, p2, w)
                    l1_star, y1_star = self.firm1(p1, w)
                    l2_star, y2_star = self.firm2(p2, w)

                    labor_market_clearing = np.isclose(l_star, l1_star + l2_star)
                    good1_market_clearing = np.isclose(c1_star, y1_star)
                    good2_market_clearing = np.isclose(c2_star, y2_star)
                    
                except Exception as e:
                    print(f"Error for p1={p1}, p2={p2}: {e}")
                    continue

        if labor_market_clearing and good1_market_clearing and good2_market_clearing:
            print(f'For p1={p1:2f} and p2={p2:.2f}, does the markets clear?\n labor: {labor_market_clearing}, good1: {good1_market_clearing}, good2: {good2_market_clearing}\n')
        else:
            print(f'Found no combination of p1 and p2, which clears all three markets.')

        return
    
    def find_equilibrium_prices(self, w):
        par = self.par
        initial_guess = [0.1, 0.1]

        def obj_p(prices):
            p1, p2 = prices
            # We use Walras' Law, meaning we only neew to clear two of the markets, to find the equilibrium prices
            exc_labor, exc_good1 = self.market_error(p1, p2, w)[0], self.market_error(p1, p2, w)[1]
            total_excess = exc_labor, exc_good1
            return total_excess
        
        print(f'Using Walras Law, clearing the markets for labor and good 1, to find equilibrium prices for a given wage w={w}')
        print(f'Initial guess of (p1, p2): {initial_guess}.\n Initial excess demand for labor: {obj_p(initial_guess)[0]:.3f}, good 1: {obj_p(initial_guess)[1]:.3f}\n')
        print(f'Finding equilibrium prices...\n ...\n')
        
        try:
            result = optimize.root(obj_p, initial_guess, method='hybr')
            if result.success:
                np.set_printoptions(precision=5)
                print(f'Convergence succesful, found equilibrium prices: p1 = {result.x[0]:.3f}, p2 = {result.x[1]:.3f}')
                print(f'Excess demand for: labor = {obj_p(result.x)[0]:.5f}, good 1 = {obj_p(result.x)[1]:.5f}')
                return result.x
            elif not result.success:
                print(f'Did not converge: {result.message}')
        except Exception as e:
            print(f"Failed to find equilibrium prices {e}.")
        
        return result.x

    def SWF(self,p1,p2,w,tau):
        par = self.par
        l_star, c1_star, c2_star = self.consumer_behavior(p1, p2, w)
        y2_star = self.firm2(p2,w)[1]
        pi1_star = self.imp_profit1(p1,w)
        pi2_star = self.imp_profit2(p2,w)
        T = tau*c2_star

        utility = np.log(c1_star**par.alpha * c2_star**(1-par.alpha)) - par.nu * l_star**(1+par.epsilon) / (1+par.epsilon)
        constraint = {'type' : 'eq', 'fun' : lambda tau: p1*c1_star + (p2+tau)*c2_star - w*l_star - T - pi1_star - pi2_star}
        
        obj_soc = -(utility - par.kappa * y2_star)

        social_sol = optimize.minimize(obj_soc, x0=0, constraints=constraint, method='SLSQP')

        return social_sol.x

    
class CareerChoice:
    def __init__(self, seed=None):
        '''Initializes the parameters of the model'''
        par = self.par = SimpleNamespace()
        par.J = 3
        par.N = 10
        par.K = 10000
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
        '''Calculates the expected utility of career choice v1 for each type of graduate'''
        par = self.par
        # creating an empty list to store the expected utility for each type of graduate
        EU = []
        # creating a for loop to loop over each type of graduate
        for i in range(1,par.N+1):
            # drawing noise terms from the given normal distribution
            # the size of the noise term is equal to numer of friends for each graduate
            eps = np.random.normal(loc=0, scale=par.sigma, size=i)
            eu = 1/i*(1*i + eps.sum())
            EU.append(eu)
        return EU
    
    def v2(self):
        '''Calculates the expected utility of career choice v2 for each type of graduate'''
        # same approach as for v1
        par = self.par
        EU = []
        for i in range(1,par.N+1):
            eps = np.random.normal(loc=0, scale=par.sigma, size=i)
            eu = 1/i*(2*i + eps.sum())
            EU.append(eu)
        return EU 

    def v3(self):
        '''Calculates the expected utility of career choice v3 for each type of graduate'''
        # same approach as for v1
        par = self.par
        EU = []
        for i in range(1,par.N+1):
            eps = np.random.normal(loc=0, scale=par.sigma, size=i)
            eu = 1/i*(3*i + eps.sum())
            EU.append(eu)
        return EU
    
    def career(self):
        '''Looping over each type of graduate end sorting them into the career with the highest expected utility.'''
        par = self.par
        # calling the expected utility functions for each career choice, getting 10 random instances for each
        # type of graduate
        EUv1 = self.v1()
        EUv2 = self.v2()
        EUv3 = self.v3()

        #creating empty lists to store the results in the correct order
        career = []
        EV = []
        noiseterm = []
        RV = []

        # looping over each type of graduate
        for i in range(0,par.N):
            # choosing the career with the highest expected utility for each type of graduate
            choice = np.max([EUv1[i], EUv2[i], EUv3[i]])
            # if the choice is v1, add v1 = 1 to carrer list as well as the associated expected utility and a new noise term
            # that needs to be used for the realized utility of choosing the given career
            if choice == EUv1[i]:
                career.append(1)
                EV.append(EUv1[i])
                noise = np.random.normal(loc=0, scale=par.sigma, size=1)
                noiseterm.append(noise[0])
            # in the same manner, proceed if the choice is v2 or v3
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
        
        # looping over each type of graduate, we calculate the realized utility by adding the noise term
        # to the base value, v_j, associated with the career they chose
        for i in range(0,par.N):
            RV.append(career[i] + noiseterm[i])
        return career, EV, RV
    
    def simulate(self):
        '''Simulates the career choices for each type of graduate for K instances'''
        par = self.par

        # creating empty dictionaries of career choice, expected utility and realized utility to store the results in correct order
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

        # looping over each type of graduate and simulating the career choice for K instances
        # for each loop, adding the K results to the corresponding dictionaries
        for i in range(0,par.N):
            n = 0
            while n <= par.K:
                career, EV, RV = self.career()
                careerdict[i+1].append(career[i])
                EVdict[i+1].append(EV[i])
                RVdict[i+1].append(RV[i])
                n += 1

        return careerdict, EVdict, RVdict   

    def career_alt(self, careerdict, RVdict):
        '''Simulates the career choices for each type of graduate for K instances, but with the possibility of switching career'''
        par = self.par
        
        # creating empty lists to sort and store correctly
        switchdict = {
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

        careerdict_alt = {
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

        EVdict_alt = {
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

        RVdict_alt = {
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

        # looping over the 10,000 instances previously simulated for each of the 10 types of graduates 
        for i in range(0,par.N):    
            for n in range(0,par.K):
                # after the graduates have been in their career for a year and the realized utility is known, they get the option
                # to switch to a different career. If they do this, they draw three new noisy signals for the two careers they did not pick
                EUv1 = self.v1()
                EUv2 = self.v2()
                EUv3 = self.v3()

                # if the graduate chose career v1 originally, they must decide between sticking to their career or paying c = 1 to switch to one of the
                # other careers based on their new drae of noisy signls. The same goes for the agents that picked the other two careers originally.
                # for each choice, the corresponding values are appended into their respective dictionaries
                # if the graduate chooses to switch, the variable added to switch will be a 1. If not, it will be a 0.
                if careerdict[i+1][n] == 1:
                    c = np.max([RVdict[i+1][n],(EUv2[i]-1), (EUv3[i]-1)])
                    if c == RVdict[i+1][n]:
                        careerdict_alt[i+1].append(1)
                        EVdict_alt[i+1].append(RVdict[i+1][n])
                        switchdict[i+1].append(0)
                        RVdict_alt[i+1].append(RVdict[i+1][n])
                    elif c == (EUv2[i]-1):
                        careerdict_alt[i+1].append(2)
                        EVdict_alt[i+1].append(EUv2[i]-1)
                        switchdict[i+1].append(1)
                        noise = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(2 - 1 + noise)
                    else: 
                        careerdict_alt[i+1].append(3)
                        EVdict_alt[i+1].append(EUv3[i]-1)
                        switchdict[i+1].append(1)
                        noise = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(3 - 1 + noise)
                elif careerdict[i+1][n] == 2:
                    c = np.max([RVdict[i+1][n],(EUv1[i]-1), (EUv3[i]-1)])
                    if c == RVdict[i+1][n]:
                        careerdict_alt[i+1].append(2)
                        EVdict_alt[i+1].append(RVdict[i+1][n])
                        switchdict[i+1].append(0)
                        RVdict_alt[i+1].append(RVdict[i+1][n])  
                    elif c == (EUv1[i]-1):
                        careerdict_alt[i+1].append(1)
                        EVdict_alt[i+1].append(EUv1[i]-1)
                        switchdict[i+1].append(1)
                        noise = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(1 - 1 + noise)
                    else:
                        careerdict_alt[i+1].append(3)
                        EVdict_alt[i+1].append(EUv3[i]-1)
                        switchdict[i+1].append(1)
                        noise = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(3 - 1 + noise)
                else:
                    c = np.max([RVdict[i+1][n],(EUv1[i]-1), (EUv2[i]-1)])
                    if c == RVdict[i+1][n]:
                        careerdict_alt[i+1].append(3)
                        EVdict_alt[i+1].append(RVdict[i+1][n])
                        switchdict[i+1].append(0)
                        RVdict_alt[i+1].append(RVdict[i+1][n])
                    elif c == (EUv1[i]-1):
                        careerdict_alt[i+1].append(1)
                        EVdict_alt[i+1].append(EUv1[i]-1)
                        switchdict[i+1].append(1)
                        noise = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(1 - 1 + noise)
                    else:
                        careerdict_alt[i+1].append(2)
                        EVdict_alt[i+1].append(EUv2[i]-1)
                        switchdict[i+1].append(1)
                        noise = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(2 - 1 + noise)
        
        return switchdict, careerdict_alt, EVdict_alt, RVdict_alt
            
    def sort_career(self, switchdict, careerdict):
        '''Sorting the binary switch-variable based on which original career was chosen'''
        par = self.par

        v1_original = {
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

        v2_original = {
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

        v3_original = {
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
        # looping over each graduate's 10,000 instances and sorting the square matrix of career choices based on the original career choice
        for i in range(0,par.N):
            for n in range(0,par.K):
                if careerdict[i+1][n] == 1:
                    if switchdict[i+1][n] == 1:
                        v1_original[i+1].append(1)
                    else:
                        v1_original[i+1].append(0)
                elif careerdict[i+1][n] == 2:
                    if switchdict[i+1][n] == 1:
                        v2_original[i+1].append(1)
                    else:
                        v2_original[i+1].append(0)
                else:
                    if switchdict[i+1][n] == 1:
                        v3_original[i+1].append(1)
                    else:
                        v3_original[i+1].append(0)

        return v1_original, v2_original, v3_original

def count(list):
    '''This function counts the share of times the number 1, 2 or 3 appears in a list'''
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

def count2(list):
    '''This function counts the share of times the number 1 appears in a list'''
    count0 = 0
    count1 = 0

    for element in list:
        if element == 0:
            count0 += 1
        else:
            count1 += 1
    
    switch = count1/len(list)
    return switch

def plot_switch(s_v1_original, s_v2_original, s_v3_original):
    '''This function plots the share of graduates switching career for each type of graduate for the three different career choices'''
    # creating a figure that will have 3 rows of subplots
    fig, axs = plt.subplots(nrows=3, figsize=(8,12), dpi=100, constrained_layout=True)
    fig.suptitle('Figure 2.4: Share of graduates switching career', fontsize=16)
    labels = ['i = 1', 'i = 2', 'i = 3', 'i = 4', 'i = 5', 'i = 6', 'i = 7', 'i = 8', 'i = 9', 'i = 10']

    # for v1 as the original choise:
    axs[0].set_title('Original career choice: v1')
    axs[0].set_ylabel('Share of graduates switching career')
    axs[0].set_xlabel('Type of graduate')
    axs[0].bar(labels, s_v1_original, color='mediumpurple')
    axs[0].set_ylim(0,1)

    # for v2 as the original choise:    
    axs[1].set_title('Original career choice: v2')
    axs[1].set_ylabel('Share of graduates switching career')
    axs[1].set_xlabel('Type of graduate')
    axs[1].bar(labels, s_v2_original, color='mediumpurple')
    axs[1].set_ylim(0,1)

    # for v3 as the original choise:
    axs[2].set_title('Original career choice: v3')
    axs[2].set_ylabel('Share of graduates switching career')
    axs[2].set_xlabel('Type of graduate')
    axs[2].bar(labels, s_v3_original, color='mediumpurple')
    axs[2].set_ylim(0,1)

    plt.show()

def plot_exp_utility(list,title):
    '''This function plots the average expected utility for each type of graduate for the three different career choices'''
    # the function takes in the list of average expected utilities for each i, as well as a title so the plot can be used for 
    # both the expected utility before and after the potential career switch
    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot(1,1,1)
    ax.set_title(title)
    labels = ['i = 1', 'i = 2', 'i = 3', 'i = 4', 'i = 5', 'i = 6', 'i = 7', 'i = 8', 'i = 9', 'i = 10']
    ax.bar(labels, list, color='mediumpurple')
    ax.set_ylim(2.5,4.0)
    ax.set_xlabel('Type of graduate')
    ax.set_ylabel('Expected utility')
    plt.show()

def plot_realized_utility(list, title):
    '''This function plots the average realized utility for each type of graduate for the three different career choices'''
    # like the plot_exp_utility function, this funtion has a variable title so it can be used before and after the potential caareer switch
    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot(1,1,1)
    ax.set_title(title)
    labels = ['i = 1', 'i = 2', 'i = 3', 'i = 4', 'i = 5', 'i = 6', 'i = 7', 'i = 8', 'i = 9', 'i = 10']
    ax.bar(labels, list, color='mediumpurple')
    ax.set_ylim(2.0,3.0)
    ax.set_xlabel('Type of graduate')
    ax.set_ylabel('Realized utility')
    plt.show()

def findABCD(rng,X,y):
    A = min((d for d in X if d[0] > y[0] and d[1] > y[1]), key=lambda d: np.sqrt((d[0] - y[0])**2 + (d[1] - y[1])**2))
    B = min((d for d in X if d[0] > y[0] and d[1] < y[1]), key=lambda d: np.sqrt((d[0] - y[0])**2 + (d[1] - y[1])**2))
    C = min((d for d in X if d[0] < y[0] and d[1] < y[1]), key=lambda d: np.sqrt((d[0] - y[0])**2 + (d[1] - y[1])**2))
    D = min((d for d in X if d[0] < y[0] and d[1] > y[1]), key=lambda d: np.sqrt((d[0] - y[0])**2 + (d[1] - y[1])**2))
    return A, B, C, D

def barycentric_c(A, B, C, y):
    #First, we define the denominator to make the calculations more simple for r_1 and r_2
    denominator = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    r_1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denominator
    r_2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denominator
    r_3 = 1 - r_1 - r_2
    return r_1, r_2, r_3

#Next step is to find out if the Bary coordinates are within the triangles
def bary_in_tri(r_1, r_2, r_3):
    return (0<= r_1 <= 1) and (0<= r_2 <= 1) and (0<= r_3 <= 1)

def f(x1, x2):
    return x1 + x2

class Barycentric():
    #Defining a function for the barycentric coordinates:
    def __init__(self,rng,X,y,A,B,C,D):
        self.rng = rng
        self.X = X
        self.y = y
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def plotABCD(self):
        #Plotting our dots and triangles
        plt.figure(figsize=(6, 6))
        #plotting points for X
        plt.scatter(self.X[:, 0], self.X[:, 1], c='grey', label='Datapoints for X, random')
        #Plotting y point
        plt.scatter(self.y[0], self.y[1], c='magenta', label='y', marker='o')
        
        #Plotting points for A, B, C and D
        points = [self.A, self.B, self.C, self.D]
        labels = ['A', 'B', 'C', 'D']
        colors = ['tab:blue', 'mediumpurple', 'blueviolet', 'violet']
        # Separating the x and y values
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]

        for i in range(len(points)):
            plt.scatter(x_values[i], y_values[i], color=colors[i], label=labels[i])

        #Making the triangles
        plt.plot([self.A[0], self.B[0], self.C[0], self.A[0]], [self.A[1], self.B[1], self.C[1], self.A[1]], c='mediumblue', ls='-.', label='ABC')
        plt.plot([self.C[0], self.D[0], self.A[0], self.C[0]], [self.C[1], self.D[1], self.A[1], self.C[1]], c='mediumpurple', ls='-.', label='CDA')

        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Figure 3.1: Barycentric Interpolation')
        plt.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.55,1.0))
        plt.grid(linestyle='--', linewidth=0.5, color='lightgrey')
        plt.show()  

    def plotABCD_alt(self,Y):
        #Plotting our dots and triangles
        plt.figure(figsize=(6, 6))
        #plotting points for X
        plt.scatter(self.X[:, 0], self.X[:, 1], c='grey', label='Datapoints for X, random')
        #Plotting y point
        plt.scatter(self.y[0], self.y[1], c='magenta', label='y', marker='o')

        #Plotting points for A, B, C and D
        points = [self.A, self.B, self.C, self.D]
        labels = ['A', 'B', 'C', 'D']
        colors = ['tab:blue', 'mediumpurple', 'blueviolet', 'violet']
        # Separating the x and y values
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]

        for i in range(len(points)):
            plt.scatter(x_values[i], y_values[i], color=colors[i], label=labels[i])

        plt.plot([self.A[0], self.B[0], self.C[0], self.A[0]], [self.A[1], self.B[1], self.C[1], self.A[1]], c='mediumblue', ls='-.', label='ABC')
        plt.plot([self.C[0], self.D[0], self.A[0], self.C[0]], [self.C[1], self.D[1], self.A[1], self.C[1]], c='mediumpurple', ls='-.', label='CDA')

        #Plotting all coordinates in Y
        for coordinate in Y:
            plt.scatter(coordinate[0], coordinate[1], c='magenta', marker='o', label='coordinate in Y')

        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Figure 3.2: Barycentric Interpolation expanded')
        plt.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.55,1.0))
        plt.grid(linestyle='--', linewidth=0.5, color='lightgrey')
        plt.show()