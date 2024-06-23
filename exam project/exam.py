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
    

    def career_alt(self, careerdict, RVdict):
        par = self.par

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

        for i in range(0,par.N):  
            for n in range(0,par.K):
                EUv1 = self.v1()
                EUv2 = self.v2()
                EUv3 = self.v3()
                if careerdict[i+1][n] == 1:
                    c = np.max([RVdict[i+1][n],(EUv2[i]-par.c), (EUv3[i]-par.c)])
                    if c == RVdict[i+1][n]:
                        careerdict_alt[i+1].append(1)
                        EVdict_alt[i+1].append(RVdict[i+1][n])
                        switchdict[i+1].append(0)
                        RVdict_alt[i+1].append(RVdict[i+1][n])
                    elif c == (EUv2[i]-par.c):
                        careerdict_alt[i+1].append(2)
                        EVdict_alt[i+1].append(EUv2[i]-par.c)
                        switchdict[i+1].append(1)
                        noiseterm = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(2 - par.c + noiseterm[0])
                    else: 
                        careerdict_alt[i+1].append(3)
                        EVdict_alt[i+1].append(EUv3[i]-par.c)
                        switchdict[i+1].append(1)
                        noiseterm = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(3 - par.c + noiseterm[0])
                elif careerdict[i+1][n] == 2:
                    c = np.max([RVdict[i+1][n],(EUv1[i]-par.c), (EUv3[i]-par.c)])
                    if c == RVdict[i+1][n]:
                        careerdict_alt[i+1].append(2)
                        EVdict_alt[i+1].append(RVdict[i+1][n])
                        switchdict[i+1].append(0)
                        RVdict_alt[i+1].append(RVdict[i+1][n])  
                    elif c == (EUv1[i]-par.c):
                        careerdict_alt[i+1].append(1)
                        EVdict_alt[i+1].append(EUv1[i]-par.c)
                        switchdict[i+1].append(1)
                        noiseterm = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(1 - par.c + noiseterm[0])
                    else:
                        careerdict_alt[i+1].append(3)
                        EVdict_alt[i+1].append(EUv3[i]-par.c)
                        switchdict[i+1].append(1)
                        noiseterm = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(3 - par.c + noiseterm[0])
                else:
                    c = np.max([RVdict[i+1][n],(EUv1[i]-par.c), (EUv2[i]-par.c)])
                    if c == RVdict[i+1][n]:
                        careerdict_alt[i+1].append(3)
                        EVdict_alt[i+1].append(RVdict[i+1][n])
                        switchdict[i+1].append(0)
                        RVdict_alt[i+1].append(RVdict[i+1][n])
                    elif c == (EUv1[i]-par.c):
                        careerdict_alt[i+1].append(1)
                        EVdict_alt[i+1].append(EUv1[i]-par.c)
                        switchdict[i+1].append(1)
                        noiseterm = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(1 - par.c + noiseterm[0])
                    else:
                        careerdict_alt[i+1].append(2)
                        EVdict_alt[i+1].append(EUv2[i]-par.c)
                        switchdict[i+1].append(1)
                        noiseterm = np.random.normal(loc=0, scale=par.sigma, size=1)
                        RVdict_alt[i+1].append(2 - par.c + noiseterm[0])
        
        return switchdict, careerdict_alt, EVdict_alt, RVdict_alt
            
    def sort_career(self, switchdict, careerdict):
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
    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot(1,1,1)
    ax.set_title(title)
    labels = ['i = 1', 'i = 2', 'i = 3', 'i = 4', 'i = 5', 'i = 6', 'i = 7', 'i = 8', 'i = 9', 'i = 10']
    ax.bar(labels, list, color='mediumpurple', alpha=0.7, label='v1')
    ax.set_ylim(2.5,4.0)
    ax.set_xlabel('Type of graduate')
    ax.set_ylabel('Expected utility')
    plt.show()


def plot_realized_utility(list, title):
    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot(1,1,1)
    ax.set_title(title)
    labels = ['i = 1', 'i = 2', 'i = 3', 'i = 4', 'i = 5', 'i = 6', 'i = 7', 'i = 8', 'i = 9', 'i = 10']
    ax.bar(labels, list, color='mediumpurple', alpha=0.7, label='v1')
    ax.set_ylim(2.0,3.4)
    ax.set_xlabel('Type of graduate')
    ax.set_ylabel('Realized utility')
    plt.show()    


class Barycentric:
    #Defining a function for the barycentric coordinates:
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

    def check_coor_tri(A, B, C, D, y):
        #Including the two triangles in the coordinates
        r_1_ABC, r_2_ABC, r_3_ABC = barycentric_c(A, B, C, y)
        r_1_CDA, r_2_CDA, r_3_CDA = barycentric_c(C, D, A, y)

        #Finally confirming if y is in the triangles
        if bary_in_tri(r_1_CDA, r_2_CDA, r_3_CDA):
            f_y = r_1_CDA * f(C[0], C[1]) + r_2_CDA * f(D[0], D[1]) + r_3_CDA * f(A[0], A[1])
            return r_1_CDA, r_2_CDA, r_3_CDA, r_1_ABC, r_2_ABC, r_3_ABC, f"Y is inside CDA. f(y) = {f_y}"
        elif bary_in_tri(r_1_ABC, r_2_ABC, r_3_ABC):
            f_y = r_1_ABC * f(A[0], A[1]) + r_2_ABC * f(B[0], B[1]) + r_3_ABC * f(C[0], C[1])
            return r_1_CDA, r_2_CDA, r_3_CDA, r_1_ABC, r_2_ABC, r_3_ABC, f"y is inside ABC. f(y) = {f_y}"
        else:
            return r_1_CDA, r_2_CDA, r_3_CDA, r_1_ABC, r_2_ABC, r_3_ABC, "y isn't in the triangles. Undefined f(y)"