
import time

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
# globals
A = 1
alpha = 0.35
delta = 0.6415
beta = 0.442
sigma = 3
SS_tol = 1e-5
labor_supply = np.array([1.0, 1.0, 0.2])
b_initial = 0
# package globals as parameters
params = (b_initial, beta, sigma, labor_supply, A, alpha, delta, SS_tol)
bvec_guess = np.array([0.1, 0.1])
# Functions that define the problem
def production(K,L, A=1, alpha=0.35):
    """
    Cobb-Douglas production function
    """
    Y = A * (K ** alpha) * (L ** (1-alpha)) # production
    return Y

def find_r(L, K, alpha=0.35, A=1, delta=0.6415):
    """
    Finds the interest rate given paramters
    """
    r = (alpha * A * ((L / K) ** (1 - alpha))) - delta
    return r

def find_w(L, K, alpha=0.35, A=1):
    """
    Finds the wage given paramters
    """
    w = (1 - alpha) * A * ((K / L) ** alpha)
    return w

def u_prime(c, sigma=3):
    """
    CRRRA marginal utility with parameter sigma
    """
    if c == 0:
        return np.inf
    else:
        return c ** (-sigma)
def ee_errors(bvec_guess, *args):
    """
    Compute vector (s-1 x 1) euler equation error
    We will try and find root of this to find optimal consumption
    """
    b_intial, beta, sigma, labor_supply, A, alpha, delta, SS_tol = args
    # Impose firm optimality
    K = b_initial + bvec_guess.sum()
    L = labor_supply.sum()
    # get prices
    r = find_r(L, K, alpha=alpha, A=A, delta=delta)
    w = find_w(L, K, alpha=alpha, A=A)
    # HH's problem
    wages = w * labor_supply
    savings = np.array([b_initial, *bvec_guess])
    sav_tomorrow = np.roll(savings, len(savings)-1)
    cons = (1 + r) * savings + wages - sav_tomorrow
    EulErr_ss = np.array([beta * (1+r) * u_prime(cons[i+1]) - u_prime(cons[i]) for i in range(len(savings)-1)])
    return EulErr_ss





def get_SS(params=params, bvec_guess=bvec_guess, SS_graphs=False):
    start_time = time.clock()

    # unpack
    b_intial, beta, sigma, labor_supply, A, alpha, delta, SS_tol = params

    # find SS
    results = opt.root(ee_errors, bvec_guess, args=params)

    # ss savings
    b_ss = np.array([*results.x, 0])

    # ss market characteristics
    K_ss = b_ss.sum()
    L_ss = labor_supply.sum()
    Y_ss = production(K_ss, L_ss, A=A, alpha=alpha)

    # ss prices
    r_ss = find_r(L_ss, K_ss, alpha=alpha, A=A, delta=delta)
    w_ss = find_w(L_ss, K_ss, alpha=alpha, A=A)

    # ss consumption
    asset_income = np.roll(b_ss, len(b_ss)-1) # rotates vector from savings decision to assett income
    c_ss = ((1 + r_ss) * asset_income) + (w_ss * labor_supply) - b_ss
    C_ss = c_ss.sum()

    # computation errors
    EulErr_ss = results.fun
    RCerr_ss = Y_ss - C_ss - delta*K_ss  # resource constraint error

    # timing
    ss_time = time.clock() - start_time

    ss_output = {
        'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss,
        'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
        'EulErr_ss': EulErr_ss, 'RCerr_ss': RCerr_ss,
        'ss_time': ss_time}
    # plotting
    if SS_graphs:
        fig, axes = plt.subplots(2,1,figsize=(10,6))

        axes[0].plot(c_ss, 'r-o')
        axes[0].set_title('Steady state consumption')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Consumption')

        axes[1].plot(b_ss, 'b-o')
        axes[1].set_title('Steady state savings')
        axes[1].set_xlabel('Age')
        axes[1].set_ylabel('Savings')

        [ax.set_xticks(range(len(bvec_guess)+1)) for ax in axes]

        plt.tight_layout()

    return ss_output
get_SS(SS_graphs=True)




# globals
A = 1
alpha = 0.35
delta = 0.6415
beta = 0.442
sigma = 3
SS_tol = 1e-5
labor_supply = np.array([1.0, 1.0, 0.2])
b_initial = 0
# package globals as parameters
params = {'beta': beta,
          'sigma': sigma,
          'labor_supply': labor_supply,
          'A': A,
          'alpha': alpha,
          'delta': delta,
          'SS_tol': SS_tol}

def ee_err_1(sav, *args):
    """
    Compute errors for the middle aged person to solve for their savings
    """
    w, r, labor_supply, b_init, s, t = args

    error = u_prime(w[t] * labor_supply[s] + (1 + r[t]) * b_init[s] - sav) - beta * (1 + r[t+1]) * u_prime((1 + r[t+1]) * sav + w[t+1] * labor_supply[s+1])

    return error

def ee_err_23(sav, *args):
    w, r, labor_supply, b_init, s, t = args

    error = np.zeros(2)
    # sav[0] = b22, sav[1]=b33
    error[0] = u_prime(w[t] - sav[0]) - beta * (1 + r[t+1]) * u_prime(w[t+1] + (1 + r[t+1]) * sav[0] - sav[1])
    error[1] = u_prime(w[t+1] + (1 + r[t+1]) * sav[0] - sav[1]) - beta * (1 + r[t+2]) * u_prime(labor_supply[s+1] * w[t+1] + (1 + r[t+2]) * sav[1])
    return error


def time_path_iteration(params=params, S=3, T=50, weight=0.3, tol=1e-12, maxiter=100):
    """
    Computes transition path of Agg Capital for OLG
    """
    ss_output = get_SS()
    b_ss = ss_output['b_ss']
    b_init = np.array([0, 0.8 * b_ss[0], 1.1 * b_ss[1]]) # t=0

    # Guess transition path, finishes at steady_state
    Kguess = np.linspace(b_init.sum(), ss_output['K_ss'], T)

    s = 1
    K_dynamic = Kguess
    b_current = np.zeros((S,T))  # initialize array to store savings decisions
    b_current[:,0] = b_init

    # Update b_path until convergence
    its = 0
    ee_diff = 7.0
    while ee_diff > tol and its < maxiter:
        its += 1
        w_dynamic = find_w(L=params['labor_supply'].sum(), K=K_dynamic)
        r_dynamic = find_r(L=params['labor_supply'].sum(), K=K_dynamic)
        for t in range(T-2):

            #solve for b32, savings decision of middle-aged in first period
            ee_param = (w_dynamic, r_dynamic, params['labor_supply'], b_current[:,t], s, t)
            b_current[s+1,t+1] = opt.root(ee_err_1, 0, args=ee_param).x
            # solve for b22, b33, savings decision of young gen in middle/old generations
            ee_param = (w_dynamic, r_dynamic, params['labor_supply'], b_init, s, t)
            b_current[s,t+1], b_current[s+1, t+2]= opt.root(ee_err_23, [0,0], args=ee_param).x
            # fill in table
            b_current[s,T-1] = b_current[s,T-2]

        # Check for convergence
        K_prime = b_current.sum(axis=0)
        print(K_prime)
        print(K_prime.shape)
        print(K_dynamic.shape)
        ee_diff = (K_prime - K_dynamic).max()

#         rc_diff = production(K_prime, L=params['labor_supply'].sum())
#         - Ct = (1 + r_dynamic) * ()
#         - np.roll(K_prime, len(K_prime)-1)
#         - (1 - delta) * K_prime

        print('Iteration number: ', its, 'Current EE difference: ', ee_diff)
        # update new capital path
        K_dynamic = weight * K_prime + (1-weight) * K_dynamic

    fig, ax  = plt.subplots(1,1,figsize=(8,6))
    plt.plot(range(T), Kguess, 'r--',lw=0.7, label='Kguess')
    plt.plot(range(T), K_dynamic , label='Capital Path Solution')
    plt.title('Transition Path of Aggregate Capital')
    plt.xlabel('Time period')
    plt.ylabel('Aggregate Capital')
    plt.legend()

    fig, ax  = plt.subplots(1,1,figsize=(8,6))
    plt.plot(range(T), r_dynamic, 'g-o',label='Interest rate Path Solution')
    plt.title('Transition Path of Aggregate Interest rate')
    plt.xlabel('Time period')
    plt.ylabel('Interest Rate')
    plt.legend()

    fig, ax  = plt.subplots(1,1,figsize=(8,6))
    plt.plot(range(T), w_dynamic, 'k-o',label='Wage Path Solution')
    plt.title('Transition Path of Wages')
    plt.xlabel('Time period')
    plt.ylabel('Wages')
    plt.legend()

    return K_dynamic, its

K_dynamic,its = time_path_iteration()
print(its)
