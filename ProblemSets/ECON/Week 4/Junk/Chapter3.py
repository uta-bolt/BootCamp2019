import numpy as np
import scipy.optimize as opt
import time
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, "C:\\Users\\utabo\\Documents\\GitHub\\BootCamp2019\\ProblemSets\\ECON\\Week 4")
import params3 as p

#==================================================================================#
def get_K(b,p):
    K=sum(b)
    return K
def get_L(n,p):
    L=sum(n)
    return L

def get_r(L,K,p):
    r=p.alpha*p.A*(L/K)**(1-p.alpha)-p.delta
    return r

def get_w(L,K,p):
    w=(1-p.alpha)*p.A*(K/L)**(p.alpha)
    return w

def get_C(n,w,r,b,p):
    c=np.zeros(p.S)
    c[0]=w*n[0]-b[0]
    for ss in range(0,p.S-2):
        c[ss+1]=w*n[ss+1]+(1+r)*b[ss]-b[ss+1]
    c[p.S-1]=w*n[p.S-1]+(1+r)*b[p.S-2]
    return c


def margu(c,p):
    marg=c**(-p.sigma)
    return marg

#======================================================================#
#start_time = time.clock() # Place at beginning of get_SS()
def euler(b,n,p):
    k=get_K(b,p)
    l=get_L(n,p)
    r=get_r(l,k,p)
    w=get_w(l,k,p)
    c=get_C(n,w,r,b,p)
    err1=np.empty(p.S-1)
    for s in range(0,p.S-1):
        err1[s]=p.beta*(1+r)*margu(c[s+1],p)-margu(c[s],p)
    return err1

def get_SS(bvec_guess, nvec,p):
    start_time = time.clock()
    b_ss= opt.root(euler, bvec_guess,args=(nvec,p), tol = 1e-10)
    l=get_L(nvec,p)
    k_ss=get_K(b_ss.x,p)
    r_ss=get_r(l,k_ss,p)
    w_ss=get_w(l,k_ss,p)
    c_ss=get_C(nvec,w_ss,r_ss,b_ss.x,p)
    Y_ss=c_ss+p.delta*k_ss
    errs=euler(b_ss.x,nvec,p)
    ss_time = time.clock() - start_time
    ss_output = {
    'b_ss': b_ss.x, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss,
    'K_ss': k_ss, 'Y_ss': Y_ss, 'EulErr_ss': errs,'ss_time': ss_time}
    return ss_output


nvec=np.ones(p.S)
for tt in range(p.S):
    if tt>54:
        nvec[tt]=0.2
print(nvec)
bvec_guess = np.ones(p.S-1)*0.15

ss=get_SS(bvec_guess, nvec,p)
print(ss)


timevec=np.linspace(1,p.S,p.S)
savings_ss=np.append(ss['b_ss'],[0])
plt.figure(0)
plt.plot(timevec,savings_ss,label="savings")
plt.title("savings")
plt.figure(1)
plt.plot(timevec, ss['c_ss'],label="cons")
plt.title("cons")
######################################################
#Early retirement
nvec=np.ones(p.S)
for tt in range(p.S):
    if tt>40:
        nvec[tt]=0.2
print(nvec)
bvec_guess = np.ones(p.S-1)*0.15

ss=get_SS(bvec_guess, nvec,p)
print(ss)


timevec=np.linspace(1,p.S,p.S)
savings_ss=np.append(ss['b_ss'],[0])
plt.figure(0)
plt.plot(timevec,savings_ss,label="savings early ret")
plt.legend()
plt.figure(1)
plt.plot(timevec, ss['c_ss'],label="cons early ret")
plt.legend()
