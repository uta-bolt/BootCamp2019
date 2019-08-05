import numpy as np
import random
import scipy.optimize as opt
import scipy.interpolate as int
import pdb
import sys

sys.path.insert(0, "C:\\Users\\utabo\\Documents\\GitHub\\BootCamp2019\\ProblemSets\\ECON\\Week 5")
import parameters as p
#====================================================#
#Problem set - Tony Smith
#====================================================#

#Get interest rate
def get_r(kbar,u,p):
    r=p.alpha*kbar**(p.alpha-1)*(1-u)**(1-p.alpha)
    return r

#get wage rate
def get_k(kbar,u,p):
    w=(1-p.alpha)*kbar**p.alpha*(1-u)**(-p.alpha)
    return w
def margu(c):
    marg=1/c
    return marg
#euler equation:
def utility(x):
    if x>0:
        utils=np.log(x)
    else:
        utils=-10**7
        pdb.set_trace()
    return utils


def negV(kprime,Vcont,eps,p,tt,r,w,kk) :
    Vprime=np.empty([2,1])
    #Vprime[0,0]=np.interp(p.kgrid,Vcont[:,0], kprime) #good
    if tt>0:
        print("round 2")
        pdb.set_trace()
        Vprime[0,0]=int.interp1d(p.kgrid,Vcont[:,0], kprime) #good
        Vprime[1,0]=int.interp1d(p.kgrid,Vcont[:,1], kprime)#bad

    else:
        Vprime=np.array([0][0])
    c=(1+r-p.delta)*kk+w*eps-kprime
    EV=sum(Vprime*p.pi) #expected value given transition probs
    value=-1*(utility(c)+p.beta*EV) #negative value of choosing kprime
    print(value)
    return value

#Problem that the consumer solves given kbar:
def vfi(kbar,p):
    pf=np.empty([p.numptsk, 2])
    vopt=np.empty([p.numptsk, 2])
    r=get_r(kbar,u,p)
    print(r)
    w=get_k(kbar,u,p)
    print(w)
    for tt in range(0,p.maxiter):
        for ixe, eps in enumerate(p.epsgrid):
            for ixk, kk in enumerate(p.kgrid):
                res= opt.minimize_scalar(negV,bracket=(0.00001,40.0),args=(Vcont,eps,p,tt,r,w,kk), method='Golden' )
                vopt[ixk,ixe] =-res.fun
                pf[ixk,ixe] =res.x
        diff=((vopt-Vcont) ** 2).sum()
        print(vopt)
        pdb.set_trace()
        if diff>p.tol:
            vopt=Vcont
        else:
            print("convergence achieved")
            epsseries=np.zeros(p.numsim)
            indk=np.zeros(p.numsim)
            #once we found a solution we want to simulate:
            np.random.seed(23423948)
            rv=np.random.uniform(0,1,p.numsim)
            epsseries=(rv>0.9)
            #for x in range(p.numsim):
            #    epsseries[x]=random.randint(0,1)
            kss=0.001
            for ixN, epsilon in enumerate(epsseries):
                print(epsseries)
                print(pf[:,0])
                print(pf[:,epsilon])
                indk[ixN]=int.interp1d(p.kgrid, pf[:,epsilon],kss)
                #indk[ixN]=np.interp(p.kgrid, pf[:,epsilon],kss)
                kss=np.mean(indk)
            break
    return vopt, pf,kss


def aiyagari(kbar,p):
    for j in range(0,p.maxiter):
        vopt, pf,kss=vfi(kbar,p)
        if abs(kss-kbar)>p.tol:
            kss=kbar
        else:
            print("convergence achieved")
            break
    return kss
#
# define k grid

u=0.1
#initialize Vcont
Vcont=np.zeros([p.numptsk,2])
aiyagari(1,p)
