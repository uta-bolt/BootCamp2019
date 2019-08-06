import numpy as np
beta=0.96
delta=0.06
alpha=0.36
numptsk=200
kgrid=np.linspace(0.001,40,numptsk)
epsgrid=np.array([0,1])
maxiter=40
tol=10**(-5)
numsim=10000
tol_outer=10**(-4)