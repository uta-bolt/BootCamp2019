#======================================================================
#
#     This routine interfaces with the TASMANIAN Sparse grid
#     The crucial part is 
#
#     aVals[iI]=solveriter.iterate(aPoints[iI], n_agents)[0]  
#     => at every gridpoint, we solve an optimization problem
#
#     Simon Scheidegger, 11/16 ; 07/17
#======================================================================
import pdb
import TasmanianSG
import numpy as np
from parameters import *
import nonlinear_solver_iterate as solveriter

#======================================================================

def sparse_grid_iter(n_agents, iDepth, valold):
    
    grid  = TasmanianSG.TasmanianSparseGrid()

    k_range=np.array([k_bar, k_up])

    ranges=np.empty((n_agents, 2))


    for i in range(n_agents):
        ranges[i]=k_range

    iDim=n_agents
    iOut=1

    grid.makeLocalPolynomialGrid(iDim, iOut, iDepth, which_basis, "localp")
    grid.setDomainTransform(ranges)

    aPoints=grid.getPoints()
    iNumP1=aPoints.shape[0]
    aVals=np.empty([iNumP1, len(thetagrid)])
    EV=np.empty([iNumP1, 1])
    file=open("comparison1.txt", 'w')
    for iI in range(iNumP1): #i think this loops through the kgrid
        for tt,theta in enumerate(thetagrid):
            aVals[iI,tt]=solveriter.iterate(aPoints[iI], n_agents, valold,theta)[0]
            #print(aVals[iI,tt],iI,tt)
            v=aVals[iI,tt]*np.ones((1,1))
            to_print=np.hstack((aPoints[iI].reshape(1,n_agents), v))
            #np.savetxt(file, to_print, fmt='%2.16f')
        EV[iI]=0.2*aVals[iI,0]+0.2*aVals[iI,1]+0.2*aVals[iI,2]+0.2*aVals[iI,3]+0.2*aVals[iI,4]
        #print(aVals)
        #pdb.set_trace()
    file.close()
    grid.loadNeededPoints(EV)
    
    f=open("grid_iter.txt", 'w')
    np.savetxt(f, aPoints, fmt='% 2.16f')
    f.close()
    #$print(v) #UB
    #print(aPoints)
    return grid, aVals, aPoints
#======================================================================
