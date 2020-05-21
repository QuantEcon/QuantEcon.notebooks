
"""
Created on Tue Jun 25 16:50:32 2013

@author: dgevans
"""

from numpy import *
from primitives import primitives_CRRA
import bellman
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

#initialize the parameters
Para = primitives_CRRA()
r = 0.9995*(1/Para.beta-1)

#create the grid
agrid = hstack((linspace(Para.a_min,Para.a_min+1.,15),linspace(Para.a_min+2.,Para.a_max,15)))
zgrid = linspace(Para.z_min,Para.z_max,10)
Para.domain = vstack([(a,z) for a in agrid for z in zgrid])

#initial value function
def V0(a,z):
    return Para.U(exp(z)+r*a)/(1-Para.beta)

#construct the Bellman Map
T = bellman.BellmanMap(Para,r)

#iterate on the Bellman Map until convergence
Vf,Vs = bellman.approximateValueFunction(T(V0),Para)
while diff > 1e-6:  
    Vfnew,Vsnew = bellman.approximateValueFunction(T(Vf),Para)
    diff =  max(abs(Vs-Vsnew))
    if rank == 0:
        print diff
    Vf = Vfnew
    Vs = Vsnew