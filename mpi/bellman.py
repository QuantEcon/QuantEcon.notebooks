
"""
Created on Tue Jun 25 13:40:30 2013
Holds all of the code for the bellman equation
@author: dgevans
"""

from scipy.optimize import minimize_scalar

from numpy import * 
from mpi4py import MPI
from scipy.interpolate import SmoothBivariateSpline
from numpy.polynomial.hermite import hermgauss

#compute nodes for gaussian quadrature.
zhatvec,z_weights = hermgauss(10)
z_weights /= sum(z_weights)


def approximateValueFunction(TV,Para):
    '''
    Approximates the value function over the grid defined by Para.domain.  Uses
    mpi. Returns both an interpolated value function and the value of TV at each point
    in the domain.
    '''
    comm = MPI.COMM_WORLD
    #first split up domain for each process
    s = comm.Get_size()
    rank = comm.Get_rank()
    n = len(Para.domain)
    m = n//s
    r = n%s
    #let each process take a slice of the domain
    mydomain = Para.domain[rank*m+min(rank,r):
                           (rank+1)*m+min(rank+1,r)]

    #get the value at each point in my domain
    myV = hstack(map(TV,mydomain))
    #gather the values for each process
    Vs = comm.gather(myV)
    
    if rank == 0:
        #fit 
        Vs = hstack(Vs).flatten()
        Vf = SmoothBivariateSpline(Para.domain[:,0],Para.domain[:,1],Vs)
    else:
        Vf = None
    return comm.bcast(Vf),comm.bcast(Vs)
    
    
    
    

class BellmanMap(object):
    '''
    Bellman map object.  Once constructed will take a value function and return
    a new value function
    '''
    
    def __init__(self,Para,r):
        '''
        Initializes the Bellman Map function.  Need parameters and r
        '''
        self.Para = Para
        self.r = r
        self.w = 1.
        
    def __call__(self,Vf):
        '''
        Given a current value function return new value function
        '''
        self.Vf = Vf
        
        return lambda x: self.maximizeObjective(x[0],x[1])[0]
        
        
    def maximizeObjective(self,a,z):
        '''
        Maximize the objective function Vf given the states a and productivity z.
        Note the state for this problem are assets and previous periods productivity
        return tuple (V,c_policy,a_policy)
        '''
        
        #total wealth for agent is easy
        W = (1+self.r)*a + exp(z)*self.w
        beta = self.Para.beta
        Vf = self.Vf
        
        
        def obj_f(aprime):
            c = W-aprime
            return -(self.Para.U(c) + beta*self.E(lambda zprime:Vf(aprime,zprime).flatten(),z))
        
        a_min = self.Para.a_min
        a_max = min(self.Para.a_max,W-0.00001)
        
        res = minimize_scalar(obj_f,bounds=(a_min,a_max),method='bounded')
        
        return -res.fun,W-res.x,res.x
         
    def E(self,f,z_):
        '''
        Compute the exepected value of a function f of zprime conditional on z
        '''
        #Pz_min,Pz_max,z_pdf = self.get_z_distribution(z_)
        z_min = array([self.Para.z_min])
        z_max = array([self.Para.z_max])
        #Compute the nodes for Gaussian quadrature
        zvec = self.Para.rho * z_ + self.Para.sigma_e*zhatvec*sqrt(2)
        
        #lower bound for z
        zvec[zvec<z_min] = z_min 
        #upper bound for z
        zvec[zvec>z_max] = z_max
        return z_weights.dot(f(zvec))