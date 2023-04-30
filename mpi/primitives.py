
"""
Created on Tue Jun 25 13:29:54 2013

@author: dgevans
"""
from numpy import *

class primitives(object):
    '''
    Class that holds the basic parameters of the problem
    '''
    
    beta = 0.95#discount factor
    
    rho = 0.9#persistence of idiosyncratic productivity
    
    sigma_e = 0.1#standard deviation of idiosyncratic productivity
    
    a_min = 0.0 #minimum assets
    
    a_max = 60.0 #maximum assets
    
    z_min = -2. #minimum productivity
    
    z_max = 2. #maximum productivity
    
    
    
class primitives_CRRA(primitives):
    '''
    Extension of primitives class holding the preference structure
    '''
    
    sigma = 2.
    
    def U(self,c):
        """
        CRRA utility function
        """
        sigma = self.sigma
        if sigma == 1.:
            return log(c)
        else:
            return (c)**(1-sigma)/(1-sigma)
            
    def Uc(self,c):
        """
        Derivative of the CRRA utility function
        """
        sigma = self.sigma
        return c**(-sigma)
        
        