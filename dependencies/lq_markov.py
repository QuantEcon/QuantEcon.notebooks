"""

Author: Sebastian Graves

Provides a class called LQ_Markov for analyzing Markov jump LQ problems.
Provides a function to map first type of Barro model into LQ Markov problem.
Provides a function to map second type of Barro model (with restructuring) into LQ Markov problem.

"""

import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from quantecon import LQ
from scipy.linalg import solve
from quantecon import MarkovChain
from numpy import dot

class LQ_Markov(object):
    """ 
    This class is for analyzing Markov jump linear quadratic optimal control problems.
    """
    
    # Use of  *args to allow there to be variable n states of the world
    def __init__(self, beta, Pi, *args):

        #Create namedtuple to keep the R,Q,A,B,C,W matrices for each state of the world
        world = namedtuple('world', ['A', 'B', 'C', 'R', 'Q', 'W'])
        
        #Create a dictionary for each of the fields in the named tuple
        #For example, self.R[i] is the R matrix from state of the world, i
        for j in world._fields:
            k = {} 
            for i in range(1,len(args)+1):
                k[i] = getattr(args[i-1],j)
            setattr(self,str(j),k)
        self.n = len(args) #number of states of the world
        self.beta, self.Pi = beta, Pi
        
        # Find dimensions of the state (x) and control (u) vectors
        self.nu, junk = self.Q[1].shape
        self.nx, self.nw = self.C[1].shape
        
        # If W matrix is not given in some state of the world, insert a matrix of zeros
        for i in range(1,len(args)+1):
            if self.W[i] is None:
                self.W[i] = np.zeros((self.nu,self.nx))
        
        # Create the P matrices, initialize as identity matrix
        P = {}
        for i in range(1,self.n+1):
            P[i] = np.eye(self.nx)
        
        # == Set up for iteration on Riccati equation == #
        iteration = 0
        tolerance = 1e-10
        max_iter = 100000
        error = tolerance + 1
        fail_msg = "Convergence failed after {} iterations."
        
        # == Main loop == #
        while error > tolerance:
            
            if iteration > max_iter:
                self.finalerror = error
                raise ValueError(fail_msg.format(iteration))
            
            else:
                P1, Diff, F = {},{},{}
                for i in range(1,self.n+1):
                    sum1, sum2, sum3, sum4,  = np.zeros((self.nx, self.nx)) , np.zeros((self.nx, self.nx)) , np.zeros((self.nu, self.nu)) , np.zeros((self.nu, self.nx))
                    for j in range(1,self.n+1):
                        sum1 = sum1 + self.beta*self.Pi[i-1,j-1]*self.A[i].T.dot(P[j]).dot(self.A[i])
                        sum2 = sum2 + self.Pi[i-1,j-1]*(self.beta*self.A[i].T.dot(P[j]).dot(self.B[i]) + self.W[i].T).dot(solve(self.Q[i] + self.beta*self.B[i].T.dot(P[j]).dot(self.B[i]),self.beta*self.B[i].T.dot(P[j]).dot(self.A[i])+ self.W[i]))
                        sum3 = sum3 + self.beta*self.Pi[i-1,j-1]*self.B[i].T.dot(P[j]).dot(self.B[i])
                        sum4 = sum4 + self.beta*self.Pi[i-1,j-1]*self.B[i].T.dot(P[j]).dot(self.A[i])
                    P1[i] = self.R[i] + sum1 - sum2
                    F[i] = solve(self.Q[i]+ sum3,sum4 + self.W[i])
                    Diff[i] = np.abs(P1[i]-P[i])
   
                error = np.max(sum(Diff.values())) #This adds the n Diff matrices together and finds the largest value
                P = P1
                iteration += 1
            
        self.P, self.F, = P, F
        
        #Rho for each state is calculated below
        self.rho = {}
        # The [i,j] element of the X matrix is trace(P_i*C_j*C_j')
        X = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                X[i,j]= np.trace(self.P[i+1].dot(self.C[j+1]).dot(self.C[j+1].T))
        rho = np.linalg.inv(np.eye(self.n)-self.beta*self.Pi).dot(np.atleast_2d(np.diagonal(self.Pi.dot(X))).T).dot(self.beta)
        for i in range(1,self.n+1):
            self.rho[i] = rho[i-1,0]

    def compute_sequence(self, x0, ts_length=None):
        """ 
        This Function simulates x,u,w.
        It is based on the compute_sequence function from the LQ class, but is amended to recognise
        the fact that the A,B,C matrices can depend on the state of the world
        """
        x0 = np.asarray(x0)
        x0 = x0.reshape(self.nx, 1)

        T = ts_length if ts_length else 100

        chain = MarkovChain(self.Pi)
        state = chain.simulate_indices(ts_length=T+1)

        x_path = np.empty((self.nx, T+1))
        u_path = np.empty((self.nu, T))
        shocks = np.random.randn(self.nw, T+1)
        w_path = np.empty((self.nx,T+1))
        for i in range(T+1):
            w_path[:,i] = self.C[state[i]+1].dot(shocks[:,i])

        x_path[:, 0] = x0.flatten()
        u_path[:, 0] = - dot(self.F[state[0]+1], x0).flatten()
        for t in range(1, T):
            Ax, Bu = dot(self.A[state[t]+1], x_path[:, t-1]), dot(self.B[state[t]+1], u_path[:, t-1])
            x_path[:, t] = Ax + Bu + w_path[:, t]
            u_path[:, t] = - dot(self.F[state[t]+1], x_path[:, t])
        Ax, Bu = dot(self.A[state[T]+1], x_path[:, T-1]), dot(self.B[state[T]+1], u_path[:, T-1])
        x_path[:, T] = Ax + Bu + w_path[:, T]

        return x_path, u_path, w_path, state

def LQ_markov_mapping(A22,C2,Ug,p1,p2,c1=0):

    """
    Function which takes A22, C2, Ug, p^t_{t+1},p^t_{t+2} and penalty parameter c1, and returns the required matrices for 
    the LQ_Markov model: A_t,B,C_t,R_t,Q_t, W_t
    c1 is the cost of issuing different quantities for different maturities.
    This version uses the condensed version of the endogenous state.
    """
    
    # Make sure all matrices can be treated as 2D arrays #
    A22 = np.atleast_2d(A22)
    C2 = np.atleast_2d(C2)
    Ug = np.atleast_2d(Ug)
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)
    
    # Find number of states (z) and shocks (w)
    nz, nw = C2.shape
    
    # Create A11, B1, S1, S2, Sg, S matrices
    A11 = np.zeros((2,2))
    A11[0,1]=1
    
    B1 = np.eye(2)
    
    S1 = np.hstack((np.eye(1),np.zeros((1,nz+1))))
    Sg = np.hstack((np.zeros((1,2)),Ug))
    S = S1 + Sg
    
    # Create M matrix
    M = np.hstack((-p1,-p2))
    
    # Create A,B,C matrices
    A_T = np.hstack((A11,np.zeros((2,nz))))
    A_B = np.hstack((np.zeros((nz,2)),A22))
    A = np.vstack((A_T,A_B))
    
    B = np.vstack((B1,np.zeros((nz,2))))
    
    C = np.vstack((np.zeros((2,nw)),C2))

    # Create Q^c matrix
    Qc = np.array([[1,-1],[-1,1]])
    
    # Create R,Q,W matrices
    
    R = S.T.dot(S)
    Q = M.T.dot(M) + c1*Qc
    W = M.T.dot(S)
    
    return A,B,C,R,Q,W

def LQ_markov_mapping_restruct(A22,C2,Ug,T,p_t,c=0):

    """
    Function which takes A22, C2, T, p_t, c and returns the required matrices for the LQ_Markov model: A,B,C,R,Q,W
    Note, p_t should be a T by 1 matrix.
    c is the cost of adjusting issuance (a scalar)
    """
    
    # Make sure all matrices can be treated as 2D arrays == #
    A22 = np.atleast_2d(A22)
    C2 = np.atleast_2d(C2)
    Ug = np.atleast_2d(Ug)
    p_t = np.atleast_2d(p_t)
    
    # Find number of states (z) and shocks (w)
    nz, nw = C2.shape
    
    # Create Sx,tSx,Ss,S_t matrices (tSx stands for \tilde S_x)
    Ss = np.hstack((np.eye(T-1),np.zeros((T-1,1))))
    Sx = np.hstack((np.zeros((T-1,1)),np.eye(T-1)))
    tSx = np.zeros((1,T))
    tSx[0,0] = 1

    S_t = np.hstack((tSx + p_t.T.dot(Ss.T).dot(Sx), Ug))
    
    # Create A,B,C matrices
    A_T = np.hstack((np.zeros((T,T)),np.zeros((T,nz))))
    A_B = np.hstack((np.zeros((nz,T)),A22))
    A = np.vstack((A_T,A_B))
    
    B = np.vstack((np.eye(T),np.zeros((nz,T))))
    
    C = np.vstack((np.zeros((T,nw)),C2))

    # Create cost matrix Sc

    Sc = np.hstack((Sx,np.zeros((T-1,nz))))
    
    # Create R_t,Q_t,W_t matrices
    
    R_c = S_t.T.dot(S_t) + c*Sc.T.dot(Sc)
    Q_c = p_t.dot(p_t.T) + c*Ss.T.dot(Ss)
    W_c = -p_t.dot(S_t) - c*Ss.T.dot(Sc)
    
    return A,B,C,R_c,Q_c,W_c
