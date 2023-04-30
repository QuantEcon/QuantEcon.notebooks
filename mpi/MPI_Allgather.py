
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# create a 2x2 array filled with the rank of the process
x = np.ones((2,2)) * rank 

# create an empty array to store the gathered result
X = np.empty((2*size,2))

#use numpy Allgather
                #send buffer   #recv buffer
comm.Allgather([x, MPI.DOUBLE],[X, MPI.DOUBLE])

if rank ==0:
    print(X)