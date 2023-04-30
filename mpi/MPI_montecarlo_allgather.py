
import numpy as np
from mpi4py import MPI

N = 10000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

x = np.random.randn(N)
my_result = np.cos(x).mean() #my average

#now gather from all processes
results = comm.gather(my_result)

print("Gather result for process " + str(rank) +":",results)

#now allgather from all processes
results = comm.allgather(my_result)

print("Allgather resuts for process " + str(rank) + ":",results)