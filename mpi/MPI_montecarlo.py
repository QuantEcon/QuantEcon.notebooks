
import numpy as np
from mpi4py import MPI

N = 10000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

x = np.random.randn(N)
my_result = np.cos(x).mean() #my average

#now average accross all processes
results = comm.gather(my_result)
if rank == 0:
    print(np.mean(results))