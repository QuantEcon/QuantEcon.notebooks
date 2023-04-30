
from mpi4py import MPI

comm = MPI.COMM_WORLD #retreive the communicator module
rank = comm.Get_rank() #get the rank of the process
size = comm.Get_size() #get the number of processes

print("Hello world from process "+str(rank)+" of " + str(size))