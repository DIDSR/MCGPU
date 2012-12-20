
# ========================================================================================
#                                  MAKEFILE MC-GPU v1.3
#
# 
#   ** Simple script to compile the code MC-GPU v1.3.
#      For information on how to compile the code for the CPU or using MPI, read the
#      file "make_MC-GPU_v1.3.sh".
#
#      The installation paths to the CUDA toolkit and SDK (http://www.nvidia.com/cuda) 
#      and the MPI libraries (openMPI) may have to be modified by the user. 
#      The zlib.h library is used to allow gzip-ed input files.
#
#      Default paths:
#         CUDA:  /usr/local/cuda
#         SDK:   /usr/local/cuda/samples
#         MPI:   /usr/include/openmpi
#
# 
#                      @file    Makefile
#                      @author  Andreu Badal [Andreu.Badal-Soler (at) fda.hhs.gov]
#                      @date    2012/12/12
#   
# ========================================================================================

SHELL = /bin/sh

# Suffixes:
.SUFFIXES: .cu .o

# Compilers and linker:
CC = nvcc

# Program's name:
PROG = MC-GPU_v1.3.x

# Include and library paths:
CUDA_PATH = /usr/local/cuda/include/
CUDA_LIB_PATH = /usr/local/cuda/lib64/
CUDA_SDK_PATH = /usr/local/cuda/samples/common/inc/
CUDA_SDK_LIB_PATH = /usr/local/cuda/samples/common/lib/linux/x86_64/
OPENMPI_PATH = /usr/include/openmpi


# Compiler's flags:
CFLAGS = -O3 -use_fast_math -m64 -DUSING_CUDA -DUSING_MPI -I./ -I$(CUDA_PATH) -I$(CUDA_SDK_PATH) -L$(CUDA_SDK_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lm -lz -I$(OPENMPI_PATH) -lmpi --ptxas-options=-v -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30 
 

# Command to erase files:
RM = /bin/rm -vf

# .cu files path:
SRCS = MC-GPU_v1.3.cu

# Building the application:
default: $(PROG)
$(PROG):
	$(CC) $(CFLAGS) $(SRCS) -o $(PROG)

# Rule for cleaning re-compilable files
clean:
	$(RM) $(PROG)

