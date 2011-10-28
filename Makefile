
# ========================================================================================
#                                  MAKEFILE MC-GPU v1.1
#
# 
#   ** Simple script to compile the code MC-GPU v1.1 for compute capabilities 1.3 and 
#      2.0 (the appropriate one is used on execution time).
#      For information on how to compile the code for the CPU or using MPI, read the
#      file "make_MC-GPU_v1.1.sh".
#
#      The installation paths to the CUDA toolkit and SDK (http://www.nvidia.com/cuda) 
#      and the MPI libraries (MPICH2: http://www.mcs.anl.gov/research/projects/mpich2/) 
#      may have to be modified by the user.
#
# 
#                      @file    Makefile
#                      @author  Andreu Badal [Andreu.Badal-Soler (at) fda.hhs.gov]
#                      @date    2010/06/25
#   
# ========================================================================================

SHELL = /bin/sh

# Suffixes:
.SUFFIXES: .cu .o

# Compilers and linker:
CC = nvcc

# Program's name:
PROG = MC-GPU_v1.2.x

# Include and library paths:
CUDA_PATH = /usr/local/cuda/include/
CUDA_LIB_PATH = /usr/local/cuda/lib64/
CUDA_SDK_PATH = /opt/cuda_SDK_4.0/C/common/inc/
CUDA_SDK_LIB_PATH = /opt/cuda_SDK_4.0/C/lib/


# Compiler's flags:
CFLAGS = -O3 -use_fast_math -DUSING_CUDA -DUSING_MPI -I./ -I$(CUDA_PATH) -I$(CUDA_SDK_PATH) -L$(CUDA_SDK_LIB_PATH) -L$(CUDA_LIB_PATH) -lcutil_x86_64 -lcudart -lm -I/usr/include/openmpi -lmpi --ptxas-options=-v -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20
 

# Command to erase files:
RM = /bin/rm -vf

# .cu files path:
SRCS = MC-GPU_v1.2.cu

# Building the application:
default: $(PROG)
$(PROG):
	$(CC) $(CFLAGS) $(SRCS) -o $(PROG)

# Rule for cleaning re-compilable files
clean:
	$(RM) $(PROG)

