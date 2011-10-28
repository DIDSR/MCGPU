#!/bin/bash
# 
#   ** Simple script to compile the code MC-GPU v1.2 **
#
#      The installations paths to the CUDA toolkit and SDK (http://www.nvidia.com/cuda) and the MPI 
#      library path have to be adapted before runing the script!
# 
#      Default paths:
#           /usr/local/cuda
#           /opt/cuda_SDK_4.0
#           /usr/include/openmpi
#
# 
#                      @file    make_MC-GPU_v1.2.sh
#                      @author  Andreu Badal [Andreu.Badal-Soler(at)fda.hhs.gov]
#                      @date    2011/07/12
#   

# -- Compile GPU code for compute capability 1.3 and 2.0, with MPI:

echo " "
echo " -- Compiling MC-GPU with CUDA 4.0 for both compute capability 1.3 and 2.0 (64 bits), with MPI:"
echo "    To run a simulation in parallel with openMPI execute:"
echo "      $ time mpirun --tag-output -v -x LD_LIBRARY_PATH -hostfile hostfile_gpunodes -n 22 /GPU_cluster/MC-GPU_v1.2.x /GPU_cluster/MC-GPU_v1.2.in | tee MC-GPU_v1.2.out"
echo " "
echo "nvcc -O3 -use_fast_math -DUSING_CUDA -DUSING_MPI MC-GPU_v1.2.cu -o MC-GPU_v1.2.x -I./ -I/usr/local/cuda/include -I/opt/cuda_SDK_4.0/C/common/inc/ -L/opt/cuda_SDK_4.0/C/lib/ -L/usr/local/cuda/lib64/ -lcutil_x86_64 -lcudart -lm --ptxas-options=-v -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20 -I./ -I/usr/include/openmpi -L/usr/lib/ -lmpi"

nvcc -O3 -use_fast_math -DUSING_CUDA -DUSING_MPI MC-GPU_v1.2.cu -o MC-GPU_v1.2.x -I./ -I/usr/local/cuda/include -I/opt/cuda_SDK_4.0/C/common/inc/ -L/opt/cuda_SDK_4.0/C/lib/ -L/usr/local/cuda/lib64/ -lcutil_x86_64 -lcudart -lm --ptxas-options=-v -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20 -I/usr/include/openmpi -lmpi



# -- CPU compilation:
 
# ** GCC (with MPI):
# gcc -x c -DUSING_MPI MC-GPU_v1.2.cu -o MC-GPU_v1.2_gcc_MPI.x -Wall -O3 -ffast-math -ftree-vectorize -ftree-vectorizer-verbose=1 -funroll-loops -static-libgcc -I./ -lm -I/usr/include/openmpi -I/usr/lib/openmpi/include/openmpi/ -L/usr/lib/openmpi/lib -lmpi

     
# ** Intel compiler (with MPI):
# icc -x c -O3 -ipo -no-prec-div -msse4.2 -parallel -Wall -DUSING_MPI MC-GPU_v1.2.cu -o MC-GPU_v1.2_icc_MPI.x -I./ -lm -I/usr/include/openmpi -L/usr/lib/openmpi/lib/ -lmpi


# ** PGI compiler:
# pgcc -fast,sse -O3 -Mipa=fast -Minfo -csuffix=cu -Mconcur MC-GPU_v1.2.cu -I./ -lm -o MC-GPU_v1.2_PGI.x

