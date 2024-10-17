set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/compilers/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.7.64")
set(CMAKE_CUDA_DEVICE_LINKER "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "11.2")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/11.7/nccl/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/11.7/nvshmem/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/targets/x86_64-linux/lib/stubs;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/include;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/include;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/11.7/nccl/include;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/11.7/nvshmem/include;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/compilers/extras/qd/include/qd;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/hpcx/latest/hcoll/include;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/hpcx/latest/ompi/include;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/hpcx/latest/sharp/include;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/hpcx/latest/ucx/mt/include;/usr/local/gdrcopy/include;/usr/include/c++/11;/usr/include/x86_64-linux-gnu/c++/11;/usr/include/c++/11/backward;/usr/lib/gcc/x86_64-linux-gnu/11/include;/usr/local/include;/usr/include/x86_64-linux-gnu;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/11.7/nccl/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/11.7/nvshmem/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/targets/x86_64-linux/lib/stubs;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/targets/x86_64-linux/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/hpcx/latest/hcoll/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/hpcx/latest/ompi/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/hpcx/latest/nccl_rdma_sharp_plugin/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/hpcx/latest/sharp/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/comm_libs/hpcx/latest/ucx/mt/lib;/usr/local/gdrcopy/lib;/usr/lib/gcc/x86_64-linux-gnu/11;/usr/lib/x86_64-linux-gnu;/usr/lib;/lib/x86_64-linux-gnu;/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")