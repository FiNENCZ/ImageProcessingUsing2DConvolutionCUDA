cmake_minimum_required(VERSION 3.0)

project(IMGPROC LANGUAGES CXX CUDA)



# Nastavení OpenMP
find_package(OpenMP REQUIRED)
find_package(PNG REQUIRED)

# Přidání exekutivního souboru


list(APPEND CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
# list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fno-strict-aliasing -lineinfo -use_fast_math --ptxas-options -dlcm=cg")
# list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")

add_executable(IMGPROC main.cu utils/pngio.cpp)

# Propojení s knihovnami
target_link_libraries(IMGPROC ${PNG_LIBRARY})