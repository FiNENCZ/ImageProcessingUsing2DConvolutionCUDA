cmake_minimum_required(VERSION 3.0)

project(TEMPLATE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Nastavení OpenMP
find_package(OpenMP REQUIRED)

# Hledání knihovny libpng
find_package(PNG REQUIRED)

# Přidání exekutivního souboru
add_executable(TEMPALTE main.cu pngio.cpp)

list(APPEND CMAKE_CXX_FLAGS "-O2 -ffast-math -Wall ${OpenMP_CXX_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fno-strict-aliasing -lineinfo -use_fast_math --ptxas-options -dlcm=cg")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")

# Přidání kompilátorových možností pro OpenMP
if(OpenMP_CXX_FOUND)
    target_compile_options(TEMPALTE PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Zahrnutí hlavičkových souborů pro libpng
include_directories(${PNG_INCLUDE_DIR})

# Propojení s knihovnami
target_link_libraries(TEMPALTE ${PNG_LIBRARY} ${OpenMP_CXX_LIBRARIES})