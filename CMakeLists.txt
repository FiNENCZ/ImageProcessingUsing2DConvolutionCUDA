cmake_minimum_required(VERSION 3.0)

project(TEMPLATE LANGUAGES CXX CUDA)

# Nastavení OpenMP
find_package(OpenMP REQUIRED)

# Hledání knihovny libpng
find_package(PNG REQUIRED)

# Přidání exekutivního souboru
add_executable(TEMPALTE main.cu)

# Přidání kompilátorových možností pro OpenMP
if(OpenMP_CXX_FOUND)
    target_compile_options(TEMPALTE PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Zahrnutí hlavičkových souborů pro libpng
include_directories(${PNG_INCLUDE_DIR})

# Propojení s knihovnami
target_link_libraries(TEMPALTE ${PNG_LIBRARY} ${OpenMP_CXX_LIBRARIES})
