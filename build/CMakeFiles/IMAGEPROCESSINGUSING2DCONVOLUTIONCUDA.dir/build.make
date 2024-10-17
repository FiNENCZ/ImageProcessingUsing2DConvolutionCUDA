# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/workspace/ImageProcessingUsing2DConvolutionCUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/workspace/ImageProcessingUsing2DConvolutionCUDA/build

# Include any dependencies generated for this target.
include CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/flags.make

CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.o: CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/flags.make
CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.o: ../main.cu
CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.o: CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/ImageProcessingUsing2DConvolutionCUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.o"
	/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.o -MF CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.o.d -x cu -c /root/workspace/ImageProcessingUsing2DConvolutionCUDA/main.cu -o CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.o

CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA
IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA_OBJECTS = \
"CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.o"

# External object files for target IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA
IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA_EXTERNAL_OBJECTS =

IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA: CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/main.cu.o
IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA: CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/build.make
IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA: CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/ImageProcessingUsing2DConvolutionCUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/build: IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA
.PHONY : CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/build

CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/cmake_clean.cmake
.PHONY : CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/clean

CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/depend:
	cd /root/workspace/ImageProcessingUsing2DConvolutionCUDA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/ImageProcessingUsing2DConvolutionCUDA /root/workspace/ImageProcessingUsing2DConvolutionCUDA /root/workspace/ImageProcessingUsing2DConvolutionCUDA/build /root/workspace/ImageProcessingUsing2DConvolutionCUDA/build /root/workspace/ImageProcessingUsing2DConvolutionCUDA/build/CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/IMAGEPROCESSINGUSING2DCONVOLUTIONCUDA.dir/depend

