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
include CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/flags.make

CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.o: CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/flags.make
CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.o: ../main.cu
CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.o: CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/ImageProcessingUsing2DConvolutionCUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.o"
	/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.o -MF CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.o.d -x cu -c /root/workspace/ImageProcessingUsing2DConvolutionCUDA/main.cu -o CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.o

CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target ImageProcessingUsing2DConvolutionCUDA
ImageProcessingUsing2DConvolutionCUDA_OBJECTS = \
"CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.o"

# External object files for target ImageProcessingUsing2DConvolutionCUDA
ImageProcessingUsing2DConvolutionCUDA_EXTERNAL_OBJECTS =

ImageProcessingUsing2DConvolutionCUDA: CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/main.cu.o
ImageProcessingUsing2DConvolutionCUDA: CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/build.make
ImageProcessingUsing2DConvolutionCUDA: /usr/lib/x86_64-linux-gnu/libpng.so
ImageProcessingUsing2DConvolutionCUDA: /usr/lib/x86_64-linux-gnu/libz.so
ImageProcessingUsing2DConvolutionCUDA: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
ImageProcessingUsing2DConvolutionCUDA: /usr/lib/x86_64-linux-gnu/libpthread.a
ImageProcessingUsing2DConvolutionCUDA: CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/ImageProcessingUsing2DConvolutionCUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable ImageProcessingUsing2DConvolutionCUDA"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/build: ImageProcessingUsing2DConvolutionCUDA
.PHONY : CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/build

CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/clean

CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/depend:
	cd /root/workspace/ImageProcessingUsing2DConvolutionCUDA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/ImageProcessingUsing2DConvolutionCUDA /root/workspace/ImageProcessingUsing2DConvolutionCUDA /root/workspace/ImageProcessingUsing2DConvolutionCUDA/build /root/workspace/ImageProcessingUsing2DConvolutionCUDA/build /root/workspace/ImageProcessingUsing2DConvolutionCUDA/build/CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ImageProcessingUsing2DConvolutionCUDA.dir/depend
