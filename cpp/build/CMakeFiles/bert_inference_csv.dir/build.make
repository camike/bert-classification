# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 4.1

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\wudiz\Desktop\sort\cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\wudiz\Desktop\sort\cpp\build

# Include any dependencies generated for this target.
include CMakeFiles/bert_inference_csv.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/bert_inference_csv.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/bert_inference_csv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bert_inference_csv.dir/flags.make

CMakeFiles/bert_inference_csv.dir/codegen:
.PHONY : CMakeFiles/bert_inference_csv.dir/codegen

CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.obj: CMakeFiles/bert_inference_csv.dir/flags.make
CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.obj: CMakeFiles/bert_inference_csv.dir/includes_CXX.rsp
CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.obj: C:/Users/wudiz/Desktop/sort/cpp/src/bert_inference_csv.cpp
CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.obj: CMakeFiles/bert_inference_csv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\wudiz\Desktop\sort\cpp\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.obj"
	C:\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.obj -MF CMakeFiles\bert_inference_csv.dir\src\bert_inference_csv.cpp.obj.d -o CMakeFiles\bert_inference_csv.dir\src\bert_inference_csv.cpp.obj -c C:\Users\wudiz\Desktop\sort\cpp\src\bert_inference_csv.cpp

CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.i"
	C:\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\wudiz\Desktop\sort\cpp\src\bert_inference_csv.cpp > CMakeFiles\bert_inference_csv.dir\src\bert_inference_csv.cpp.i

CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.s"
	C:\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\wudiz\Desktop\sort\cpp\src\bert_inference_csv.cpp -o CMakeFiles\bert_inference_csv.dir\src\bert_inference_csv.cpp.s

# Object files for target bert_inference_csv
bert_inference_csv_OBJECTS = \
"CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.obj"

# External object files for target bert_inference_csv
bert_inference_csv_EXTERNAL_OBJECTS =

bin/bert_inference_csv.exe: CMakeFiles/bert_inference_csv.dir/src/bert_inference_csv.cpp.obj
bin/bert_inference_csv.exe: CMakeFiles/bert_inference_csv.dir/build.make
bin/bert_inference_csv.exe: C:/Users/wudiz/Desktop/sort/cpp/lib/onnxruntime/lib/onnxruntime.dll
bin/bert_inference_csv.exe: CMakeFiles/bert_inference_csv.dir/linkLibs.rsp
bin/bert_inference_csv.exe: CMakeFiles/bert_inference_csv.dir/objects1.rsp
bin/bert_inference_csv.exe: CMakeFiles/bert_inference_csv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:\Users\wudiz\Desktop\sort\cpp\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin\bert_inference_csv.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\bert_inference_csv.dir\link.txt --verbose=$(VERBOSE)
	"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different C:/Users/wudiz/Desktop/sort/cpp/lib/onnxruntime/bin/onnxruntime.dll C:/Users/wudiz/Desktop/sort/cpp/build/bin
	"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different C:/Users/wudiz/Desktop/sort/cpp/lib/onnxruntime/bin/onnxruntime_providers_shared.dll C:/Users/wudiz/Desktop/sort/cpp/build/bin
	"C:\Program Files\CMake\bin\cmake.exe" -E make_directory C:/Users/wudiz/Desktop/sort/cpp/build/bin/model
	"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different C:/Users/wudiz/Desktop/sort/cpp/model/model.onnx C:/Users/wudiz/Desktop/sort/cpp/build/bin/model/
	"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different C:/Users/wudiz/Desktop/sort/cpp/model/vocab.txt C:/Users/wudiz/Desktop/sort/cpp/build/bin/model/

# Rule to build all files generated by this target.
CMakeFiles/bert_inference_csv.dir/build: bin/bert_inference_csv.exe
.PHONY : CMakeFiles/bert_inference_csv.dir/build

CMakeFiles/bert_inference_csv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\bert_inference_csv.dir\cmake_clean.cmake
.PHONY : CMakeFiles/bert_inference_csv.dir/clean

CMakeFiles/bert_inference_csv.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\wudiz\Desktop\sort\cpp C:\Users\wudiz\Desktop\sort\cpp C:\Users\wudiz\Desktop\sort\cpp\build C:\Users\wudiz\Desktop\sort\cpp\build C:\Users\wudiz\Desktop\sort\cpp\build\CMakeFiles\bert_inference_csv.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/bert_inference_csv.dir/depend

