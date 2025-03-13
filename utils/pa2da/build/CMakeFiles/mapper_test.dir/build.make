# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_SOURCE_DIR = /workplace/emulators/radiation-error-emulator/newdram/ramv3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workplace/emulators/radiation-error-emulator/newdram/ramv3/build

# Include any dependencies generated for this target.
include CMakeFiles/mapper_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mapper_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mapper_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mapper_test.dir/flags.make

CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.o: CMakeFiles/mapper_test.dir/flags.make
CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.o: /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/addr_mapper.cpp
CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.o: CMakeFiles/mapper_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workplace/emulators/radiation-error-emulator/newdram/ramv3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.o -MF CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.o.d -o CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.o -c /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/addr_mapper.cpp

CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/addr_mapper.cpp > CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.i

CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/addr_mapper.cpp -o CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.s

CMakeFiles/mapper_test.dir/src/config_loader.cpp.o: CMakeFiles/mapper_test.dir/flags.make
CMakeFiles/mapper_test.dir/src/config_loader.cpp.o: /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/config_loader.cpp
CMakeFiles/mapper_test.dir/src/config_loader.cpp.o: CMakeFiles/mapper_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workplace/emulators/radiation-error-emulator/newdram/ramv3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/mapper_test.dir/src/config_loader.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mapper_test.dir/src/config_loader.cpp.o -MF CMakeFiles/mapper_test.dir/src/config_loader.cpp.o.d -o CMakeFiles/mapper_test.dir/src/config_loader.cpp.o -c /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/config_loader.cpp

CMakeFiles/mapper_test.dir/src/config_loader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mapper_test.dir/src/config_loader.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/config_loader.cpp > CMakeFiles/mapper_test.dir/src/config_loader.cpp.i

CMakeFiles/mapper_test.dir/src/config_loader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mapper_test.dir/src/config_loader.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/config_loader.cpp -o CMakeFiles/mapper_test.dir/src/config_loader.cpp.s

CMakeFiles/mapper_test.dir/src/main.cpp.o: CMakeFiles/mapper_test.dir/flags.make
CMakeFiles/mapper_test.dir/src/main.cpp.o: /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/main.cpp
CMakeFiles/mapper_test.dir/src/main.cpp.o: CMakeFiles/mapper_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workplace/emulators/radiation-error-emulator/newdram/ramv3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/mapper_test.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mapper_test.dir/src/main.cpp.o -MF CMakeFiles/mapper_test.dir/src/main.cpp.o.d -o CMakeFiles/mapper_test.dir/src/main.cpp.o -c /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/main.cpp

CMakeFiles/mapper_test.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mapper_test.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/main.cpp > CMakeFiles/mapper_test.dir/src/main.cpp.i

CMakeFiles/mapper_test.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mapper_test.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workplace/emulators/radiation-error-emulator/newdram/ramv3/src/main.cpp -o CMakeFiles/mapper_test.dir/src/main.cpp.s

# Object files for target mapper_test
mapper_test_OBJECTS = \
"CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.o" \
"CMakeFiles/mapper_test.dir/src/config_loader.cpp.o" \
"CMakeFiles/mapper_test.dir/src/main.cpp.o"

# External object files for target mapper_test
mapper_test_EXTERNAL_OBJECTS =

mapper_test: CMakeFiles/mapper_test.dir/src/addr_mapper.cpp.o
mapper_test: CMakeFiles/mapper_test.dir/src/config_loader.cpp.o
mapper_test: CMakeFiles/mapper_test.dir/src/main.cpp.o
mapper_test: CMakeFiles/mapper_test.dir/build.make
mapper_test: /usr/lib/aarch64-linux-gnu/libyaml-cpp.so.0.5.2
mapper_test: CMakeFiles/mapper_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workplace/emulators/radiation-error-emulator/newdram/ramv3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable mapper_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mapper_test.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/cmake -E make_directory /workplace/emulators/radiation-error-emulator/newdram/ramv3/build/configs
	/usr/bin/cmake -E copy_directory /workplace/emulators/radiation-error-emulator/newdram/ramv3/configs /workplace/emulators/radiation-error-emulator/newdram/ramv3/build/configs

# Rule to build all files generated by this target.
CMakeFiles/mapper_test.dir/build: mapper_test
.PHONY : CMakeFiles/mapper_test.dir/build

CMakeFiles/mapper_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mapper_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mapper_test.dir/clean

CMakeFiles/mapper_test.dir/depend:
	cd /workplace/emulators/radiation-error-emulator/newdram/ramv3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workplace/emulators/radiation-error-emulator/newdram/ramv3 /workplace/emulators/radiation-error-emulator/newdram/ramv3 /workplace/emulators/radiation-error-emulator/newdram/ramv3/build /workplace/emulators/radiation-error-emulator/newdram/ramv3/build /workplace/emulators/radiation-error-emulator/newdram/ramv3/build/CMakeFiles/mapper_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mapper_test.dir/depend

