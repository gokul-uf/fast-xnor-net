# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.6.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.6.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net

# Include any dependencies generated for this target.
include lib/CMakeFiles/pcm.dir/depend.make

# Include the progress variables for this target.
include lib/CMakeFiles/pcm.dir/progress.make

# Include the compile flags for this target's objects.
include lib/CMakeFiles/pcm.dir/flags.make

lib/CMakeFiles/pcm.dir/perf/perf.cpp.o: lib/CMakeFiles/pcm.dir/flags.make
lib/CMakeFiles/pcm.dir/perf/perf.cpp.o: lib/perf/perf.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/CMakeFiles/pcm.dir/perf/perf.cpp.o"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcm.dir/perf/perf.cpp.o -c /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/perf/perf.cpp

lib/CMakeFiles/pcm.dir/perf/perf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcm.dir/perf/perf.cpp.i"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/perf/perf.cpp > CMakeFiles/pcm.dir/perf/perf.cpp.i

lib/CMakeFiles/pcm.dir/perf/perf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcm.dir/perf/perf.cpp.s"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/perf/perf.cpp -o CMakeFiles/pcm.dir/perf/perf.cpp.s

lib/CMakeFiles/pcm.dir/perf/perf.cpp.o.requires:

.PHONY : lib/CMakeFiles/pcm.dir/perf/perf.cpp.o.requires

lib/CMakeFiles/pcm.dir/perf/perf.cpp.o.provides: lib/CMakeFiles/pcm.dir/perf/perf.cpp.o.requires
	$(MAKE) -f lib/CMakeFiles/pcm.dir/build.make lib/CMakeFiles/pcm.dir/perf/perf.cpp.o.provides.build
.PHONY : lib/CMakeFiles/pcm.dir/perf/perf.cpp.o.provides

lib/CMakeFiles/pcm.dir/perf/perf.cpp.o.provides.build: lib/CMakeFiles/pcm.dir/perf/perf.cpp.o


lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o: lib/CMakeFiles/pcm.dir/flags.make
lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o: lib/pcm/cpucounters.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o -c /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/cpucounters.cpp

lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcm.dir/pcm/cpucounters.cpp.i"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/cpucounters.cpp > CMakeFiles/pcm.dir/pcm/cpucounters.cpp.i

lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcm.dir/pcm/cpucounters.cpp.s"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/cpucounters.cpp -o CMakeFiles/pcm.dir/pcm/cpucounters.cpp.s

lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o.requires:

.PHONY : lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o.requires

lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o.provides: lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o.requires
	$(MAKE) -f lib/CMakeFiles/pcm.dir/build.make lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o.provides.build
.PHONY : lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o.provides

lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o.provides.build: lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o


lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o: lib/CMakeFiles/pcm.dir/flags.make
lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o: lib/pcm/msr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcm.dir/pcm/msr.cpp.o -c /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/msr.cpp

lib/CMakeFiles/pcm.dir/pcm/msr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcm.dir/pcm/msr.cpp.i"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/msr.cpp > CMakeFiles/pcm.dir/pcm/msr.cpp.i

lib/CMakeFiles/pcm.dir/pcm/msr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcm.dir/pcm/msr.cpp.s"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/msr.cpp -o CMakeFiles/pcm.dir/pcm/msr.cpp.s

lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o.requires:

.PHONY : lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o.requires

lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o.provides: lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o.requires
	$(MAKE) -f lib/CMakeFiles/pcm.dir/build.make lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o.provides.build
.PHONY : lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o.provides

lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o.provides.build: lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o


lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o: lib/CMakeFiles/pcm.dir/flags.make
lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o: lib/pcm/client_bw.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcm.dir/pcm/client_bw.cpp.o -c /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/client_bw.cpp

lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcm.dir/pcm/client_bw.cpp.i"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/client_bw.cpp > CMakeFiles/pcm.dir/pcm/client_bw.cpp.i

lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcm.dir/pcm/client_bw.cpp.s"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/client_bw.cpp -o CMakeFiles/pcm.dir/pcm/client_bw.cpp.s

lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o.requires:

.PHONY : lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o.requires

lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o.provides: lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o.requires
	$(MAKE) -f lib/CMakeFiles/pcm.dir/build.make lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o.provides.build
.PHONY : lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o.provides

lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o.provides.build: lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o


lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o: lib/CMakeFiles/pcm.dir/flags.make
lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o: lib/pcm/pci.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcm.dir/pcm/pci.cpp.o -c /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/pci.cpp

lib/CMakeFiles/pcm.dir/pcm/pci.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcm.dir/pcm/pci.cpp.i"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/pci.cpp > CMakeFiles/pcm.dir/pcm/pci.cpp.i

lib/CMakeFiles/pcm.dir/pcm/pci.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcm.dir/pcm/pci.cpp.s"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/pci.cpp -o CMakeFiles/pcm.dir/pcm/pci.cpp.s

lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o.requires:

.PHONY : lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o.requires

lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o.provides: lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o.requires
	$(MAKE) -f lib/CMakeFiles/pcm.dir/build.make lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o.provides.build
.PHONY : lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o.provides

lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o.provides.build: lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o


lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o: lib/CMakeFiles/pcm.dir/flags.make
lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o: lib/pcm/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcm.dir/pcm/utils.cpp.o -c /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/utils.cpp

lib/CMakeFiles/pcm.dir/pcm/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcm.dir/pcm/utils.cpp.i"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/utils.cpp > CMakeFiles/pcm.dir/pcm/utils.cpp.i

lib/CMakeFiles/pcm.dir/pcm/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcm.dir/pcm/utils.cpp.s"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/utils.cpp -o CMakeFiles/pcm.dir/pcm/utils.cpp.s

lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o.requires:

.PHONY : lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o.requires

lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o.provides: lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o.requires
	$(MAKE) -f lib/CMakeFiles/pcm.dir/build.make lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o.provides.build
.PHONY : lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o.provides

lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o.provides.build: lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o


lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o: lib/CMakeFiles/pcm.dir/flags.make
lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o: lib/pcm/MacMSRDriver/MSRAccessor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o -c /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/MacMSRDriver/MSRAccessor.cpp

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.i"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/MacMSRDriver/MSRAccessor.cpp > CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.i

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.s"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/MacMSRDriver/MSRAccessor.cpp -o CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.s

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o.requires:

.PHONY : lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o.requires

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o.provides: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o.requires
	$(MAKE) -f lib/CMakeFiles/pcm.dir/build.make lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o.provides.build
.PHONY : lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o.provides

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o.provides.build: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o


lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o: lib/CMakeFiles/pcm.dir/flags.make
lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o: lib/pcm/MacMSRDriver/DriverInterface.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o   -c /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/MacMSRDriver/DriverInterface.c

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.i"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/MacMSRDriver/DriverInterface.c > CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.i

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.s"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/MacMSRDriver/DriverInterface.c -o CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.s

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o.requires:

.PHONY : lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o.requires

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o.provides: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o.requires
	$(MAKE) -f lib/CMakeFiles/pcm.dir/build.make lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o.provides.build
.PHONY : lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o.provides

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o.provides.build: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o


lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o: lib/CMakeFiles/pcm.dir/flags.make
lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o: lib/pcm/MacMSRDriver/PCIDriverInterface.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o -c /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/MacMSRDriver/PCIDriverInterface.cpp

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.i"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/MacMSRDriver/PCIDriverInterface.cpp > CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.i

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.s"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/pcm/MacMSRDriver/PCIDriverInterface.cpp -o CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.s

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o.requires:

.PHONY : lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o.requires

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o.provides: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o.requires
	$(MAKE) -f lib/CMakeFiles/pcm.dir/build.make lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o.provides.build
.PHONY : lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o.provides

lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o.provides.build: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o


# Object files for target pcm
pcm_OBJECTS = \
"CMakeFiles/pcm.dir/perf/perf.cpp.o" \
"CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o" \
"CMakeFiles/pcm.dir/pcm/msr.cpp.o" \
"CMakeFiles/pcm.dir/pcm/client_bw.cpp.o" \
"CMakeFiles/pcm.dir/pcm/pci.cpp.o" \
"CMakeFiles/pcm.dir/pcm/utils.cpp.o" \
"CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o" \
"CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o" \
"CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o"

# External object files for target pcm
pcm_EXTERNAL_OBJECTS =

bin/libpcm.a: lib/CMakeFiles/pcm.dir/perf/perf.cpp.o
bin/libpcm.a: lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o
bin/libpcm.a: lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o
bin/libpcm.a: lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o
bin/libpcm.a: lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o
bin/libpcm.a: lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o
bin/libpcm.a: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o
bin/libpcm.a: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o
bin/libpcm.a: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o
bin/libpcm.a: lib/CMakeFiles/pcm.dir/build.make
bin/libpcm.a: lib/CMakeFiles/pcm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX static library ../bin/libpcm.a"
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && $(CMAKE_COMMAND) -P CMakeFiles/pcm.dir/cmake_clean_target.cmake
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pcm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/CMakeFiles/pcm.dir/build: bin/libpcm.a

.PHONY : lib/CMakeFiles/pcm.dir/build

lib/CMakeFiles/pcm.dir/requires: lib/CMakeFiles/pcm.dir/perf/perf.cpp.o.requires
lib/CMakeFiles/pcm.dir/requires: lib/CMakeFiles/pcm.dir/pcm/cpucounters.cpp.o.requires
lib/CMakeFiles/pcm.dir/requires: lib/CMakeFiles/pcm.dir/pcm/msr.cpp.o.requires
lib/CMakeFiles/pcm.dir/requires: lib/CMakeFiles/pcm.dir/pcm/client_bw.cpp.o.requires
lib/CMakeFiles/pcm.dir/requires: lib/CMakeFiles/pcm.dir/pcm/pci.cpp.o.requires
lib/CMakeFiles/pcm.dir/requires: lib/CMakeFiles/pcm.dir/pcm/utils.cpp.o.requires
lib/CMakeFiles/pcm.dir/requires: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/MSRAccessor.cpp.o.requires
lib/CMakeFiles/pcm.dir/requires: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/DriverInterface.c.o.requires
lib/CMakeFiles/pcm.dir/requires: lib/CMakeFiles/pcm.dir/pcm/MacMSRDriver/PCIDriverInterface.cpp.o.requires

.PHONY : lib/CMakeFiles/pcm.dir/requires

lib/CMakeFiles/pcm.dir/clean:
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib && $(CMAKE_COMMAND) -P CMakeFiles/pcm.dir/cmake_clean.cmake
.PHONY : lib/CMakeFiles/pcm.dir/clean

lib/CMakeFiles/pcm.dir/depend:
	cd /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib /Users/lostbenjamin/Documents/ETH_Zurich/courses/fast_code/fast-xnor-net/lib/CMakeFiles/pcm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/CMakeFiles/pcm.dir/depend

