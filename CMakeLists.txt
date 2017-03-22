#      _________   _____________________  ____  ______
#     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
#    / /_  / /| | \__ \ / / / /   / / / / / / / __/
#   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
#  /_/   /_/  |_/____//_/  \____/\____/_____/_____/


cmake_minimum_required (VERSION 3.0.2)
set(CMAKE_VERBOSE_MAKEFILE ON)
project (FNC)

# ================================================================================
# Reset the C flags
# ================================================================================
set(CMAKE_C_FLAGS           "")
set(CMAKE_C_FLAGS_RELEASE   "")
set(CMAKE_C_FLAGS_DEBUG     "")

# ================================================================================
# Build the executable
# ================================================================================

# Add the include directories
include_directories(include)

# ================================================================================
# C Global Flags
# ================================================================================

if (CMAKE_C_COMPILER_ID MATCHES "Clang")

    # clang - CC flags
    set(CMAKE_C_FLAGS "-O3")

elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")

    # GNU GCC - CC flags
    set(CMAKE_C_FLAGS "-O3")

endif()

set(SOURCE_FILES
    src/main.c
    src/main.h
)

add_executable(FNC ${SOURCE_FILES})