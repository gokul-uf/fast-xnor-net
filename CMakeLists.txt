#      _________   _____________________  ____  ______
#     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
#    / /_  / /| | \__ \ / / / /   / / / / / / / __/
#   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
#  /_/   /_/  |_/____//_/  \____/\____/_____/_____/


cmake_minimum_required (VERSION 3.0.2)
set(CMAKE_VERBOSE_MAKEFILE ON)
project (FNC_XNORNET)
include(CMakeLibs.cmake)

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
    src/xnornet.c
    src/xnornet.h
    src/conv_layer.c
    src/conv_layer.h
    src/fully_con_layer.c
    src/fully_con_layer.h
    src/mnist_wrapper.c
    src/mnist_wrapper.h
    src/pool_layer.c
    src/pool_layer.h
    src/tensor.c
    src/tensor.h
    src/back_prop.c
    src/back_prop.h
)

add_executable(FNC_XNORNET ${SOURCE_FILES})

target_link_libraries(FNC_XNORNET m)
target_link_libraries(FNC_XNORNET pcm)
if (DEFINED PCM_DEPENDENCIES)
    add_dependencies(FNC_XNORNET ${PCM_DEPENDENCIES})
endif(DEFINED PCM_DEPENDENCIES)
