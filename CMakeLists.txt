# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(gysela-mini-app_io C CXX)

## Use CTest for running tests
include(CTest)

find_package(GTest "1.12" QUIET)

## Look for a pre-installed paraconf
find_package(paraconf REQUIRED COMPONENTS C)

set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
add_link_options(-no-pie)

###############################################################################################
#                           List of options for Gyselalib++ library
###############################################################################################
set(GYSELALIBXX_BUILD_TESTING OFF)
set(GYSELALIBXX_BUILD_SIMULATIONS OFF)
set(GYSELALIBXX_BUILD_DOCUMENTATION OFF)
set(GYSELALIBXX_INCLUDE_TESTING_DEPENDENCIES ON)


###############################################################################################
#                              Build libraries and executables
###############################################################################################

set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

## Link to Gyselalibxx++ library
add_subdirectory(src/external/gyselalibxx)

## Change the C++ flags to the Gyselalibxx++ specific ones
set(CMAKE_CXX_FLAGS "${GYSELALIBXX_DEFAULT_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

## Build the C++ source library in `src/C++/`
add_subdirectory(src/C++)

## Build the simulations (applications mains) in `apps/`
add_subdirectory(apps/)
