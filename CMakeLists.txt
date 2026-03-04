# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(gysela-mini-app_io C CXX)

## Use CTest for running tests
include(CTest)

find_package(GTest "1.12" QUIET)

## Look for a pre-installed paraconf
find_package(paraconf REQUIRED COMPONENTS C)

###############################################################################################
#                           List of options for Gyselalib++ library
###############################################################################################
set(GYSELALIBXX_BUILD_TESTING OFF)
set(GYSELALIBXX_BUILD_SIMULATIONS OFF)
set(GYSELALIBXX_BUILD_DOCUMENTATION OFF)
set(GYSELALIBXX_INCLUDE_TESTING_DEPENDENCIES ON)


###############################################################################################
#                                   List of options
###############################################################################################
option(GYSELA_MINI_APP_COMPILE_SOURCE "Enable compilation of the source code (this should be set to off to build documentation without the C++ dependencies)" ON)
option(GYSELA_MINI_APP_BUILD_DOCUMENTATION "Build the documentation." OFF)
option(GYSELA_MINI_APP_BUILD_SIMULATIONS "Build the simulations." ON)


###############################################################################################
#                              Build libraries and executables
###############################################################################################

if ("${GYSELA_MINI_APP_COMPILE_SOURCE}")

set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

## Link to Gyselalibxx++ library
add_subdirectory(external/gyselalibxx)

## Change the C++ flags to the Gyselalibxx++ specific ones
set(CMAKE_CXX_FLAGS "${GYSELALIBXX_DEFAULT_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

## Get the main code in `src/`
add_subdirectory(src/)

## Build the simulations (applications mains) in `apps/`
if("${GYSELA_MINI_APP_BUILD_SIMULATIONS}")
    add_subdirectory(apps/)
endif()

endif() # GYSELA_MINI_APP_COMPILE_SOURCE
