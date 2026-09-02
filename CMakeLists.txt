# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(gysela-mini-app_io C CXX)

###############################################################################################
#                           List of options for Gyselalib++ library
###############################################################################################
set(GYSELALIBXX_BUILD_TESTING OFF)
set(GYSELALIBXX_BUILD_SIMULATIONS OFF)
set(GYSELALIBXX_BUILD_DOCUMENTATION OFF)


###############################################################################################
#                              Build libraries and executables
###############################################################################################

## Link to Gyselalibxx++ library
add_subdirectory(src/external/gyselalibxx)

## Build the C++ source library in `src/C++/`
add_subdirectory(src/C++)

## Build the simulations (applications mains) in `apps/`
add_subdirectory(apps/)
