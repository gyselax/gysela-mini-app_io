# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(gysela-mini-app_io CXX)

option(GYSELA_MINI_APP_USE_EXTERNAL_GYSELALIBXX "Use an external Gyselalib++ installation" OFF)

###############################################################################################
#                              Build libraries and executables
###############################################################################################

find_package(DDC REQUIRED COMPONENTS pdi)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(paraconf REQUIRED COMPONENTS C)
find_package(PDI REQUIRED COMPONENTS C)

## Link to Gyselalibxx++ library
if(GYSELA_MINI_APP_USE_EXTERNAL_GYSELALIBXX)
  find_package(gyselalibxx REQUIRED)
else()
  set(GYSELALIBXX_BUILD_TESTING OFF)
  set(GYSELALIBXX_BUILD_SIMULATIONS OFF)
  set(GYSELALIBXX_BUILD_DOCUMENTATION OFF)
  add_subdirectory(src/external/gyselalibxx)
endif()

## Build the C++ source library in `src/C++/`
add_subdirectory(src/C++)

## Build the simulations (applications mains) in `apps/`
add_subdirectory(apps/)
