# SPDX-License-Identifier: MIT

# Create a library from the source files
add_library(mini_app_io_lib
    fluid_moments/fluid_moments.cpp
)

target_include_directories(mini_app_io_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fluid_moments>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(mini_app_io_lib PUBLIC
    gyselalibxx::quadrature
    gyselalibxx::utils
    gyselalibxx::speciesinfo
    gyselalibxx::io
    gyselalibxx::mpi_parallelisation
)
