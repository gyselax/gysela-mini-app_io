# SPDX-License-Identifier: MIT

add_executable(gys_write_methods
    ${PROJECT_SOURCE_DIR}/src/pdi-write-methods/gys_write_methods.cpp
)

target_link_libraries(gys_write_methods
    PUBLIC
        mini_app_io_lib
        DDC::core
        DDC::pdi
        PDI::PDI_C
        paraconf::paraconf
        MPI::MPI_CXX
)

install(TARGETS gys_write_methods)
