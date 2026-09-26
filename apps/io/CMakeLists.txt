# SPDX-License-Identifier: MIT

add_executable(gys_io
    gys_io.cpp
)

option(SUBFILING "Enable HDF5 Subfiling support" OFF)

if(SUBFILING)
	target_compile_definitions(gys_io PRIVATE SUBFILING)
endif()

target_link_libraries(gys_io
    PUBLIC
        mini_app_io_lib
        DDC::core
        DDC::pdi
        PDI::PDI_C
        paraconf::paraconf
        MPI::MPI_CXX
)

install(TARGETS gys_io)
