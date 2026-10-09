# SPDX-License-Identifier: MIT

add_executable(gys_checkpointing
    ${PROJECT_SOURCE_DIR}/src/checkpointing/gys_checkpointing.cpp
)

target_link_libraries(gys_checkpointing
    PUBLIC
        mini_app_io_lib
        DDC::core
        DDC::pdi
        PDI::PDI_C
        paraconf::paraconf
        MPI::MPI_CXX
)

install(TARGETS gys_checkpointing)
