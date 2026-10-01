# SPDX-License-Identifier: MIT

add_executable(gys_io
    ${PROJECT_SOURCE_DIR}/src/io/gys_io.cpp
)

target_include_directories(gys_io PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/io
)

target_link_libraries(gys_io
    PUBLIC
        gysela_mini_app_lib
        DDC::core
        DDC::pdi
        PDI::PDI_C
        paraconf::paraconf
        MPI::MPI_CXX
)

# Set PYTHONPATH to the repo root so the `processing` python package is importable
get_filename_component(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR} ABSOLUTE)
set(PYTHON_MODULE_PATH "${PROJECT_ROOT_DIR}")
set_target_properties(gys_io PROPERTIES
    ENVIRONMENT "PYTHONPATH=${PYTHON_MODULE_PATH}:$ENV{PYTHONPATH}"
)

install(TARGETS gys_io)
