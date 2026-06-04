add_executable(compression_app landau2X2V_compression.cpp)

target_include_directories(compression_app 
    PRIVATE 
        "${CMAKE_SOURCE_DIR}/src/external/gyselalibxx/simulations/geometryXYVxVy"
        "${CMAKE_SOURCE_DIR}/src/external/gyselalibxx/simulations/geometryXVx"
)

target_link_libraries(compression_app
    PUBLIC
        mini_app_io_lib
        DDC::core
        PDI::PDI_C
        paraconf::paraconf
        MPI::MPI_CXX
        gslx::initialisation_xyvxvy
        gslx::vlasov_xyvxvy
        gslx::poisson_xy
        gslx::advection
        gslx::speciesinfo
        gslx::time_integration_xyvxvy
        gslx::io
        gslx::utils
)

get_filename_component(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR} ABSOLUTE)
set(PYTHON_MODULE_PATH "${PROJECT_ROOT_DIR}/src/python")
set_target_properties(compression_app PROPERTIES
    ENVIRONMENT "PYTHONPATH=${PYTHON_MODULE_PATH}:$ENV{PYTHONPATH}"
)

install(TARGETS compression_app)
