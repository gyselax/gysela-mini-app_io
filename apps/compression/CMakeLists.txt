add_executable(gys_compress gys_compress.cpp predcorr_compress.cpp)

target_link_libraries(gys_compress
    PUBLIC
        mini_app_io_lib
        DDC::core
        PDI::PDI_C
        paraconf::paraconf
        MPI::MPI_CXX
        gslx::initialisation_xyvxvy_double
        gslx::vlasov_xyvxvy_double
        gslx::poisson_xy_double
        gslx::advection
        gslx::speciesinfo
        gslx::time_integration_xyvxvy_double
        gslx::io
        gslx::utils
)

install(TARGETS gys_compress)
