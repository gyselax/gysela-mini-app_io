add_executable(gys_compress gys_compress.cpp predcorr_compress.cpp)

target_link_libraries(gys_compress
    PUBLIC
        mini_app_io_lib
        DDC::core
        PDI::PDI_C
        paraconf::paraconf
        MPI::MPI_CXX
        gyselalibxx::initialisation_xyvxvy_double
        gyselalibxx::vlasov_xyvxvy_double
        gyselalibxx::poisson_xy_double
        gyselalibxx::advection
        gyselalibxx::speciesinfo
        gyselalibxx::time_integration_xyvxvy_double
        gyselalibxx::io
        gyselalibxx::utils
)

install(TARGETS gys_compress)
