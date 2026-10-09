option(BUILD_IO_APP "Build the IO mini app" ON)

if(BUILD_IO_APP)
    add_subdirectory(io)
endif()
