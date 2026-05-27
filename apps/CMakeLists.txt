option(BUILD_IO_APP "Build the IO mini app" ON)
option(BUILD_COMPRESSION_APP "Build the Compression mini app" ON)

if(BUILD_IO_APP)
    add_subdirectory(io)
endif()

if(BUILD_COMPRESSION_APP)
    add_subdirectory(compression)
endif()
