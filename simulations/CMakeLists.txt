option(BUILD_CHECKPOINT_APP "Build the Checkpointing mini app" ON)
option(BUILD_PDI_WM_APP "Build the PDI Write Methods mini app" ON)

if(BUILD_CHECKPOINT_APP)
    add_subdirectory(checkpointing)
endif()

if(BUILD_PDI_WM_APP)
    add_subdirectory(pdi-write-methods)
endif()
