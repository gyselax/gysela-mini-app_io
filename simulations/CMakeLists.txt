option(BUILD_CHECKPOINT_APP "Build the Checkpointing mini app" ON)

if(BUILD_CHECKPOINT_APP)
    add_subdirectory(checkpointing)
endif()
