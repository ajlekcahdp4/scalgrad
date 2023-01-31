set (SCALGRAD_EXMAPLE_SOURCES
    main.cc
)

add_executable(scalgrad-example ${SCALGRAD_EXMAPLE_SOURCES})
target_include_directories(scalgrad-example PRIVATE ${SCALGRAD_INCLUDE_DIR})