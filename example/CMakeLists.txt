set (SCALGRAD_EXMAPLE_SOURCES
    src/main.cc
)

add_executable(scalgrad-example ${SCALGRAD_EXMAPLE_SOURCES})
target_include_directories(scalgrad-example PRIVATE ${SCALGRAD_INCLUDE_DIR})


install (TARGETS scalgrad-example DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin )