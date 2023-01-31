set (UNIT_TEST_SOURCES
    src/main.cc
    src/test-engine.cc
)

if (${ENABLE_GTEST})
    add_executable(unit-test ${UNIT_TEST_SOURCES})
    target_include_directories(unit-test PUBLIC ${SCALGRAD_INCLUDE_DIR})
    target_link_libraries(unit-test ${GTEST_BOTH_LIBRARIES})
    gtest_discover_tests(unit-test)
endif()