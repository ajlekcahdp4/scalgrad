cmake_minimum_required(VERSION 3.14)

project(nanograd)

set (NANOGRAD_TEST_SOURCES
    main.cc
)

add_executable(nanograd-test ${NANOGRAD_TEST_SOURCES})