cmake_minimum_required(VERSION 3.16)
project(gllib VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL REQUIRED)

add_subdirectory(gllib)

option(GLLIB_BUILD_EXAMPLES "Build examples" ON)
if(GLLIB_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

find_package(Doxygen QUIET)
if(Doxygen_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
    )
endif()
