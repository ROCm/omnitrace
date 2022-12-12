cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-causal-example LANGUAGES CXX)

if(OMNITRACE_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(${PROJECT_NAME} IN_LIST OMNITRACE_DISABLE_EXAMPLES OR ${_DIR} IN_LIST
                                                             OMNITRACE_DISABLE_EXAMPLES)
        return()
    endif()
endif()

set(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)
if(NOT TARGET omnitrace::omnitrace-user-library)
    find_package(omnitrace REQUIRED COMPONENTS user)
endif()
add_executable(causal causal.cpp)
target_compile_options(causal PRIVATE -g -gdwarf-3)
target_link_libraries(causal PRIVATE Threads::Threads omnitrace::omnitrace-user-library
                                     ${CMAKE_DL_LIBS})

if(OMNITRACE_INSTALL_EXAMPLES)
    install(
        TARGETS causal
        DESTINATION bin
        COMPONENT omnitrace-examples)
endif()
