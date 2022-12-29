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

add_library(causal-interface-library INTERFACE)
target_compile_options(causal-interface-library INTERFACE -g3 -gdwarf-3
                                                          -fno-omit-frame-pointer)
target_link_libraries(causal-interface-library INTERFACE Threads::Threads
                                                         ${CMAKE_DL_LIBS})

find_package(coz-profiler)

function(omni_causal_add_executable _SUFFIX)
    add_executable(causal${_SUFFIX} causal.cpp impl.cpp)
    target_compile_definitions(causal${_SUFFIX} PRIVATE ${ARGN})
    target_link_libraries(causal${_SUFFIX} PRIVATE causal-interface-library
                                                   omnitrace::omnitrace-user-library)

    add_executable(causal-omni${_SUFFIX} causal.cpp impl.cpp)
    target_compile_definitions(causal-omni${_SUFFIX} PRIVATE USE_OMNI=1 ${ARGN})
    target_link_libraries(causal-omni${_SUFFIX} PRIVATE causal-interface-library
                                                        omnitrace::omnitrace-user-library)

    if(coz-profiler_FOUND)
        add_executable(causal-coz${_SUFFIX} causal.cpp impl.cpp)
        target_compile_definitions(causal-coz${_SUFFIX} PRIVATE USE_COZ=1 ${ARGN})
        target_link_libraries(causal-coz${_SUFFIX} PRIVATE causal-interface-library
                                                           coz::coz)
    endif()

    if(OMNITRACE_INSTALL_EXAMPLES)
        install(
            TARGETS causal${_SUFFIX} causal-omni${_SUFFIX} causal-coz${_SUFFIX}
            DESTINATION bin
            COMPONENT omnitrace-examples
            OPTIONAL)
    endif()

endfunction()

omni_causal_add_executable("" USE_RNG=1 USE_CPU=1)
omni_causal_add_executable("-rng" USE_RNG=1 USE_CPU=0)
omni_causal_add_executable("-cpu" USE_RNG=0 USE_CPU=1)
