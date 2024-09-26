#
#   function for
#
include_guard(DIRECTORY)

if(NOT TARGET rocprofsys::rocprofsys-user-library)
    find_package(rocprofsys REQUIRED COMPONENTS user)
endif()

if(NOT coz-profiler_FOUND)
    find_package(coz-profiler QUIET)
endif()

if(NOT TARGET omni-causal-examples)
    add_custom_target(omni-causal-examples)
endif()

function(omnitrace_causal_example_executable _NAME)
    cmake_parse_arguments(
        CAUSAL "" "" "SOURCES;DEFINITIONS;INCLUDE_DIRECTORIES;LINK_LIBRARIES" ${ARGN})

    function(omnitrace_causal_example_interface _TARGET)
        if(NOT TARGET ${_TARGET})
            find_package(Threads REQUIRED)
            add_library(${_TARGET} INTERFACE)
            target_link_libraries(${_TARGET} INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
        endif()
    endfunction()

    omnitrace_causal_example_interface(omni-causal-example-lib-debug)
    omnitrace_causal_example_interface(omni-causal-example-lib-no-debug)

    target_compile_options(omni-causal-example-lib-debug
                           INTERFACE -g3 -fno-omit-frame-pointer)
    target_compile_options(omni-causal-example-lib-no-debug INTERFACE -g0)

    add_executable(${_NAME} ${CAUSAL_SOURCES})
    target_compile_definitions(${_NAME} PRIVATE USE_COZ=0 USE_OMNI=0
                                                ${CAUSAL_DEFINITIONS})
    target_include_directories(${_NAME} PRIVATE ${OMNITRACE_EXAMPLE_ROOT_DIR}/causal
                                                ${CAUSAL_INCLUDE_DIRECTORIES})
    target_link_libraries(
        ${_NAME} PRIVATE ${CAUSAL_LINK_LIBRARIES} rocprofsys::rocprofsys-user-library
                         omni-causal-example-lib-debug)

    add_executable(${_NAME}-omni ${CAUSAL_SOURCES})
    target_compile_definitions(${_NAME}-omni PRIVATE USE_COZ=0 USE_OMNI=1
                                                     ${CAUSAL_DEFINITIONS})
    target_include_directories(${_NAME}-omni PRIVATE ${OMNITRACE_EXAMPLE_ROOT_DIR}/causal
                                                     ${CAUSAL_INCLUDE_DIRECTORIES})
    target_link_libraries(
        ${_NAME}-omni
        PRIVATE ${CAUSAL_LINK_LIBRARIES} rocprofsys::rocprofsys-user-library
                omni-causal-example-lib-debug)

    add_executable(${_NAME}-ndebug ${CAUSAL_SOURCES})
    target_compile_definitions(${_NAME}-ndebug PRIVATE USE_COZ=0 USE_OMNI=0
                                                       ${CAUSAL_DEFINITIONS})
    target_include_directories(
        ${_NAME}-ndebug PRIVATE ${OMNITRACE_EXAMPLE_ROOT_DIR}/causal
                                ${CAUSAL_INCLUDE_DIRECTORIES})
    target_link_libraries(
        ${_NAME}-ndebug
        PRIVATE ${CAUSAL_LINK_LIBRARIES} rocprofsys::rocprofsys-user-library
                omni-causal-example-lib-no-debug)

    add_executable(${_NAME}-omni-ndebug ${CAUSAL_SOURCES})
    target_compile_definitions(${_NAME}-omni-ndebug PRIVATE USE_COZ=0 USE_OMNI=1
                                                            ${CAUSAL_DEFINITIONS})
    target_include_directories(
        ${_NAME}-omni-ndebug PRIVATE ${OMNITRACE_EXAMPLE_ROOT_DIR}/causal
                                     ${CAUSAL_INCLUDE_DIRECTORIES})
    target_link_libraries(
        ${_NAME}-omni-ndebug
        PRIVATE ${CAUSAL_LINK_LIBRARIES} rocprofsys::rocprofsys-user-library
                omni-causal-example-lib-no-debug)

    add_dependencies(omni-causal-examples ${_NAME} ${_NAME}-omni ${_NAME}-ndebug
                     ${_NAME}-omni-ndebug)

    if(coz-profiler_FOUND)
        omnitrace_causal_example_interface(omni-causal-example-lib-coz)
        target_compile_options(omni-causal-example-lib-coz
                               INTERFACE -g3 -gdwarf-3 -fno-omit-frame-pointer)

        add_executable(${_NAME}-coz ${CAUSAL_SOURCES})
        target_compile_definitions(${_NAME}-coz PRIVATE USE_COZ=1 USE_OMNI=0
                                                        ${CAUSAL_DEFINITIONS})
        target_include_directories(
            ${_NAME}-coz PRIVATE ${OMNITRACE_EXAMPLE_ROOT_DIR}/causal
                                 ${CAUSAL_INCLUDE_DIRECTORIES})
        target_link_libraries(${_NAME}-coz PRIVATE ${CAUSAL_LINK_LIBRARIES}
                                                   omni-causal-example-lib-coz coz::coz)

        add_dependencies(omni-causal-examples ${_NAME}-coz)
    endif()

    if(OMNITRACE_INSTALL_EXAMPLES)
        install(
            TARGETS ${_NAME} ${_NAME}-omni ${_NAME}-coz
            DESTINATION bin
            COMPONENT rocprofsys-examples
            OPTIONAL)
    endif()
endfunction()
