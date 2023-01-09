#
#   function for
#
include_guard(DIRECTORY)

if(NOT TARGET omnitrace::omnitrace-user-library)
    find_package(omnitrace REQUIRED COMPONENTS user)
endif()

if(NOT coz-profiler_FOUND)
    find_package(coz-profiler)
endif()

function(omnitrace_causal_example_executable _NAME)
    cmake_parse_arguments(CAUSAL "" "TAG"
        "SOURCES;DEFINITIONS;INCLUDE_DIRECTORIES;LINK_LIBRARIES" ${ARGN})

    if(CAUSAL_TAG)
        set(_SUFFIX "-${CAUSAL_TAG}")
    else()
        set(_SUFFIX "")
    endif()

    add_executable(${_NAME}${_SUFFIX} ${CAUSAL_SOURCES})
    target_compile_definitions(${_NAME}${_SUFFIX} PRIVATE USE_COZ=0 USE_OMNI=0
                                                          ${CAUSAL_DEFINITIONS})
    target_include_directories(
        ${_NAME}${_SUFFIX} PRIVATE ${OMNITRACE_EXAMPLE_ROOT_DIR}/causal
                                   ${CAUSAL_INCLUDE_DIRECTORIES})
    target_link_libraries(${_NAME}${_SUFFIX} PRIVATE ${CAUSAL_LINK_LIBRARIES}
                                                     omnitrace::omnitrace-user-library)

    add_executable(${_NAME}-omni${_SUFFIX} ${CAUSAL_SOURCES})
    target_compile_definitions(${_NAME}-omni${_SUFFIX} PRIVATE USE_COZ=0 USE_OMNI=1
                                                               ${CAUSAL_DEFINITIONS})
    target_include_directories(
        ${_NAME}-omni${_SUFFIX} PRIVATE ${OMNITRACE_EXAMPLE_ROOT_DIR}/causal
                                        ${CAUSAL_INCLUDE_DIRECTORIES})
    target_link_libraries(
        ${_NAME}-omni${_SUFFIX} PRIVATE ${CAUSAL_LINK_LIBRARIES}
                                        omnitrace::omnitrace-user-library)

    if(coz-profiler_FOUND)
        add_executable(${_NAME}-coz${_SUFFIX} ${CAUSAL_SOURCES})
        target_compile_definitions(${_NAME}-coz${_SUFFIX} PRIVATE USE_COZ=1 USE_OMNI=0
                                                                  ${CAUSAL_DEFINITIONS})
        target_include_directories(
            ${_NAME}-coz${_SUFFIX} PRIVATE ${OMNITRACE_EXAMPLE_ROOT_DIR}/causal
                                           ${CAUSAL_INCLUDE_DIRECTORIES})
        target_link_libraries(${_NAME}-coz${_SUFFIX} PRIVATE ${CAUSAL_LINK_LIBRARIES}
                                                             coz::coz)
    endif()

    if(OMNITRACE_INSTALL_EXAMPLES)
        install(
            TARGETS ${_NAME}${_SUFFIX} ${_NAME}-omni${_SUFFIX} ${_NAME}-coz${_SUFFIX}
            DESTINATION bin
            COMPONENT omnitrace-examples
            OPTIONAL)
    endif()
endfunction()
