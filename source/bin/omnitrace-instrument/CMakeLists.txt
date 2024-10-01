# ------------------------------------------------------------------------------#
#
# rocprofsys-instrument target (formerly rocprofsys-exe target prior to 1.8.1)
#
# ------------------------------------------------------------------------------#

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

add_executable(rocprofsys-instrument)

target_sources(
    rocprofsys-instrument
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/details.cpp
            ${CMAKE_CURRENT_LIST_DIR}/function_signature.cpp
            ${CMAKE_CURRENT_LIST_DIR}/function_signature.hpp
            ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
            ${CMAKE_CURRENT_LIST_DIR}/info.hpp
            ${CMAKE_CURRENT_LIST_DIR}/internal_libs.cpp
            ${CMAKE_CURRENT_LIST_DIR}/internal_libs.hpp
            ${CMAKE_CURRENT_LIST_DIR}/log.cpp
            ${CMAKE_CURRENT_LIST_DIR}/log.hpp
            ${CMAKE_CURRENT_LIST_DIR}/module_function.cpp
            ${CMAKE_CURRENT_LIST_DIR}/module_function.hpp
            ${CMAKE_CURRENT_LIST_DIR}/omnitrace-instrument.cpp
            ${CMAKE_CURRENT_LIST_DIR}/omnitrace-instrument.hpp)

target_link_libraries(
    rocprofsys-instrument
    PRIVATE rocprofsys::rocprofsys-headers
            rocprofsys::rocprofsys-dyninst
            rocprofsys::rocprofsys-compile-options
            rocprofsys::rocprofsys-compile-definitions
            rocprofsys::rocprofsys-sanitizer
            timemory::timemory-headers
            timemory::timemory-extensions
            timemory::timemory-core)

set_target_properties(
    rocprofsys-instrument
    PROPERTIES BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
               INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}"
               INSTALL_RPATH_USE_LINK_PATH ON
               OUTPUT_NAME "rocprof-sys-instrument")

if(OMNITRACE_BUILD_DYNINST)
    target_compile_definitions(rocprofsys-instrument PRIVATE OMNITRACE_BUILD_DYNINST=1)
endif()

add_target_flag_if_avail(rocprofsys-instrument "-Wno-deprecated-declarations")

omnitrace_strip_target(rocprofsys-instrument)

if(CMAKE_BUILD_TYPE MATCHES "^(DEBUG|Debug)")
    string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    target_compile_options(rocprofsys-instrument PRIVATE ${_FLAGS})
endif()

install(
    TARGETS rocprofsys-instrument
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
