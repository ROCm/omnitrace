# ----------------------------------------------------------------------------- #
#
# rocprofsys: contains all instrumentation functionality
#
# rocprofsys-dl: contains minimal symbols and dlopen's rocprofsys
#
# rocprofsys-user: contains symbols for user API
#
# ----------------------------------------------------------------------------- #

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
    cmake_policy(SET CMP0115 NEW)
endif()

if(OMNITRACE_USE_ROCPROFILER
   AND rocprofiler_LIBRARY_DIR
   AND ROCmVersion_TRIPLE_VERSION VERSION_LESS 5.2.0
   AND NOT CMAKE_INSTALL_RPATH_USE_LINK_PATH)
    set(OMNITRACE_LIB_INSTALL_RPATH
        "\$ORIGIN:\$ORIGIN/${PACKAGE_NAME}:${rocprofiler_LIBRARY_DIR}")
else()
    set(OMNITRACE_LIB_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/${PACKAGE_NAME}")
endif()

# ------------------------------------------------------------------------------#
#
# rocprofsys interface library
#
# ------------------------------------------------------------------------------#

add_library(rocprofsys-interface-library INTERFACE)
add_library(rocprofsys::rocprofsys-interface-library ALIAS rocprofsys-interface-library)

target_include_directories(
    rocprofsys-interface-library INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                                           ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace)

target_link_libraries(
    rocprofsys-interface-library
    INTERFACE
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-headers>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-threading>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-common-library>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-compile-options>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-compile-definitions>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-perfetto>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-timemory>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-elfutils>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-bfd>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-mpi>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-ptl>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-hip>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-roctracer>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-rocprofiler>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-rocm-smi>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-rccl>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-static-libgcc-optional>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-static-libstdcxx-optional>
        $<BUILD_INTERFACE:rocprofsys::rocprofsys-sanitizer>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${OMNITRACE_BUILD_LTO}>,rocprofsys::rocprofsys-lto,>>
    )

# ------------------------------------------------------------------------------#
#
# rocprofsys internal libraries
#
# ------------------------------------------------------------------------------#

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(binary)

# ------------------------------------------------------------------------------#
#
# rocprofsys exported libraries
#
# ------------------------------------------------------------------------------#

add_subdirectory(omnitrace)
add_subdirectory(omnitrace-dl)
add_subdirectory(omnitrace-rt)
add_subdirectory(omnitrace-user)
