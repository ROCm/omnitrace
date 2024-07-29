# configure packaging

function(omnitrace_parse_release)
    if(EXISTS /etc/lsb-release AND NOT IS_DIRECTORY /etc/lsb-release)
        file(READ /etc/lsb-release _LSB_RELEASE)
        if(_LSB_RELEASE)
            string(REGEX
                   REPLACE "DISTRIB_ID=(.*)\nDISTRIB_RELEASE=(.*)\nDISTRIB_CODENAME=.*"
                           "\\1-\\2" _SYSTEM_NAME "${_LSB_RELEASE}")
        endif()
    elseif(EXISTS /etc/os-release AND NOT IS_DIRECTORY /etc/os-release)
        file(READ /etc/os-release _OS_RELEASE)
        if(_OS_RELEASE)
            string(REPLACE "\"" "" _OS_RELEASE "${_OS_RELEASE}")
            string(REPLACE "-" " " _OS_RELEASE "${_OS_RELEASE}")
            string(REGEX REPLACE "NAME=.*\nVERSION=([0-9\.]+).*\nID=([a-z]+).*" "\\2-\\1"
                                 _SYSTEM_NAME "${_OS_RELEASE}")
        endif()
    endif()
    string(TOLOWER "${_SYSTEM_NAME}" _SYSTEM_NAME)
    if(NOT _SYSTEM_NAME)
        set(_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
    endif()
    set(_SYSTEM_NAME
        "${_SYSTEM_NAME}"
        PARENT_SCOPE)
endfunction()

# parse either /etc/lsb-release or /etc/os-release
omnitrace_parse_release()

if(NOT _SYSTEM_NAME)
    set(_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
endif()

# Add packaging directives
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Runtime instrumentation and binary rewriting for Perfetto via Dyninst")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT "jonathan.madsen@amd.com")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(OMNITRACE_CPACK_SYSTEM_NAME
    "${_SYSTEM_NAME}"
    CACHE STRING "System name, e.g. Linux or Ubuntu-20.04")
set(OMNITRACE_CPACK_PACKAGE_SUFFIX "")

if(OMNITRACE_USE_HIP
   OR OMNITRACE_USE_ROCTRACER
   OR OMNITRACE_USE_ROCM_SMI)
    set(OMNITRACE_CPACK_PACKAGE_SUFFIX
        "${OMNITRACE_CPACK_PACKAGE_SUFFIX}-ROCm-${ROCmVersion_NUMERIC_VERSION}")
endif()

if(OMNITRACE_USE_PAPI)
    set(OMNITRACE_CPACK_PACKAGE_SUFFIX "${OMNITRACE_CPACK_PACKAGE_SUFFIX}-PAPI")
endif()

if(OMNITRACE_USE_OMPT)
    set(OMNITRACE_CPACK_PACKAGE_SUFFIX "${OMNITRACE_CPACK_PACKAGE_SUFFIX}-OMPT")
endif()

if(OMNITRACE_USE_MPI)
    set(VALID_MPI_IMPLS "mpich" "openmpi")
    if("${MPI_C_COMPILER_INCLUDE_DIRS};${MPI_C_HEADER_DIR}" MATCHES "openmpi")
        set(OMNITRACE_MPI_IMPL "openmpi")
    elseif("${MPI_C_COMPILER_INCLUDE_DIRS};${MPI_C_HEADER_DIR}" MATCHES "mpich")
        set(OMNITRACE_MPI_IMPL "mpich")
    else()
        message(
            WARNING
                "MPI implementation could not be determined. Please set OMNITRACE_MPI_IMPL to one of the following for CPack: ${VALID_MPI_IMPLS}"
            )
    endif()
    if(OMNITRACE_MPI_IMPL AND NOT "${OMNITRACE_MPI_IMPL}" IN_LIST VALID_MPI_IMPLS)
        message(
            SEND_ERROR
                "Invalid OMNITRACE_MPI_IMPL (${OMNITRACE_MPI_IMPL}). Should be one of: ${VALID_MPI_IMPLS}"
            )
    else()
        omnitrace_add_feature(OMNITRACE_MPI_IMPL
                              "MPI implementation for CPack DEBIAN depends")
    endif()

    if("${OMNITRACE_MPI_IMPL}" STREQUAL "openmpi")
        set(OMNITRACE_MPI_IMPL_UPPER "OpenMPI")
    elseif("${OMNITRACE_MPI_IMPL}" STREQUAL "mpich")
        set(OMNITRACE_MPI_IMPL_UPPER "MPICH")
    else()
        set(OMNITRACE_MPI_IMPL_UPPER "MPI")
    endif()
    set(OMNITRACE_CPACK_PACKAGE_SUFFIX
        "${OMNITRACE_CPACK_PACKAGE_SUFFIX}-${OMNITRACE_MPI_IMPL_UPPER}")
endif()

if(OMNITRACE_USE_PYTHON)
    set(_OMNITRACE_PYTHON_NAME "Python3")
    foreach(_VER ${OMNITRACE_PYTHON_VERSIONS})
        if("${_VER}" VERSION_LESS 3.0.0)
            set(_OMNITRACE_PYTHON_NAME "Python")
        endif()
    endforeach()
    set(OMNITRACE_CPACK_PACKAGE_SUFFIX "${OMNITRACE_CPACK_PACKAGE_SUFFIX}-Python3")
endif()

set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${OMNITRACE_VERSION}-${OMNITRACE_CPACK_SYSTEM_NAME}${OMNITRACE_CPACK_PACKAGE_SUFFIX}"
    )
if(DEFINED ENV{CPACK_PACKAGE_FILE_NAME})
    set(CPACK_PACKAGE_FILE_NAME $ENV{CPACK_PACKAGE_FILE_NAME})
endif()

set(OMNITRACE_PACKAGE_FILE_NAME
    ${CPACK_PACKAGE_NAME}-${OMNITRACE_VERSION}-${OMNITRACE_CPACK_SYSTEM_NAME}${OMNITRACE_CPACK_PACKAGE_SUFFIX}
    )
omnitrace_add_feature(OMNITRACE_PACKAGE_FILE_NAME "CPack filename")

# -------------------------------------------------------------------------------------- #
#
# Debian package specific variables
#
# -------------------------------------------------------------------------------------- #

set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/ROCm/omnitrace")
set(CPACK_DEBIAN_PACKAGE_RELEASE
    "${OMNITRACE_CPACK_SYSTEM_NAME}${OMNITRACE_CPACK_PACKAGE_SUFFIX}")
string(REGEX REPLACE "([a-zA-Z])-([0-9])" "\\1\\2" CPACK_DEBIAN_PACKAGE_RELEASE
                     "${CPACK_DEBIAN_PACKAGE_RELEASE}")
string(REPLACE "-" "~" CPACK_DEBIAN_PACKAGE_RELEASE "${CPACK_DEBIAN_PACKAGE_RELEASE}")

set(_DEBIAN_PACKAGE_DEPENDS "")
if(DYNINST_USE_OpenMP)
    list(APPEND _DEBIAN_PACKAGE_DEPENDS libgomp1)
endif()
if(OMNITRACE_USE_PAPI AND NOT OMNITRACE_BUILD_PAPI)
    list(APPEND _DEBIAN_PACKAGE_DEPENDS libpapi-dev libpfm4)
endif()
if(NOT OMNITRACE_BUILD_DYNINST)
    if(NOT DYNINST_BUILD_BOOST)
        foreach(_BOOST_COMPONENT atomic system thread date-time filesystem timer)
            list(APPEND _DEBIAN_PACKAGE_DEPENDS
                 "libboost-${_BOOST_COMPONENT}-dev (>= 1.67.0)")
        endforeach()
    endif()
    if(NOT DYNINST_BUILD_TBB)
        list(APPEND _DEBIAN_PACKAGE_DEPENDS "libtbb-dev (>= 2018.6)")
    endif()
    if(NOT DYNINST_BUILD_LIBIBERTY)
        list(APPEND _DEBIAN_PACKAGE_DEPENDS "libiberty-dev (>= 20170913)")
    endif()
endif()
if(ROCmVersion_FOUND)
    set(_ROCPROFILER_SUFFIX " (>= 1.0.0.${ROCmVersion_NUMERIC_VERSION})")
    set(_ROCTRACER_SUFFIX " (>= 1.0.0.${ROCmVersion_NUMERIC_VERSION})")
    set(_ROCM_SMI_SUFFIX
        " (>= ${ROCmVersion_MAJOR_VERSION}.0.0.${ROCmVersion_NUMERIC_VERSION})")
endif()
if(OMNITRACE_USE_ROCM_SMI)
    list(APPEND _DEBIAN_PACKAGE_DEPENDS "rocm-smi-lib${_ROCM_SMI_SUFFIX}")
endif()
if(OMNITRACE_USE_ROCTRACER)
    list(APPEND _DEBIAN_PACKAGE_DEPENDS "roctracer-dev${_ROCTRACER_SUFFIX}")
endif()
if(OMNITRACE_USE_ROCPROFILER)
    list(APPEND _DEBIAN_PACKAGE_DEPENDS "rocprofiler-dev${_ROCPROFILER_SUFFIX}")
endif()
if(OMNITRACE_USE_MPI)
    if("${OMNITRACE_MPI_IMPL}" STREQUAL "openmpi")
        list(APPEND _DEBIAN_PACKAGE_DEPENDS "libopenmpi-dev")
    elseif("${OMNITRACE_MPI_IMPL}" STREQUAL "mpich")
        list(APPEND _DEBIAN_PACKAGE_DEPENDS "libmpich-dev")
    endif()
endif()
string(REPLACE ";" ", " _DEBIAN_PACKAGE_DEPENDS "${_DEBIAN_PACKAGE_DEPENDS}")
set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "${_DEBIAN_PACKAGE_DEPENDS}"
    CACHE STRING "Debian package dependencies" FORCE)

set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

# -------------------------------------------------------------------------------------- #
#
# RPM package specific variables
#
# -------------------------------------------------------------------------------------- #

if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_PACKAGE_RELEASE
    "${OMNITRACE_CPACK_SYSTEM_NAME}${OMNITRACE_CPACK_PACKAGE_SUFFIX}")
string(REGEX REPLACE "([a-zA-Z])-([0-9])" "\\1\\2" CPACK_RPM_PACKAGE_RELEASE
                     "${CPACK_RPM_PACKAGE_RELEASE}")
string(REPLACE "-" "~" CPACK_RPM_PACKAGE_RELEASE "${CPACK_RPM_PACKAGE_RELEASE}")

set(_RPM_PACKAGE_PROVIDES "")

if(OMNITRACE_BUILD_LIBUNWIND)
    list(APPEND _RPM_PACKAGE_PROVIDES "libunwind.so.99()(64bit)")
    list(APPEND _RPM_PACKAGE_PROVIDES "libunwind-x86_64.so.99()(64bit)")
    list(APPEND _RPM_PACKAGE_PROVIDES "libunwind-setjmp.so.0()(64bit)")
    list(APPEND _RPM_PACKAGE_PROVIDES "libunwind-ptrace.so.0()(64bit)")
endif()

string(REPLACE ";" ", " CPACK_RPM_PACKAGE_PROVIDES "${_RPM_PACKAGE_PROVIDES}")
set(CPACK_RPM_PACKAGE_PROVIDES
    "${CPACK_RPM_PACKAGE_PROVIDES}"
    CACHE STRING "RPM package provides" FORCE)

set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
set(CPACK_RPM_PACKAGE_AUTOREQPROV ON)

# -------------------------------------------------------------------------------------- #
#
# Prepare final CPACK parameters
#
# -------------------------------------------------------------------------------------- #

set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
    )

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()

if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

omnitrace_add_feature(CPACK_PACKAGE_NAME "Package name")
omnitrace_add_feature(CPACK_PACKAGE_VERSION "Package version")
omnitrace_add_feature(CPACK_PACKAGING_INSTALL_PREFIX "Package installation prefix")

omnitrace_add_feature(CPACK_DEBIAN_FILE_NAME "Debian file name")
omnitrace_add_feature(CPACK_DEBIAN_PACKAGE_RELEASE "Debian package release version")
omnitrace_add_feature(CPACK_DEBIAN_PACKAGE_DEPENDS "Debian package dependencies")
omnitrace_add_feature(CPACK_DEBIAN_PACKAGE_SHLIBDEPS
                      "Debian package shared library dependencies")

omnitrace_add_feature(CPACK_RPM_FILE_NAME "RPM file name")
omnitrace_add_feature(CPACK_RPM_PACKAGE_RELEASE "RPM package release version")
omnitrace_add_feature(CPACK_RPM_PACKAGE_REQUIRES "RPM package dependencies")
omnitrace_add_feature(CPACK_RPM_PACKAGE_AUTOREQPROV
                      "RPM package auto generate requires and provides")
omnitrace_add_feature(CPACK_RPM_PACKAGE_REQUIRES "RPM package requires")
omnitrace_add_feature(CPACK_RPM_PACKAGE_PROVIDES "RPM package provides")

include(CPack)
