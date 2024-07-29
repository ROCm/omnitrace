# include guard
include_guard(DIRECTORY)

# ########################################################################################
#
# External Packages are found here
#
# ########################################################################################

omnitrace_add_interface_library(
    omnitrace-headers "Provides minimal set of include flags to compile with omnitrace")
omnitrace_add_interface_library(omnitrace-threading "Enables multithreading support")
omnitrace_add_interface_library(
    omnitrace-dyninst
    "Provides flags and libraries for Dyninst (dynamic instrumentation)")
omnitrace_add_interface_library(omnitrace-hip "Provides flags and libraries for HIP")
omnitrace_add_interface_library(omnitrace-roctracer
                                "Provides flags and libraries for roctracer")
omnitrace_add_interface_library(omnitrace-rocprofiler
                                "Provides flags and libraries for rocprofiler")
omnitrace_add_interface_library(omnitrace-rocm-smi
                                "Provides flags and libraries for rocm-smi")
omnitrace_add_interface_library(
    omnitrace-rccl "Provides flags for ROCm Communication Collectives Library (RCCL)")
omnitrace_add_interface_library(omnitrace-mpi "Provides MPI or MPI headers")
omnitrace_add_interface_library(omnitrace-bfd "Provides Binary File Descriptor (BFD)")
omnitrace_add_interface_library(omnitrace-ptl "Enables PTL support (tasking)")
omnitrace_add_interface_library(omnitrace-papi "Enable PAPI support")
omnitrace_add_interface_library(omnitrace-ompt "Enable OMPT support")
omnitrace_add_interface_library(omnitrace-python "Enables Python support")
omnitrace_add_interface_library(omnitrace-elfutils "Provides ElfUtils")
omnitrace_add_interface_library(omnitrace-perfetto "Enables Perfetto support")
omnitrace_add_interface_library(omnitrace-timemory "Provides timemory libraries")
omnitrace_add_interface_library(omnitrace-timemory-config
                                "CMake interface library applied to all timemory targets")
omnitrace_add_interface_library(omnitrace-compile-definitions "Compile definitions")

# libraries with relevant compile definitions
set(OMNITRACE_EXTENSION_LIBRARIES
    omnitrace::omnitrace-hip
    omnitrace::omnitrace-roctracer
    omnitrace::omnitrace-rocprofiler
    omnitrace::omnitrace-rocm-smi
    omnitrace::omnitrace-rccl
    omnitrace::omnitrace-bfd
    omnitrace::omnitrace-mpi
    omnitrace::omnitrace-ptl
    omnitrace::omnitrace-ompt
    omnitrace::omnitrace-papi
    omnitrace::omnitrace-perfetto)

target_include_directories(
    omnitrace-headers
    INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/lib>
              $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/lib/core>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/lib>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/lib/omnitrace>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/lib/omnitrace-dl>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/lib/omnitrace-user>)

# include threading because of rooflines
target_link_libraries(omnitrace-headers INTERFACE omnitrace::omnitrace-threading)

# ensure the env overrides the appending /opt/rocm later
string(REPLACE ":" ";" CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}")

set(OMNITRACE_DEFAULT_ROCM_PATH
    /opt/rocm
    CACHE PATH "Default search path for ROCM")
if(EXISTS ${OMNITRACE_DEFAULT_ROCM_PATH})
    get_filename_component(_OMNITRACE_DEFAULT_ROCM_PATH "${OMNITRACE_DEFAULT_ROCM_PATH}"
                           REALPATH)

    if(NOT "${_OMNITRACE_DEFAULT_ROCM_PATH}" STREQUAL "${OMNITRACE_DEFAULT_ROCM_PATH}")
        set(OMNITRACE_DEFAULT_ROCM_PATH
            "${_OMNITRACE_DEFAULT_ROCM_PATH}"
            CACHE PATH "Default search path for ROCM" FORCE)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Threading
#
# ----------------------------------------------------------------------------------------#

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG OFF)

find_library(pthread_LIBRARY NAMES pthread pthreads)
find_package_handle_standard_args(pthread-library REQUIRED_VARS pthread_LIBRARY)

find_library(pthread_LIBRARY NAMES pthread pthreads)
find_package_handle_standard_args(pthread-library REQUIRED_VARS pthread_LIBRARY)

if(pthread_LIBRARY)
    target_link_libraries(omnitrace-threading INTERFACE ${pthread_LIBRARY})
else()
    find_package(Threads ${omnitrace_FIND_QUIETLY} ${omnitrace_FIND_REQUIREMENT})
    if(Threads_FOUND)
        target_link_libraries(omnitrace-threading INTERFACE Threads::Threads)
    endif()
endif()

foreach(_LIB dl rt)
    find_library(${_LIB}_LIBRARY NAMES ${_LIB})
    find_package_handle_standard_args(${_LIB}-library REQUIRED_VARS ${_LIB}_LIBRARY)
    if(${_LIB}_LIBRARY)
        target_link_libraries(omnitrace-threading INTERFACE ${${_LIB}_LIBRARY})
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#
#
# hip version
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_HIP
   OR OMNITRACE_USE_ROCTRACER
   OR OMNITRACE_USE_ROCPROFILER
   OR OMNITRACE_USE_ROCM_SMI)
    find_package(ROCmVersion)

    if(NOT ROCmVersion_FOUND)
        find_package(hip ${omnitrace_FIND_QUIETLY} REQUIRED HINTS
                     ${OMNITRACE_DEFAULT_ROCM_PATH} PATHS ${OMNITRACE_DEFAULT_ROCM_PATH})
        if(SPACK_BUILD)
            find_package(ROCmVersion HINTS ${ROCM_PATH} PATHS ${ROCM_PATH})
        else()
            find_package(ROCmVersion REQUIRED HINTS ${ROCM_PATH} PATHS ${ROCM_PATH})
        endif()
    endif()

    if(NOT ROCmVersion_FOUND)
        rocm_version_compute("${hip_VERSION}" _local)

        foreach(_V ${ROCmVersion_VARIABLES})
            set(_CACHE_VAR ROCmVersion_${_V}_VERSION)
            set(_LOCAL_VAR _local_${_V}_VERSION)
            set(ROCmVersion_${_V}_VERSION
                "${${_LOCAL_VAR}}"
                CACHE STRING "ROCm ${_V} version")
            rocm_version_watch_for_change(${_CACHE_VAR})
        endforeach()
    else()
        list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR})
    endif()

    set(OMNITRACE_ROCM_VERSION ${ROCmVersion_FULL_VERSION})
    set(OMNITRACE_HIP_VERSION_MAJOR ${ROCmVersion_MAJOR_VERSION})
    set(OMNITRACE_HIP_VERSION_MINOR ${ROCmVersion_MINOR_VERSION})
    set(OMNITRACE_HIP_VERSION_PATCH ${ROCmVersion_PATCH_VERSION})
    set(OMNITRACE_HIP_VERSION ${ROCmVersion_TRIPLE_VERSION})

    if(OMNITRACE_HIP_VERSION_MAJOR GREATER_EQUAL 4 AND OMNITRACE_HIP_VERSION_MINOR
                                                       GREATER 3)
        set(roctracer_kfdwrapper_LIBRARY)
    endif()

    if(NOT roctracer_kfdwrapper_LIBRARY)
        set(roctracer_kfdwrapper_LIBRARY)
    endif()

    omnitrace_add_feature(OMNITRACE_ROCM_VERSION "ROCm version used by omnitrace")
else()
    set(OMNITRACE_HIP_VERSION "0.0.0")
    set(OMNITRACE_HIP_VERSION_MAJOR 0)
    set(OMNITRACE_HIP_VERSION_MINOR 0)
    set(OMNITRACE_HIP_VERSION_PATCH 0)
endif()

# ----------------------------------------------------------------------------------------#
#
# HIP
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_HIP)
    find_package(hip ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(omnitrace-hip INTERFACE OMNITRACE_USE_HIP)
    target_link_libraries(omnitrace-hip INTERFACE hip::host)
endif()

# ----------------------------------------------------------------------------------------#
#
# roctracer
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_ROCTRACER)
    find_package(roctracer ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(omnitrace-roctracer
                                         INTERFACE OMNITRACE_USE_ROCTRACER)
    target_link_libraries(omnitrace-roctracer INTERFACE roctracer::roctracer
                                                        omnitrace::omnitrace-hip)
endif()

# ----------------------------------------------------------------------------------------#
#
# rocprofiler
#
# ----------------------------------------------------------------------------------------#
if(OMNITRACE_USE_ROCPROFILER)
    find_package(rocprofiler ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(omnitrace-rocprofiler
                                         INTERFACE OMNITRACE_USE_ROCPROFILER)
    target_link_libraries(omnitrace-rocprofiler INTERFACE rocprofiler::rocprofiler)
endif()

# ----------------------------------------------------------------------------------------#
#
# rocm-smi
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_ROCM_SMI)
    find_package(rocm-smi ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(omnitrace-rocm-smi
                                         INTERFACE OMNITRACE_USE_ROCM_SMI)
    target_link_libraries(omnitrace-rocm-smi INTERFACE rocm-smi::rocm-smi)
endif()

# ----------------------------------------------------------------------------------------#
#
# RCCL
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_RCCL)
    find_package(RCCL-Headers ${omnitrace_FIND_QUIETLY} REQUIRED)
    target_link_libraries(omnitrace-rccl INTERFACE roc::rccl-headers)
    omnitrace_target_compile_definitions(omnitrace-rccl INTERFACE OMNITRACE_USE_RCCL)
endif()

# ----------------------------------------------------------------------------------------#
#
# MPI
#
# ----------------------------------------------------------------------------------------#

# suppress warning during CI that MPI_HEADERS_ALLOW_MPICH was unused
set(_OMNITRACE_MPI_HEADERS_ALLOW_MPICH ${MPI_HEADERS_ALLOW_MPICH})

if(OMNITRACE_USE_MPI)
    find_package(MPI ${omnitrace_FIND_QUIETLY} REQUIRED)
    target_link_libraries(omnitrace-mpi INTERFACE MPI::MPI_C MPI::MPI_CXX)
    omnitrace_target_compile_definitions(omnitrace-mpi INTERFACE TIMEMORY_USE_MPI=1
                                                                 OMNITRACE_USE_MPI)
elseif(OMNITRACE_USE_MPI_HEADERS)
    find_package(MPI-Headers ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(
        omnitrace-mpi INTERFACE TIMEMORY_USE_MPI_HEADERS=1 OMNITRACE_USE_MPI_HEADERS)
    target_link_libraries(omnitrace-mpi INTERFACE MPI::MPI_HEADERS)
endif()

# ----------------------------------------------------------------------------------------#
#
# OMPT
#
# ----------------------------------------------------------------------------------------#

omnitrace_target_compile_definitions(
    omnitrace-ompt INTERFACE OMNITRACE_USE_OMPT=$<BOOL:${OMNITRACE_USE_OMPT}>)

# ----------------------------------------------------------------------------------------#
#
# ElfUtils
#
# ----------------------------------------------------------------------------------------#

include(ElfUtils)

target_include_directories(omnitrace-elfutils SYSTEM INTERFACE ${ElfUtils_INCLUDE_DIRS})
target_compile_definitions(omnitrace-elfutils INTERFACE ${ElfUtils_DEFINITIONS})
target_link_directories(omnitrace-elfutils INTERFACE ${ElfUtils_LIBRARY_DIRS})
target_link_libraries(omnitrace-elfutils INTERFACE ${ElfUtils_LIBRARIES})

# ----------------------------------------------------------------------------------------#
#
# Dyninst
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_BUILD_DYNINST)
    omnitrace_checkout_git_submodule(
        RELATIVE_PATH external/dyninst
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/dyninst.git
        REPO_BRANCH omnitrace)

    set(DYNINST_OPTION_PREFIX ON)
    set(DYNINST_BUILD_DOCS OFF)
    set(DYNINST_BUILD_RTLIB OFF)
    set(DYNINST_QUIET_CONFIG
        ON
        CACHE BOOL "Suppress dyninst cmake messages")
    set(DYNINST_BUILD_PARSE_THAT
        OFF
        CACHE BOOL "Build dyninst parseThat executable")
    set(DYNINST_BUILD_SHARED_LIBS
        ON
        CACHE BOOL "Build shared dyninst libraries")
    set(DYNINST_BUILD_STATIC_LIBS
        OFF
        CACHE BOOL "Build static dyninst libraries")
    set(DYNINST_ENABLE_LTO
        OFF
        CACHE BOOL "Enable LTO for dyninst libraries")

    if(NOT DEFINED CMAKE_INSTALL_RPATH)
        set(CMAKE_INSTALL_RPATH "")
    endif()

    if(NOT DEFINED CMAKE_BUILD_RPATH)
        set(CMAKE_BUILD_RPATH "")
    endif()

    omnitrace_save_variables(
        PIC VARIABLES CMAKE_POSITION_INDEPENDENT_CODE CMAKE_INSTALL_RPATH
                      CMAKE_BUILD_RPATH CMAKE_INSTALL_RPATH_USE_LINK_PATH)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
    set(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/omnitrace")
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/omnitrace")
    set(DYNINST_TPL_INSTALL_PREFIX
        "omnitrace"
        CACHE PATH "Third-party library install-tree install prefix" FORCE)
    set(DYNINST_TPL_INSTALL_LIB_DIR
        "omnitrace"
        CACHE PATH "Third-party library install-tree install library prefix" FORCE)
    add_subdirectory(external/dyninst EXCLUDE_FROM_ALL)
    omnitrace_restore_variables(
        PIC VARIABLES CMAKE_POSITION_INDEPENDENT_CODE CMAKE_INSTALL_RPATH
                      CMAKE_BUILD_RPATH CMAKE_INSTALL_RPATH_USE_LINK_PATH)

    add_library(Dyninst::Dyninst INTERFACE IMPORTED)
    foreach(_LIB common dyninstAPI parseAPI instructionAPI symtabAPI stackwalk)
        target_link_libraries(Dyninst::Dyninst INTERFACE Dyninst::${_LIB})
    endforeach()

    foreach(
        _LIB
        common
        dynDwarf
        dynElf
        dyninstAPI
        instructionAPI
        parseAPI
        patchAPI
        pcontrol
        stackwalk
        symtabAPI)
        if(TARGET ${_LIB})
            install(
                TARGETS ${_LIB}
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace
                COMPONENT dyninst
                PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/omnitrace/dyninst)
        endif()
    endforeach()

    # for packaging
    install(
        DIRECTORY ${DYNINST_TPL_STAGING_PREFIX}/lib/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace
        COMPONENT dyninst
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*")

    target_link_libraries(omnitrace-dyninst INTERFACE Dyninst::Dyninst)

else()
    find_package(Dyninst ${omnitrace_FIND_QUIETLY} REQUIRED
                 COMPONENTS dyninstAPI parseAPI instructionAPI symtabAPI)

    if(TARGET Dyninst::Dyninst) # updated Dyninst CMake system was found
        target_link_libraries(omnitrace-dyninst INTERFACE Dyninst::Dyninst)
    else() # updated Dyninst CMake system was not found
        set(_BOOST_COMPONENTS atomic system thread date_time)
        set(omnitrace_BOOST_COMPONENTS
            "${_BOOST_COMPONENTS}"
            CACHE STRING "Boost components used by Dyninst in omnitrace")
        set(Boost_NO_BOOST_CMAKE ON)
        find_package(Boost QUIET REQUIRED COMPONENTS ${omnitrace_BOOST_COMPONENTS})

        # some installs of dyninst don't set this properly
        if(EXISTS "${DYNINST_INCLUDE_DIR}" AND NOT DYNINST_HEADER_DIR)
            get_filename_component(DYNINST_HEADER_DIR "${DYNINST_INCLUDE_DIR}" REALPATH
                                   CACHE)
        else()
            find_path(
                DYNINST_HEADER_DIR
                NAMES BPatch.h dyninstAPI_RT.h
                HINTS ${Dyninst_ROOT_DIR} ${Dyninst_DIR} ${Dyninst_DIR}/../../..
                PATHS ${Dyninst_ROOT_DIR} ${Dyninst_DIR} ${Dyninst_DIR}/../../..
                PATH_SUFFIXES include)
        endif()

        # try to find TBB
        find_package(TBB QUIET)

        # if fail try to use the Dyninst installed FindTBB.cmake
        if(NOT TBB_FOUND)
            list(APPEND CMAKE_MODULE_PATH ${Dyninst_DIR}/Modules)
            find_package(TBB QUIET)
        endif()

        if(NOT TBB_FOUND)
            find_path(
                TBB_INCLUDE_DIR
                NAMES tbb/tbb.h
                PATH_SUFFIXES include)
        endif()

        target_link_libraries(omnitrace-dyninst INTERFACE ${DYNINST_LIBRARIES}
                                                          ${Boost_LIBRARIES})
        foreach(
            _TARG
            dyninst
            dyninstAPI
            instructionAPI
            symtabAPI
            parseAPI
            headers
            atomic
            system
            thread
            date_time
            TBB)
            if(TARGET Dyninst::${_TARG})
                target_link_libraries(omnitrace-dyninst INTERFACE Dyninst::${_TARG})
            elseif(TARGET Boost::${_TARG})
                target_link_libraries(omnitrace-dyninst INTERFACE Boost::${_TARG})
            elseif(TARGET ${_TARG})
                target_link_libraries(omnitrace-dyninst INTERFACE ${_TARG})
            endif()
        endforeach()
        target_include_directories(
            omnitrace-dyninst SYSTEM INTERFACE ${TBB_INCLUDE_DIR} ${Boost_INCLUDE_DIRS}
                                               ${DYNINST_HEADER_DIR})
        omnitrace_target_compile_definitions(omnitrace-dyninst
                                             INTERFACE OMNITRACE_USE_DYNINST)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Modify CMAKE_C_FLAGS and CMAKE_CXX_FLAGS with -static-libgcc and -static-libstdc++
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_BUILD_STATIC_LIBGCC)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        omnitrace_save_variables(STATIC_LIBGCC_C VARIABLES CMAKE_C_FLAGS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static-libgcc")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        omnitrace_save_variables(STATIC_LIBGCC_CXX VARIABLES CMAKE_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc")
    else()
        set(OMNITRACE_BUILD_STATIC_LIBGCC OFF)
    endif()
endif()

if(OMNITRACE_BUILD_STATIC_LIBSTDCXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        omnitrace_save_variables(STATIC_LIBSTDCXX_CXX VARIABLES CMAKE_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
    else()
        set(OMNITRACE_BUILD_STATIC_LIBSTDCXX OFF)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Perfetto
#
# ----------------------------------------------------------------------------------------#

set(perfetto_DIR ${PROJECT_SOURCE_DIR}/external/perfetto)
omnitrace_checkout_git_submodule(
    RELATIVE_PATH external/perfetto
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://android.googlesource.com/platform/external/perfetto
    REPO_BRANCH v28.0
    TEST_FILE sdk/perfetto.cc)

include(Perfetto)

# ----------------------------------------------------------------------------------------#
#
# ELFIO
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_BUILD_DEVICETRACE)
    omnitrace_checkout_git_submodule(
        RELATIVE_PATH external/elfio
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/ELFIO.git
        REPO_BRANCH set-offset-support)

    add_subdirectory(external/elfio)
endif()

# ----------------------------------------------------------------------------------------#
#
# papi submodule
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_PAPI AND OMNITRACE_BUILD_PAPI)
    include(PAPI)
endif()

# ----------------------------------------------------------------------------------------#
#
# timemory submodule
#
# ----------------------------------------------------------------------------------------#

target_compile_definitions(
    omnitrace-timemory-config
    INTERFACE TIMEMORY_PAPI_ARRAY_SIZE=12 TIMEMORY_USE_ROOFLINE=0 TIMEMORY_USE_ERT=0
              TIMEMORY_USE_CONTAINERS=0 TIMEMORY_USE_ERT_EXTERN=0
              TIMEMORY_USE_CONTAINERS_EXTERN=0)

if(OMNITRACE_BUILD_STACK_PROTECTOR)
    add_target_flag_if_avail(omnitrace-timemory-config "-fstack-protector-strong"
                             "-Wstack-protector")
endif()

if(OMNITRACE_BUILD_DEBUG)
    add_target_flag_if_avail(omnitrace-timemory-config "-fno-omit-frame-pointer" "-g3")
endif()

set(TIMEMORY_EXTERNAL_INTERFACE_LIBRARY
    omnitrace-timemory-config
    CACHE STRING "timemory configuration interface library")
set(TIMEMORY_INSTALL_HEADERS
    OFF
    CACHE BOOL "Disable timemory header install")
set(TIMEMORY_INSTALL_CONFIG
    OFF
    CACHE BOOL "Disable timemory cmake configuration install")
set(TIMEMORY_INSTALL_LIBRARIES
    OFF
    CACHE BOOL "Disable timemory installation of libraries not needed at runtime")
set(TIMEMORY_INSTALL_ALL
    OFF
    CACHE BOOL "Disable install target depending on all target")
set(TIMEMORY_BUILD_C
    OFF
    CACHE BOOL "Disable timemory C library")
set(TIMEMORY_BUILD_FORTRAN
    OFF
    CACHE BOOL "Disable timemory Fortran library")
set(TIMEMORY_BUILD_TOOLS
    OFF
    CACHE BOOL "Ensure timem executable is built")
set(TIMEMORY_BUILD_EXCLUDE_FROM_ALL
    ON
    CACHE BOOL "Set timemory to only build dependencies")
set(TIMEMORY_BUILD_HIDDEN_VISIBILITY
    ON
    CACHE BOOL "Build timemory with hidden visibility")
set(TIMEMORY_QUIET_CONFIG
    ON
    CACHE BOOL "Make timemory configuration quieter")

# timemory feature settings
set(TIMEMORY_USE_MPI
    ${OMNITRACE_USE_MPI}
    CACHE BOOL "Enable MPI support in timemory" FORCE)
set(TIMEMORY_USE_GOTCHA
    ON
    CACHE BOOL "Enable GOTCHA support in timemory")
set(TIMEMORY_USE_PERFETTO
    OFF
    CACHE BOOL "Disable perfetto support in timemory")
set(TIMEMORY_USE_OMPT
    ${OMNITRACE_USE_OMPT}
    CACHE BOOL "Enable OMPT support in timemory" FORCE)
set(TIMEMORY_USE_PAPI
    ${OMNITRACE_USE_PAPI}
    CACHE BOOL "Enable PAPI support in timemory" FORCE)
set(TIMEMORY_USE_BFD
    ${OMNITRACE_USE_BFD}
    CACHE BOOL "Enable BFD support in timemory" FORCE)
set(TIMEMORY_USE_LIBUNWIND
    ON
    CACHE BOOL "Enable libunwind support in timemory")
set(TIMEMORY_USE_VISIBILITY
    OFF
    CACHE BOOL "Enable/disable using visibility decorations")
set(TIMEMORY_USE_SANITIZER
    ${OMNITRACE_USE_SANITIZER}
    CACHE BOOL "Build with -fsanitze=\${OMNITRACE_SANITIZER_TYPE}" FORCE)
set(TIMEMORY_SANITIZER_TYPE
    ${OMNITRACE_SANITIZER_TYPE}
    CACHE STRING "Sanitizer type, e.g. leak, thread, address, memory, etc." FORCE)

if(DEFINED TIMEMORY_BUILD_GOTCHA AND NOT TIMEMORY_BUILD_GOTCHA)
    omnitrace_message(
        FATAL_ERROR
        "Using an external gotcha is not allowed due to known bug that has not been accepted upstream"
        )
endif()

# timemory feature build settings
set(TIMEMORY_BUILD_GOTCHA
    ON
    CACHE BOOL "Enable building GOTCHA library from submodule" FORCE)
set(TIMEMORY_BUILD_LIBUNWIND
    ${OMNITRACE_BUILD_LIBUNWIND}
    CACHE BOOL "Enable building libunwind library from submodule" FORCE)
set(TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS
    ${OMNITRACE_BUILD_EXTRA_OPTIMIZATIONS}
    CACHE BOOL "Enable building GOTCHA library from submodule" FORCE)
set(TIMEMORY_BUILD_ERT
    OFF
    CACHE BOOL "Disable building ERT support" FORCE)
set(TIMEMORY_BUILD_CONTAINERS
    OFF
    CACHE BOOL "Disable building container extern templates (unused)" FORCE)

# timemory build settings
set(TIMEMORY_TLS_MODEL
    "global-dynamic"
    CACHE STRING "Thread-local static model" FORCE)
set(TIMEMORY_MAX_THREADS
    "${OMNITRACE_MAX_THREADS}"
    CACHE STRING "Max statically-allocated threads" FORCE)
set(TIMEMORY_SETTINGS_PREFIX
    "OMNITRACE_"
    CACHE STRING "Prefix used for settings and environment variables")
set(TIMEMORY_PROJECT_NAME
    "omnitrace"
    CACHE STRING "Name for configuration")
set(TIMEMORY_CXX_LIBRARY_EXCLUDE
    "kokkosp.cpp;pthread.cpp;timemory_c.cpp;trace.cpp;weak.cpp;library.cpp"
    CACHE STRING "Timemory C++ library implementation files to exclude from compiling")

mark_as_advanced(TIMEMORY_SETTINGS_PREFIX)
mark_as_advanced(TIMEMORY_PROJECT_NAME)

omnitrace_checkout_git_submodule(
    RELATIVE_PATH external/timemory
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://github.com/NERSC/timemory.git
    REPO_BRANCH omnitrace)

omnitrace_save_variables(
    BUILD_CONFIG VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS
                           CMAKE_POSITION_INDEPENDENT_CODE CMAKE_PREFIX_PATH)

# ensure timemory builds PIC static libs so that we don't have to install timemory shared
# lib
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(TIMEMORY_CTP_OPTIONS GLOBAL)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # results in undefined symbols to component::base<T>::load()
    set(TIMEMORY_BUILD_HIDDEN_VISIBILITY
        OFF
        CACHE BOOL "" FORCE)
endif()

add_subdirectory(external/timemory EXCLUDE_FROM_ALL)

install(
    TARGETS gotcha
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace
    COMPONENT gotcha)
if(OMNITRACE_BUILD_LIBUNWIND)
    install(
        DIRECTORY ${PROJECT_BINARY_DIR}/external/timemory/external/libunwind/install/lib/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace
        COMPONENT libunwind
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
endif()

omnitrace_restore_variables(
    BUILD_CONFIG VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS
                           CMAKE_POSITION_INDEPENDENT_CODE CMAKE_PREFIX_PATH)

if(TARGET omnitrace-papi-build)
    foreach(_TARGET PAPI::papi timemory-core timemory-common timemory-papi-component
                    timemory-cxx)
        if(TARGET "${_TARGET}")
            add_dependencies(${_TARGET} omnitrace-papi-build)
        endif()
        foreach(_LINK shared static)
            if(TARGET "${_TARGET}-${_LINK}")
                add_dependencies(${_TARGET}-${_LINK} omnitrace-papi-build)
            endif()
        endforeach()
    endforeach()
endif()

target_link_libraries(
    omnitrace-timemory
    INTERFACE $<BUILD_INTERFACE:timemory::timemory-headers>
              $<BUILD_INTERFACE:timemory::timemory-gotcha>
              $<BUILD_INTERFACE:timemory::timemory-cxx-static>)

target_link_libraries(omnitrace-bfd INTERFACE $<BUILD_INTERFACE:timemory::timemory-bfd>)

if(OMNITRACE_USE_BFD)
    omnitrace_target_compile_definitions(omnitrace-bfd INTERFACE OMNITRACE_USE_BFD)
endif()

# ----------------------------------------------------------------------------------------#
#
# PTL (Parallel Tasking Library) submodule
#
# ----------------------------------------------------------------------------------------#

# timemory might provide PTL::ptl-shared
if(NOT TARGET PTL::ptl-shared)
    omnitrace_checkout_git_submodule(
        RELATIVE_PATH external/PTL
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/PTL.git
        REPO_BRANCH omnitrace)

    set(PTL_BUILD_EXAMPLES OFF)
    set(PTL_USE_TBB OFF)
    set(PTL_USE_GPU OFF)
    set(PTL_DEVELOPER_INSTALL OFF)

    if(NOT DEFINED BUILD_OBJECT_LIBS)
        set(BUILD_OBJECT_LIBS OFF)
    endif()
    omnitrace_save_variables(
        BUILD_CONFIG
        VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS BUILD_OBJECT_LIBS
                  CMAKE_POSITION_INDEPENDENT_CODE CMAKE_CXX_VISIBILITY_PRESET
                  CMAKE_VISIBILITY_INLINES_HIDDEN)

    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_STATIC_LIBS OFF)
    set(BUILD_OBJECT_LIBS ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

    add_subdirectory(external/PTL EXCLUDE_FROM_ALL)

    omnitrace_restore_variables(
        BUILD_CONFIG
        VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS BUILD_OBJECT_LIBS
                  CMAKE_POSITION_INDEPENDENT_CODE CMAKE_CXX_VISIBILITY_PRESET
                  CMAKE_VISIBILITY_INLINES_HIDDEN)
endif()

target_sources(omnitrace-ptl
               INTERFACE $<BUILD_INTERFACE:$<TARGET_OBJECTS:PTL::ptl-object>>)
target_include_directories(
    omnitrace-ptl INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/PTL/source>
                            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/PTL/source>)

# ----------------------------------------------------------------------------------------#
#
# Restore the CMAKE_C_FLAGS and CMAKE_CXX_FLAGS in the inverse order
#
# ----------------------------------------------------------------------------------------#

# override compiler macros
include(Compilers)

if(OMNITRACE_BUILD_STATIC_LIBSTDCXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        omnitrace_restore_variables(STATIC_LIBSTDCXX_CXX VARIABLES CMAKE_CXX_FLAGS)
    endif()
endif()

if(OMNITRACE_BUILD_STATIC_LIBGCC)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        omnitrace_restore_variables(STATIC_LIBGCC_C VARIABLES CMAKE_C_FLAGS)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        omnitrace_restore_variables(STATIC_LIBGCC_CXX VARIABLES CMAKE_CXX_FLAGS)
    endif()
endif()

omnitrace_add_feature(CMAKE_C_FLAGS "C compiler flags")
omnitrace_add_feature(CMAKE_CXX_FLAGS "C++ compiler flags")

# ----------------------------------------------------------------------------------------#
#
# Python
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_PYTHON)
    if(OMNITRACE_USE_PYTHON AND NOT OMNITRACE_BUILD_PYTHON)
        find_package(pybind11 REQUIRED)
    endif()

    include(ConfigPython)
    include(PyBind11Tools)

    omnitrace_watch_for_change(OMNITRACE_PYTHON_ROOT_DIRS _PYTHON_DIRS_CHANGED)

    if(_PYTHON_DIRS_CHANGED)
        unset(OMNITRACE_PYTHON_VERSION CACHE)
        unset(OMNITRACE_PYTHON_VERSIONS CACHE)
        unset(OMNITRACE_INSTALL_PYTHONDIR CACHE)
    else()
        foreach(_VAR PREFIX ENVS)
            omnitrace_watch_for_change(OMNITRACE_PYTHON_${_VAR} _CHANGED)

            if(_CHANGED)
                unset(OMNITRACE_PYTHON_ROOT_DIRS CACHE)
                unset(OMNITRACE_PYTHON_VERSIONS CACHE)
                unset(OMNITRACE_INSTALL_PYTHONDIR CACHE)
                break()
            endif()
        endforeach()
    endif()

    if(OMNITRACE_PYTHON_PREFIX AND OMNITRACE_PYTHON_ENVS)
        omnitrace_directory(
            FAIL
            PREFIX ${OMNITRACE_PYTHON_PREFIX}
            PATHS ${OMNITRACE_PYTHON_ENVS}
            OUTPUT_VARIABLE _PYTHON_ROOT_DIRS)
        set(OMNITRACE_PYTHON_ROOT_DIRS
            "${_PYTHON_ROOT_DIRS}"
            CACHE INTERNAL "Root directories for python")
    endif()

    if(NOT OMNITRACE_PYTHON_VERSIONS AND OMNITRACE_PYTHON_VERSION)
        set(OMNITRACE_PYTHON_VERSIONS "${OMNITRACE_PYTHON_VERSION}")

        if(NOT OMNITRACE_PYTHON_ROOT_DIRS)
            omnitrace_find_python(_PY VERSION ${OMNITRACE_PYTHON_VERSION})
            set(OMNITRACE_PYTHON_ROOT_DIRS
                "${_PY_ROOT_DIR}"
                CACHE INTERNAL "" FORCE)
        endif()

        unset(OMNITRACE_PYTHON_VERSION CACHE)
        unset(OMNITRACE_INSTALL_PYTHONDIR CACHE)
    elseif(
        NOT OMNITRACE_PYTHON_VERSIONS
        AND NOT OMNITRACE_PYTHON_VERSION
        AND OMNITRACE_PYTHON_ROOT_DIRS)
        set(_PY_VERSIONS)

        foreach(_DIR ${OMNITRACE_PYTHON_ROOT_DIRS})
            omnitrace_find_python(_PY ROOT_DIR ${_DIR})

            if(NOT _PY_FOUND)
                continue()
            endif()

            if(NOT "${_PY_VERSION}" IN_LIST _PY_VERSIONS)
                list(APPEND _PY_VERSIONS "${_PY_VERSION}")
            endif()
        endforeach()

        set(OMNITRACE_PYTHON_VERSIONS
            "${_PY_VERSIONS}"
            CACHE INTERNAL "" FORCE)
    elseif(
        NOT OMNITRACE_PYTHON_VERSIONS
        AND NOT OMNITRACE_PYTHON_VERSION
        AND NOT OMNITRACE_PYTHON_ROOT_DIRS)
        omnitrace_find_python(_PY REQUIRED)
        set(OMNITRACE_PYTHON_ROOT_DIRS
            "${_PY_ROOT_DIR}"
            CACHE INTERNAL "" FORCE)
        set(OMNITRACE_PYTHON_VERSIONS
            "${_PY_VERSION}"
            CACHE INTERNAL "" FORCE)
    endif()

    omnitrace_watch_for_change(OMNITRACE_PYTHON_ROOT_DIRS)
    omnitrace_watch_for_change(OMNITRACE_PYTHON_VERSIONS)

    omnitrace_check_python_dirs_and_versions(FAIL)

    list(LENGTH OMNITRACE_PYTHON_VERSIONS _NUM_PYTHON_VERSIONS)

    if(_NUM_PYTHON_VERSIONS GREATER 1)
        set(OMNITRACE_INSTALL_PYTHONDIR
            "${CMAKE_INSTALL_LIBDIR}/python/site-packages"
            CACHE STRING "Installation prefix for python")
    else()
        set(OMNITRACE_INSTALL_PYTHONDIR
            "${CMAKE_INSTALL_LIBDIR}/python${OMNITRACE_PYTHON_VERSIONS}/site-packages"
            CACHE STRING "Installation prefix for python")
    endif()
else()
    set(OMNITRACE_INSTALL_PYTHONDIR
        "${CMAKE_INSTALL_LIBDIR}/python/site-packages"
        CACHE STRING "Installation prefix for python")
endif()

omnitrace_watch_for_change(OMNITRACE_INSTALL_PYTHONDIR)
set(CMAKE_INSTALL_PYTHONDIR ${OMNITRACE_INSTALL_PYTHONDIR})

# ----------------------------------------------------------------------------------------#
#
# Compile definitions
#
# ----------------------------------------------------------------------------------------#

if("${CMAKE_BUILD_TYPE}" MATCHES "Release" AND NOT OMNITRACE_BUILD_DEBUG)
    add_target_flag_if_avail(omnitrace-compile-options "-g1")
endif()

target_compile_definitions(omnitrace-compile-definitions
                           INTERFACE OMNITRACE_MAX_THREADS=${OMNITRACE_MAX_THREADS})

foreach(_LIB ${OMNITRACE_EXTENSION_LIBRARIES})
    get_target_property(_COMPILE_DEFS ${_LIB} INTERFACE_COMPILE_DEFINITIONS)
    if(_COMPILE_DEFS)
        foreach(_DEF ${_COMPILE_DEFS})
            if("${_DEF}" MATCHES "OMNITRACE_")
                target_compile_definitions(omnitrace-compile-definitions
                                           INTERFACE ${_DEF})
            endif()
        endforeach()
    endif()
endforeach()
