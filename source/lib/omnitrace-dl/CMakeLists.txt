# ------------------------------------------------------------------------------#
#
# rocprofsys dl library
#
# ------------------------------------------------------------------------------#

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_SKIP_RPATH OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_CXX_VISIBILITY_PRESET "internal")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

add_library(rocprofsys-dl-library SHARED)
add_library(rocprofsys::rocprofsys-dl-library ALIAS rocprofsys-dl-library)

target_sources(
    rocprofsys-dl-library
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dl.cpp ${CMAKE_CURRENT_SOURCE_DIR}/main.c
            ${CMAKE_CURRENT_SOURCE_DIR}/dl/dl.hpp)
target_include_directories(
    rocprofsys-dl-library
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../omnitrace-user>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../omnitrace>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(
    rocprofsys-dl-library
    PUBLIC $<BUILD_INTERFACE:${dl_LIBRARY}> $<BUILD_INTERFACE:rocprofsys::common-library>
           $<BUILD_INTERFACE:rocprofsys::rocprofsys-compile-definitions>
           $<BUILD_INTERFACE:Threads::Threads>)

add_target_cxx_flag_if_avail(rocprofsys-dl-library "-ftls-model=global-dynamic")
add_target_cxx_flag_if_avail(rocprofsys-dl-library "-g3")

set_target_properties(
    rocprofsys-dl-library
    PROPERTIES OUTPUT_NAME rocprof-sys-dl
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN")

omnitrace_strip_target(rocprofsys-dl-library)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/dl/dl.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PACKAGE_NAME}/dl)

install(TARGETS rocprofsys-dl-library DESTINATION ${CMAKE_INSTALL_LIBDIR})
