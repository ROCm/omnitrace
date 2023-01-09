#
set(component_sources
    ${CMAKE_CURRENT_LIST_DIR}/backtrace_causal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/blocking_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/causal_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/progress_point.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unblocking_gotcha.cpp)

set(component_headers
    ${CMAKE_CURRENT_LIST_DIR}/backtrace_causal.hpp
    ${CMAKE_CURRENT_LIST_DIR}/blocking_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/causal_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/progress_point.hpp
    ${CMAKE_CURRENT_LIST_DIR}/unblocking_gotcha.hpp)

target_sources(omnitrace-object-library PRIVATE ${component_sources} ${component_headers})
