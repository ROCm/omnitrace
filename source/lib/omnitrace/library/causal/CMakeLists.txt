#
set(casual_sources
    ${CMAKE_CURRENT_LIST_DIR}/blocking_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/delay.cpp
    ${CMAKE_CURRENT_LIST_DIR}/experiment.cpp
    ${CMAKE_CURRENT_LIST_DIR}/progress_point.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unblocking_gotcha.cpp)

set(casual_headers
    ${CMAKE_CURRENT_LIST_DIR}/blocking_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/delay.hpp
    ${CMAKE_CURRENT_LIST_DIR}/experiment.hpp
    ${CMAKE_CURRENT_LIST_DIR}/progress_point.hpp
    ${CMAKE_CURRENT_LIST_DIR}/unblocking_gotcha.hpp)

target_sources(omnitrace-object-library PRIVATE ${casual_sources} ${casual_headers})
