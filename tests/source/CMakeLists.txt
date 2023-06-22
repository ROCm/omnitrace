set(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)

add_library(tests-compile-options INTERFACE)
target_compile_options(tests-compile-options INTERFACE -g)

add_executable(thread-limit thread-limit.cpp)
target_compile_definitions(thread-limit PRIVATE MAX_THREADS=${OMNITRACE_MAX_THREADS})
target_link_libraries(thread-limit PRIVATE Threads::Threads tests-compile-options)

set(_thread_limit_environment
    "${_base_environment}" "OMNITRACE_USE_PERFETTO=ON" "OMNITRACE_USE_TIMEMORY=ON"
    "OMNITRACE_COUT_OUTPUT=ON" "OMNITRACE_USE_SAMPLING=ON" "OMNITRACE_SAMPLING_FREQ=250"
    "OMNITRACE_VERBOSE=2" "OMNITRACE_TIMEMORY_COMPONENTS=wall_clock,peak_rss,page_rss")

math(EXPR EXCESS_THREADS "${OMNITRACE_MAX_THREADS} + 24")

omnitrace_add_test(
    SKIP_BASELINE
    NAME thread-limit
    TARGET thread-limit
    LABELS "max-threads"
    REWRITE_ARGS -e -v 2 --label return args
    RUNTIME_ARGS -e -v 1 --label return args
    RUN_ARGS 35 2 ${EXCESS_THREADS}
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    ENVIRONMENT "${_thread_limit_environment}")
