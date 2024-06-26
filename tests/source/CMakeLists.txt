set(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)

add_library(tests-compile-options INTERFACE)
target_compile_options(tests-compile-options INTERFACE -g)

add_executable(thread-limit thread-limit.cpp)
target_compile_definitions(thread-limit PRIVATE MAX_THREADS=${OMNITRACE_MAX_THREADS})
target_link_libraries(thread-limit PRIVATE Threads::Threads tests-compile-options)

set(_thread_limit_environment
    "${_base_environment}" "OMNITRACE_TRACE=ON" "OMNITRACE_PROFILE=ON"
    "OMNITRACE_COUT_OUTPUT=ON" "OMNITRACE_USE_SAMPLING=ON" "OMNITRACE_SAMPLING_FREQ=250"
    "OMNITRACE_VERBOSE=2" "OMNITRACE_TIMEMORY_COMPONENTS=wall_clock,peak_rss,page_rss")

math(EXPR THREAD_LIMIT_TEST_VALUE "${OMNITRACE_MAX_THREADS} + 24")
math(EXPR THREAD_LIMIT_TEST_VALUE_PLUS_ONE "${THREAD_LIMIT_TEST_VALUE} + 1")

set(_thread_limit_pass_regex "\\|${THREAD_LIMIT_TEST_VALUE}>>>")
set(_thread_limit_fail_regex
    "\\|${THREAD_LIMIT_TEST_VALUE_PLUS_ONE}>>>|OMNITRACE_ABORT_FAIL_REGEX")

omnitrace_add_test(
    SKIP_BASELINE
    NAME thread-limit
    TARGET thread-limit
    LABELS "max-threads"
    REWRITE_ARGS -e -v 2 -i 1024 --label return args
    RUNTIME_ARGS -e -v 1 -i 1024 --label return args
    RUN_ARGS 35 2 ${THREAD_LIMIT_TEST_VALUE}
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    RUNTIME_PASS_REGEX "${_thread_limit_pass_regex}"
    SAMPLING_PASS_REGEX "${_thread_limit_pass_regex}"
    REWRITE_RUN_PASS_REGEX "${_thread_limit_pass_regex}"
    RUNTIME_FAIL_REGEX "${_thread_limit_fail_regex}"
    SAMPLING_FAIL_REGEX "${_thread_limit_fail_regex}"
    REWRITE_RUN_FAIL_REGEX "${_thread_limit_fail_regex}"
    ENVIRONMENT "${_thread_limit_environment}")
