# -------------------------------------------------------------------------------------- #
#
# attach tests
#
# -------------------------------------------------------------------------------------- #

set(_VALID_PTRACE_SCOPE OFF)

if(EXISTS "/proc/sys/kernel/yama/ptrace_scope")
    file(READ "/proc/sys/kernel/yama/ptrace_scope" _PTRACE_SCOPE LIMIT 1)

    if("${_PTRACE_SCOPE}" EQUAL 0)
        set(_VALID_PTRACE_SCOPE ON)
    endif()
else()
    omnitrace_message(
        AUTHOR_WARNING
        "Disabling attach tests. Run 'echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope' to enable attaching to process"
        )
endif()

if(NOT _VALID_PTRACE_SCOPE)
    return()
endif()

if(NOT TARGET parallel-overhead)
    return()
endif()

add_test(
    NAME parallel-overhead-attach
    COMMAND
        ${CMAKE_CURRENT_LIST_DIR}/run-rocprofsys-pid.sh
        $<TARGET_FILE:rocprofsys-instrument> -ME "\.c$" -E fib -e -v 1 --label return args
        file -l -- $<TARGET_FILE:parallel-overhead> 30 8 1000
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set(_parallel_overhead_attach_environ
    "${_attach_environment}" "OMNITRACE_OUTPUT_PATH=rocprofsys-tests-output"
    "OMNITRACE_OUTPUT_PREFIX=parallel-overhead-attach/")

set_tests_properties(
    parallel-overhead-attach
    PROPERTIES ENVIRONMENT
               "${_parallel_overhead_attach_environ}"
               TIMEOUT
               300
               LABELS
               "parallel-overhead;attach"
               PASS_REGULAR_EXPRESSION
               "Outputting.*(perfetto-trace.proto).*Outputting.*(wall_clock.txt)"
               FAIL_REGULAR_EXPRESSION
               "Dyninst was unable to attach to the specified process")
