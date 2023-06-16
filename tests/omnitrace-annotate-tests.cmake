# -------------------------------------------------------------------------------------- #
#
# papi tests
#
# -------------------------------------------------------------------------------------- #

if(omnitrace_perf_event_paranoid LESS_EQUAL 3
   OR omnitrace_cap_sys_admin EQUAL 0
   OR omnitrace_cap_perfmon EQUAL 0)
    set(_annotate_environment
        "${_base_environment}"
        "OMNITRACE_TIMEMORY_COMPONENTS=thread_cpu_clock papi_array"
        "OMNITRACE_PAPI_EVENTS=perf::PERF_COUNT_SW_CPU_CLOCK"
        "OMNITRACE_USE_SAMPLING=OFF")

    omnitrace_add_test(
        SKIP_BASELINE SKIP_RUNTIME
        NAME annotate
        TARGET parallel-overhead
        RUN_ARGS 30 2 200
        REWRITE_ARGS -e -v 2
        ENVIRONMENT "${_annotate_environment}"
        LABELS "annotate;papi")

    omnitrace_add_validation_test(
        NAME annotate-binary-rewrite
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "annotate;papi"
        ARGS --key-names perf::PERF_COUNT_SW_CPU_CLOCK thread_cpu_clock --key-counts 8 8)

    omnitrace_add_validation_test(
        NAME annotate-sampling
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "papi"
        ARGS --key-names thread_cpu_clock --key-counts 6)
else()
    set(_annotate_environment
        "${_base_environment}" "OMNITRACE_TIMEMORY_COMPONENTS=thread_cpu_clock"
        "OMNITRACE_USE_SAMPLING=OFF")

    omnitrace_add_test(
        SKIP_BASELINE SKIP_RUNTIME
        NAME papi
        TARGET parallel-overhead
        RUN_ARGS 30 2 200
        REWRITE_ARGS -e -v 2
        ENVIRONMENT "${_annotate_environment}"
        LABELS "annotate")

    omnitrace_add_validation_test(
        NAME annotate-binary-rewrite
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "annotate"
        ARGS --key-names thread_cpu_clock --key-counts 8)

    omnitrace_add_validation_test(
        NAME annotate-sampling
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "annotate"
        ARGS --key-names thread_cpu_clock --key-counts 6)
endif()
