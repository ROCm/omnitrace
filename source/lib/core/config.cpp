// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "config.hpp"
#include "common/defines.h"
#include "constraint.hpp"
#include "debug.hpp"
#include "defines.hpp"
#include "gpu.hpp"
#include "mproc.hpp"
#include "perf.hpp"
#include "perfetto.hpp"
#include "utility.hpp"

#include <asm-generic/errno-base.h>
#include <timemory/backends/capability.hpp>
#include <timemory/backends/dmp.hpp>
#include <timemory/backends/mpi.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/environment.hpp>
#include <timemory/environment/types.hpp>
#include <timemory/log/color.hpp>
#include <timemory/log/logger.hpp>
#include <timemory/manager.hpp>
#include <timemory/process/process.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/settings.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/declaration.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>
#include <timemory/utility/signals.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <linux/capability.h>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>

namespace omnitrace
{
using settings = tim::settings;

namespace
{
TIMEMORY_NOINLINE bool&
_settings_are_configured()
{
    static bool _v = false;
    return _v;
}

auto
get_config()
{
    if(!_settings_are_configured())
    {
        static auto _once = (configure_settings(), true);
        (void) _once;
    }
    return settings::shared_instance();
}

std::string
get_setting_name(std::string _v)
{
    static const auto _prefix = tim::string_view_t{ "rocprofsys_" };
    for(auto& itr : _v)
        itr = tolower(itr);
    auto _pos = _v.find(_prefix);
    if(_pos == 0) return _v.substr(_prefix.length());
    return _v;
}

template <typename Tp>
Tp
get_available_categories()
{
    auto _v = Tp{};
    for(auto itr : { OMNITRACE_PERFETTO_CATEGORIES })
        tim::utility::emplace(_v, itr.name);
    return _v;
}

using utility::parse_numeric_range;

#define OMNITRACE_CONFIG_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE, ...)        \
    [&]() {                                                                              \
        auto _ret = _config->insert<TYPE, TYPE>(                                         \
            ENV_NAME, get_setting_name(ENV_NAME), DESCRIPTION, TYPE{ INITIAL_VALUE },    \
            std::set<std::string>{ "custom", "rocprofsys", "librocprof-sys",              \
                                   __VA_ARGS__ });                                       \
        if(!_ret.second)                                                                 \
        {                                                                                \
            OMNITRACE_PRINT("Warning! Duplicate setting: %s / %s\n",                     \
                            get_setting_name(ENV_NAME).c_str(), ENV_NAME);               \
        }                                                                                \
        return _config->find(ENV_NAME)->second;                                          \
    }()

// below does not include "librocprof-sys"
#define OMNITRACE_CONFIG_EXT_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE, ...)    \
    [&]() {                                                                              \
        auto _ret = _config->insert<TYPE, TYPE>(                                         \
            ENV_NAME, get_setting_name(ENV_NAME), DESCRIPTION, TYPE{ INITIAL_VALUE },    \
            std::set<std::string>{ "custom", "rocprofsys", __VA_ARGS__ });                \
        if(!_ret.second)                                                                 \
        {                                                                                \
            OMNITRACE_PRINT("Warning! Duplicate setting: %s / %s\n",                     \
                            get_setting_name(ENV_NAME).c_str(), ENV_NAME);               \
        }                                                                                \
        return _config->find(ENV_NAME)->second;                                          \
    }()

// setting + command line option
#define OMNITRACE_CONFIG_CL_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE,          \
                                    CMD_LINE, ...)                                       \
    [&]() {                                                                              \
        auto _ret = _config->insert<TYPE, TYPE>(                                         \
            ENV_NAME, get_setting_name(ENV_NAME), DESCRIPTION, TYPE{ INITIAL_VALUE },    \
            std::set<std::string>{ "custom", "rocprofsys", "librocprof-sys",              \
                                   __VA_ARGS__ },                                        \
            std::vector<std::string>{ CMD_LINE });                                       \
        if(!_ret.second)                                                                 \
        {                                                                                \
            OMNITRACE_PRINT("Warning! Duplicate setting: %s / %s\n",                     \
                            get_setting_name(ENV_NAME).c_str(), ENV_NAME);               \
        }                                                                                \
        return _config->find(ENV_NAME)->second;                                          \
    }()
}  // namespace

inline namespace config
{
namespace
{
auto cfg_fini_callbacks = std::vector<std::function<void()>>{};
}

void
finalize()
{
    OMNITRACE_DEBUG("[omnitrace_finalize] Disabling signal handling...\n");
    tim::signals::disable_signal_detection();
    _settings_are_configured() = false;
    for(const auto& itr : cfg_fini_callbacks)
        if(itr) itr();
}

bool
settings_are_configured()
{
    return _settings_are_configured();
}

void
configure_settings(bool _init)
{
    static bool _once = false;
    if(_once) return;
    _once = true;

    if(settings_are_configured()) return;

    if(get_is_continuous_integration() && get_state() < State::Init)
    {
        timemory_print_demangled_backtrace<64>();
        OMNITRACE_THROW("config::configure_settings() called before "
                        "omnitrace_init_library. state = %s",
                        std::to_string(get_state()).c_str());
    }

    tim::manager::add_metadata("OMNITRACE_VERSION", OMNITRACE_VERSION_STRING);
    tim::manager::add_metadata("OMNITRACE_VERSION_MAJOR", OMNITRACE_VERSION_MAJOR);
    tim::manager::add_metadata("OMNITRACE_VERSION_MINOR", OMNITRACE_VERSION_MINOR);
    tim::manager::add_metadata("OMNITRACE_VERSION_PATCH", OMNITRACE_VERSION_PATCH);
    tim::manager::add_metadata("OMNITRACE_GIT_DESCRIBE", OMNITRACE_GIT_DESCRIBE);
    tim::manager::add_metadata("OMNITRACE_GIT_REVISION", OMNITRACE_GIT_REVISION);

    tim::manager::add_metadata("OMNITRACE_LIBRARY_ARCH", OMNITRACE_LIBRARY_ARCH);
    tim::manager::add_metadata("OMNITRACE_SYSTEM_NAME", OMNITRACE_SYSTEM_NAME);
    tim::manager::add_metadata("OMNITRACE_SYSTEM_PROCESSOR", OMNITRACE_SYSTEM_PROCESSOR);
    tim::manager::add_metadata("OMNITRACE_SYSTEM_VERSION", OMNITRACE_SYSTEM_VERSION);

    tim::manager::add_metadata("OMNITRACE_COMPILER_ID", OMNITRACE_COMPILER_ID);
    tim::manager::add_metadata("OMNITRACE_COMPILER_VERSION", OMNITRACE_COMPILER_VERSION);

#if OMNITRACE_HIP_VERSION > 0
    tim::manager::add_metadata("OMNITRACE_HIP_VERSION", OMNITRACE_HIP_VERSION_STRING);
    tim::manager::add_metadata("OMNITRACE_HIP_VERSION_MAJOR",
                               OMNITRACE_HIP_VERSION_MAJOR);
    tim::manager::add_metadata("OMNITRACE_HIP_VERSION_MINOR",
                               OMNITRACE_HIP_VERSION_MINOR);
    tim::manager::add_metadata("OMNITRACE_HIP_VERSION_PATCH",
                               OMNITRACE_HIP_VERSION_PATCH);
#endif

    auto _config = settings::shared_instance();

    // if using timemory, default to perfetto being off
    auto _default_perfetto_v = !tim::get_env<bool>("OMNITRACE_PROFILE", false, false);

    auto _system_backend =
        tim::get_env("OMNITRACE_PERFETTO_BACKEND_SYSTEM", false, false);

    auto _omnitrace_debug = _config->get<bool>("OMNITRACE_DEBUG");
    if(_omnitrace_debug) tim::set_env("TIMEMORY_DEBUG_SETTINGS", "1", 0);

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_MODE",
        "Data collection mode. Used to set default values for OMNITRACE_USE_* options. "
        "Typically set by omnitrace binary instrumenter.",
        std::string{ "trace" }, "backend", "advanced", "mode")
        ->set_choices({ "trace", "sampling", "causal", "coverage" });

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_CI",
                             "Enable some runtime validation checks (typically enabled "
                             "for continuous integration)",
                             false, "debugging", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_MONOCHROME", "Disable colorized logging",
                             false, "debugging", "advanced");

    OMNITRACE_CONFIG_EXT_SETTING(int, "OMNITRACE_DL_VERBOSE",
                                 "Verbosity within the omnitrace-dl library", 0,
                                 "debugging", "librocprof-sys-dl", "advanced");

    OMNITRACE_CONFIG_SETTING(
        size_t, "OMNITRACE_NUM_THREADS_HINT",
        "This is hint for how many threads are expected to be created in the "
        "application. Setting this value allows omnitrace to preallocate resources "
        "during initialization and warn about any potential issues. For example, when "
        "call-stack sampling, each thread has a unique sampler instance which "
        "communicates with an allocator instance running in a background thread. Each "
        "allocator only handles N sampling instances (where N is the value of "
        "OMNITRACE_SAMPLING_ALLOCATOR_SIZE). When this hint is set to >= the number of "
        "threads that get sampled, omnitrace can start all the background threads during "
        "initialization",
        get_env<size_t>("OMNITRACE_NUM_THREADS", 1), "threading", "performance",
        "sampling", "parallelism", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_TRACE", "Enable perfetto backend",
                             _default_perfetto_v, "backend", "perfetto");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_PERFETTO",
                             "[DEPRECATED] Renamed to OMNITRACE_TRACE",
                             _default_perfetto_v, "backend", "perfetto", "deprecated");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_PROFILE", "Enable timemory backend",
                             !_config->get<bool>("OMNITRACE_TRACE"), "backend",
                             "timemory");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_TIMEMORY", "[DEPRECATED] Renamed to OMNITRACE_PROFILE",
        !_config->get<bool>("OMNITRACE_TRACE"), "backend", "timemory", "deprecated");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_CAUSAL",
                             "Enable causal profiling analysis", false, "backend",
                             "causal", "analysis");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_ROCTRACER",
                             "Enable ROCm API and kernel tracing", true, "backend",
                             "roctracer", "rocm");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_ROCPROFILER",
                             "Enable ROCm hardware counters", true, "backend",
                             "rocprofiler", "rocm");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_ROCM_SMI",
        "Enable sampling GPU power, temp, utilization, and memory usage", true, "backend",
        "rocm_smi", "rocm", "process_sampling");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_ROCTX",
        "Enable ROCtx API. Warning! Out-of-order ranges may corrupt perfetto flamegraph",
        false, "backend", "roctracer", "rocm", "roctx");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_SAMPLING",
                             "Enable statistical sampling of call-stack", false,
                             "backend", "sampling");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_THREAD_SAMPLING",
                             "[DEPRECATED] Renamed to OMNITRACE_USE_PROCESS_SAMPLING",
                             true, "backend", "sampling", "process_sampling",
                             "deprecated", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_PROCESS_SAMPLING",
        "Enable a background thread which samples process-level and system metrics "
        "such as the CPU/GPU freq, power, memory usage, etc.",
        true, "backend", "sampling", "process_sampling");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_PID",
        "Enable tagging filenames with process identifier (either MPI rank or pid)", true,
        "io", "filename");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_KOKKOSP",
                             "Enable support for Kokkos Tools", false, "kokkos",
                             "backend");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_MPIP",
                             "Enable support for MPI functions", true, "mpi", "backend",
                             "parallelism");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_RCCLP",
        "Enable support for ROCm Communication Collectives Library (RCCL) Performance",
        false, "rocm", "rccl", "backend");

    OMNITRACE_CONFIG_CL_SETTING(
        bool, "OMNITRACE_KOKKOSP_KERNEL_LOGGER", "Enables kernel logging", false,
        "--omnitrace-kokkos-kernel-logger", "kokkos", "debugging", "advanced");

    OMNITRACE_CONFIG_SETTING(int64_t, "OMNITRACE_KOKKOSP_NAME_LENGTH_MAX",
                             "Set this to a value > 0 to help avoid unnamed Kokkos Tools "
                             "callbacks. Generally, unnamed callbacks are the demangled "
                             "name of the function, which is very long",
                             0, "kokkos", "debugging", "advanced");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_KOKKOSP_PREFIX",
                             "Set to [kokkos] to maintain old naming convention", "",
                             "kokkos", "debugging", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_KOKKOSP_DEEP_COPY",
        "Enable tracking deep copies (warning: may corrupt flamegraph in perfetto)",
        false, "kokkos", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_OMPT",
                             "Enable support for OpenMP-Tools", false, "openmp", "ompt",
                             "backend");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_CODE_COVERAGE",
                             "Enable support for code coverage", false, "coverage",
                             "backend", "advanced");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_TRACE_DELAY",
        "Time in seconds to wait before enabling trace/profile data collection. If "
        "multiple delays + durations are needed, see OMNITRACE_TRACE_PERIODS.",
        0.0, "trace", "profile", "perfetto", "timemory");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_TRACE_DURATION",
        "If > 0.0, time (in seconds) to collect trace/profile data. If multiple delays + "
        "durations are needed, see OMNITRACE_TRACE_PERIODS.",
        0.0, "trace", "profile", "perfetto", "timemory");

    auto _clock_choices = std::vector<std::string>{};
    for(const auto& itr : constraint::get_valid_clock_ids())
    {
        _clock_choices.emplace_back(
            join("", "(", join('|', itr.name, itr.value, itr.raw_name), ")"));
    }

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_TRACE_PERIODS",
                             "Similar to specify trace delay and/or duration except in "
                             "the form <DELAY>:<DURATION>, <DELAY>:<DURATION>:<REPEAT>, "
                             "and/or <DELAY>:<DURATION>:<REPEAT>:<CLOCK_ID>",
                             std::string{}, "trace", "profile", "perfetto", "timemory");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_TRACE_PERIOD_CLOCK_ID",
        "Set the default clock ID for OMNITRACE_TRACE_DELAY, OMNITRACE_TRACE_DURATION, "
        "and/or OMNITRACE_TRACE_PERIODS. E.g. \"realtime\" == the delay/duration is "
        "governed by the elapsed realtime, \"cputime\" == the delay/duration is governed "
        "by the elapsed CPU-time within the process, etc. Note: when using CPU-based "
        "timing, it is recommened to scale the value by the number of threads and be "
        "aware that omnitrace may contribute to advancing the process CPU-time",
        "CLOCK_REALTIME", "trace", "profile", "perfetto", "timemory")
        ->set_choices(_clock_choices);

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_SAMPLING_FREQ",
        "Number of software interrupts per second when OMNITTRACE_USE_SAMPLING=ON", 300.0,
        "sampling", "process_sampling");

    OMNITRACE_CONFIG_SETTING(double, "OMNITRACE_SAMPLING_CPUTIME_FREQ",
                             "Number of software interrupts per second of CPU-time. "
                             "Defaults to OMNITRACE_SAMPLING_FREQ when <= 0.0",
                             -1.0, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_SAMPLING_REALTIME_FREQ",
        "Number of software interrupts per second of real (wall) time. "
        "Defaults to OMNITRACE_SAMPLING_FREQ when <= 0.0",
        -1.0, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(double, "OMNITRACE_SAMPLING_OVERFLOW_FREQ",
                             "Number of events in between each sample. "
                             "Defaults to OMNITRACE_SAMPLING_FREQ when <= 0.0",
                             -1.0, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_SAMPLING_DELAY",
        "Time (in seconds) to wait before the first sampling signal is delivered, "
        "increasing this value can fix deadlocks during init",
        0.5, "sampling", "process_sampling");

    OMNITRACE_CONFIG_SETTING(double, "OMNITRACE_SAMPLING_CPUTIME_DELAY",
                             "Time (in seconds) to wait before the first CPU-time "
                             "sampling signal is delivered. "
                             "Defaults to OMNITRACE_SAMPLING_DELAY when <= 0.0",
                             -1.0, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_SAMPLING_REALTIME_DELAY",
        "Time (in seconds) to wait before the first real (wall) time sampling signal is "
        "delivered. Defaults to OMNITRACE_SAMPLING_DELAY when <= 0.0",
        -1.0, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(double, "OMNITRACE_SAMPLING_DURATION",
                             "If > 0.0, time (in seconds) to sample before stopping", 0.0,
                             "sampling", "process_sampling");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_PROCESS_SAMPLING_FREQ",
        "Number of measurements per second when OMNITTRACE_USE_PROCESS_SAMPLING=ON. If "
        "set to zero, uses OMNITRACE_SAMPLING_FREQ value",
        0.0, "process_sampling");

    OMNITRACE_CONFIG_SETTING(double, "OMNITRACE_PROCESS_SAMPLING_DURATION",
                             "If > 0.0, time (in seconds) to sample before stopping. If "
                             "less than zero, uses OMNITRACE_SAMPLING_DURATION",
                             -1.0, "sampling", "process_sampling");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_SAMPLING_CPUS",
        "CPUs to collect frequency information for. Values should be separated by commas "
        "and can be explicit or ranges, e.g. 0,1,5-8. An empty value implies 'all' and "
        "'none' suppresses all CPU frequency sampling",
        std::string{}, "process_sampling");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_ROCM_SMI_DEVICES",
                             "[DEPRECATED] Renamed to OMNITRACE_SAMPLING_GPUS",
                             std::string{ "all" }, "rocm_smi", "rocm", "process_sampling",
                             "deprecated", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_SAMPLING_GPUS",
        "Devices to query when OMNITRACE_USE_ROCM_SMI=ON. Values should be separated by "
        "commas and can be explicit or ranges, e.g. 0,1,5-8. An empty value implies "
        "'all' and 'none' suppresses all GPU sampling",
        std::string{ "all" }, "rocm_smi", "rocm", "process_sampling");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_SAMPLING_TIDS",
        "Limit call-stack sampling to specific thread IDs, starting at zero for the main "
        "thread. Be aware that some libraries, such as ROCm may create additional "
        "threads which increment the TID count. However, no threads started by omnitrace "
        "will increment the TID count. Values should be separated by commas and can be "
        "explicit or ranges, e.g. 0,1,5-8. An empty value implies all TIDs.",
        std::string{}, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_SAMPLING_CPUTIME_TIDS",
        "Same as OMNITRACE_SAMPLING_TIDS but applies specifically to samplers whose "
        "timers are based on the CPU-time. This is useful when you want to restrict "
        "samples to particular threads.",
        std::string{}, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_SAMPLING_REALTIME_TIDS",
        "Same as OMNITRACE_SAMPLING_TIDS but applies specifically to samplers whose "
        "timers are based on the real (wall) time. This is useful when you want to "
        "restrict samples to particular threads.",
        std::string{}, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_SAMPLING_OVERFLOW_TIDS",
        "Same as OMNITRACE_SAMPLING_TIDS but applies specifically to samplers whose "
        "samples are based on the overflow of a particular event. This is useful when "
        "you want to restrict samples to particular threads.",
        std::string{}, "sampling", "advanced");

    auto _backend = tim::get_env_choice<std::string>(
        "OMNITRACE_PERFETTO_BACKEND",
        (_system_backend) ? "system"      // if OMNITRACE_PERFETTO_BACKEND_SYSTEM is true,
                                          // default to system.
                          : "inprocess",  // Otherwise, default to inprocess
        { "inprocess", "system", "all" }, false);

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_PERFETTO_BACKEND",
                             "Specify the perfetto backend to activate. Options are: "
                             "'inprocess', 'system', or 'all'",
                             _backend, "perfetto")
        ->set_choices({ "inprocess", "system", "all" });

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_TRACE_THREAD_LOCKS",
                             "Enable tracing calls to pthread_mutex_lock, "
                             "pthread_mutex_unlock, pthread_mutex_trylock",
                             false, "backend", "parallelism", "gotcha", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_TRACE_THREAD_RW_LOCKS",
                             "Enable tracing calls to pthread_rwlock_* functions. May "
                             "cause deadlocks with ROCm-enabled OpenMPI.",
                             false, "backend", "parallelism", "gotcha", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_TRACE_THREAD_SPIN_LOCKS",
                             "Enable tracing calls to pthread_spin_* functions. May "
                             "cause deadlocks with MPI distributions.",
                             false, "backend", "parallelism", "gotcha", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_TRACE_THREAD_BARRIERS",
                             "Enable tracing calls to pthread_barrier functions.", true,
                             "backend", "parallelism", "gotcha", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_TRACE_THREAD_JOIN",
                             "Enable tracing calls to pthread_join functions.", true,
                             "backend", "parallelism", "gotcha", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_SAMPLING_KEEP_INTERNAL",
        "Configure whether the statistical samples should include call-stack entries "
        "from internal routines in omnitrace. E.g. when ON, the call-stack will show "
        "functions like omnitrace_push_trace. If disabled, omnitrace will attempt to "
        "filter out internal routines from the sampling call-stacks",
        true, "sampling", "data", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_SAMPLING_INCLUDE_INLINES",
                             "Create entries for inlined functions when available", false,
                             "sampling", "data", "advanced");

    OMNITRACE_CONFIG_SETTING(
        size_t, "OMNITRACE_SAMPLING_ALLOCATOR_SIZE",
        "The number of sampled threads handled by an allocator running in a background "
        "thread. Each thread that is sampled communicates with an allocator running in a "
        "background thread which handles storing/caching the data when it's buffer is "
        "full. Setting this value too high (i.e. equal to the number of threads when the "
        "thread count is high) may cause loss of data -- the sampler may fill a new "
        "buffer and overwrite old buffer data before the allocator can process it. "
        "Setting this value to 1 will result in a background allocator thread for every "
        "thread started by the application.",
        8, "sampling", "debugging", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_SAMPLING_OVERFLOW",
                             "Enable sampling via an overflow of a HW counter. This "
                             "requires Linux perf (/proc/sys/kernel/perf_event_paranoid "
                             "created by OS) with a value of 2 or less in that file",
                             false, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_SAMPLING_REALTIME",
        "Enable sampling frequency via a wall-clock timer. This may result in typically "
        "idle child threads consuming an unnecessary large amount of CPU time.",
        false, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_SAMPLING_CPUTIME",
        "Enable sampling frequency via a timer that measures both CPU time used by the "
        "current process, and CPU time expended on behalf of the process by the system. "
        "This is recommended.",
        false, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(int, "OMNITRACE_SAMPLING_CPUTIME_SIGNAL",
                             "Modify this value only if the target process is also using "
                             "the same signal (SIGPROF)",
                             SIGPROF, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(int, "OMNITRACE_SAMPLING_REALTIME_SIGNAL",
                             "Modify this value only if the target process is also using "
                             "the same signal (SIGRTMIN)",
                             SIGRTMIN, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(int, "OMNITRACE_SAMPLING_OVERFLOW_SIGNAL",
                             "Modify this value only if the target process is also using "
                             "the same signal (SIGRTMIN + 1)",
                             SIGRTMIN + 1, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_SAMPLING_OVERFLOW_EVENT",
                             "Metric for overflow sampling",
                             std::string{ "perf::PERF_COUNT_HW_CACHE_REFERENCES" },
                             "sampling", "hardware_counters")
        ->set_choices(perf::get_config_choices());

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_HIP_API",
                             "Enable HIP API tracing support", true, "roctracer", "rocm",
                             "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_ROCTRACER_HIP_API_BACKTRACE",
        "Enable annotating the perfetto debug annotation with backtraces", false,
        "roctracer", "rocm", "perfetto", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_HIP_ACTIVITY",
                             "Enable HIP activity tracing support", true, "roctracer",
                             "rocm", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_HSA_ACTIVITY",
                             "Enable HSA activity tracing support", false, "roctracer",
                             "rocm", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_HSA_API",
                             "Enable HSA API tracing support", false, "roctracer", "rocm",
                             "advanced");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_ROCTRACER_HSA_API_TYPES",
                             "HSA API type to collect", "", "roctracer", "rocm",
                             "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_DISCARD_BARRIERS",
                             "Skip barrier marker events in traces", false, "roctracer",
                             "rocm", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_ROCM_EVENTS",
        "ROCm hardware counters. Use ':device=N' syntax to specify collection on device "
        "number N, e.g. ':device=0'. If no device specification is provided, the event "
        "is collected on every available device",
        "", "rocprofiler", "rocm", "hardware_counters");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_ROCM_SMI_METRICS",
                             "rocm-smi metrics to collect: busy, temp, power, mem_usage",
                             "busy,temp,power,mem_usage", "backend", "rocm_smi", "rocm",
                             "process_sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(size_t, "OMNITRACE_PERFETTO_SHMEM_SIZE_HINT_KB",
                             "Hint for shared-memory buffer size in perfetto (in KB)",
                             size_t{ 4096 }, "perfetto", "data", "advanced");

    OMNITRACE_CONFIG_SETTING(size_t, "OMNITRACE_PERFETTO_BUFFER_SIZE_KB",
                             "Size of perfetto buffer (in KB)", size_t{ 1024000 },
                             "perfetto", "data");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_PERFETTO_COMBINE_TRACES",
                             "Combine Perfetto traces. If not explicitly set, it will "
                             "default to the value of OMNITRACE_COLLAPSE_PROCESSES",
                             false, "perfetto", "data", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_PERFETTO_ROCTRACER_PER_STREAM",
        "Separate roctracer GPU side traces (copies, kernels) into separate "
        "tracks based on the stream they're enqueued into",
        true, "perfetto", "roctracer", "rocm", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_PERFETTO_FILL_POLICY",
        "Behavior when perfetto buffer is full. 'discard' will ignore new entries, "
        "'ring_buffer' will overwrite old entries",
        "discard", "perfetto", "data")
        ->set_choices({ "fill", "discard" });

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_ENABLE_CATEGORIES",
                             "Enable collecting profiling and trace data for these "
                             "categories and disable all other categories",
                             "", "trace", "profile", "perfetto", "timemory", "data",
                             "category", "advanced")
        ->set_choices(get_available_categories<std::vector<std::string>>());

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_DISABLE_CATEGORIES",
        "Disable collecting profiling and trace data for these categories", "", "trace",
        "profile", "perfetto", "timemory", "data", "category", "advanced")
        ->set_choices(get_available_categories<std::vector<std::string>>());

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_PERFETTO_ANNOTATIONS",
                             "Include debug annotations in perfetto trace. When enabled, "
                             "this feature will encode information such as the values of "
                             "the function arguments (when available). Disabling this "
                             "feature may dramatically reduce the size of the trace",
                             true, "perfetto", "data", "debugging", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_PERFETTO_COMPACT_ROCTRACER_ANNOTATIONS",
        "When PERFETTO_ANNOTATIONS, USE_ROCTRACER, and ROCTRACER_HIP_API are all "
        "enabled, enabling this option will result in the arg information for HIP API "
        "calls to all be within one annotation (e.g., args=\"stream=0x0, dst=0x1F, "
        "sizeBytes=64, src=0x08, kind=1\"). When disabled, each parameter will be an "
        "individual annotation (e.g. stream, dst, sizeBytes, etc.). The benefit of the "
        "former is that it is faster to serialize and consumes less file space; the "
        "benefit of the latter is that it becomes much easier to find slices in the "
        "trace with the same value",
        false, "perfetto", "data", "debugging", "roctracer", "rocm", "advanced");

    OMNITRACE_CONFIG_SETTING(
        uint64_t, "OMNITRACE_THREAD_POOL_SIZE",
        "Max number of threads for processing background tasks",
        std::max<uint64_t>(std::min<uint64_t>(4, std::thread::hardware_concurrency() / 2),
                           1),
        "parallelism", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_TIMEMORY_COMPONENTS",
        "List of components to collect via timemory (see `rocprof-sys-avail -C`)",
        "wall_clock", "timemory", "component");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_OUTPUT_FILE",
                             "[DEPRECATED] See OMNITRACE_PERFETTO_FILE", std::string{},
                             "perfetto", "io", "filename", "deprecated", "advanced");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_PERFETTO_FILE", "Perfetto filename",
                             std::string{ "perfetto-trace.proto" }, "perfetto", "io",
                             "filename", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_TEMPORARY_FILES",
                             "Write data to temporary files to minimize the memory usage "
                             "of omnitrace, e.g. call-stack samples will be periodically "
                             "written to a file and re-loaded during finalization",
                             true, "io", "data", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_TMPDIR", "Base directory for temporary files",
        get_env<std::string>("TMPDIR", "/tmp"), "io", "data", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_CAUSAL_BACKEND",
        "Backend for call-stack sampling. See "
        "https://rocm.github.io/omnitrace/causal_profiling.html#backends for more "
        "info. If set to \"auto\", omnitrace will attempt to use the perf backend and "
        "fallback on the timer backend if unavailable",
        std::string{ "auto" }, "causal", "analysis")
        ->set_choices({ "auto", "perf", "timer" });

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_CAUSAL_MODE",
        "Perform causal experiments at the function-scope or line-scope. Ideally, use "
        "function first to locate function with highest impact and then switch to line "
        "mode + OMNITRACE_CAUSAL_FUNCTION_SCOPE set to the function being targeted.",
        std::string{ "function" }, "causal", "analysis")
        ->set_choices({ "func", "line", "function" });

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_CAUSAL_DELAY",
        "Length of time to wait (in seconds) before starting the first causal experiment",
        0.0, "causal", "analysis");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_CAUSAL_DURATION",
        "Length of time to perform causal experimentation (in seconds) after the first "
        "experiment has started. After this amount of time has elapsed, no more causal "
        "experiments will be performed and the application will continue without any "
        "overhead from causal profiling. Any value <= 0 means until the application "
        "completes",
        0.0, "causal", "analysis");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_CAUSAL_END_TO_END",
        "Perform causal experiment over the length of the entire application", false,
        "causal", "analysis", "advanced");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_CAUSAL_FILE",
                             "Name of causal output filename (w/o extension)",
                             std::string{ "experiments" }, "causal", "analysis",
                             "advanced", "io");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_CAUSAL_FILE_RESET",
        "Overwrite any existing causal output file instead of appending to it", false,
        "causal", "analysis", "advanced", "io");

    OMNITRACE_CONFIG_SETTING(
        uint64_t, "OMNITRACE_CAUSAL_RANDOM_SEED",
        "Seed for random number generator which selects speedups and experiments -- "
        "please note that the lines selected for experimentation are not reproducible "
        "but the speedup selection is. If set to zero, std::random_device{}() will be "
        "used.",
        0, "causal", "analysis");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_CAUSAL_FIXED_SPEEDUP",
                             "List of virtual speedups between 0 and 100 (inclusive) to "
                             "sample from for causal profiling",
                             std::string{}, "causal", "analysis", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_CAUSAL_BINARY_SCOPE",
        "Limits causal experiments to the binaries matching the provided list of regular "
        "expressions (separated by tab, semi-colon, and/or quotes (single or double))",
        std::string{ "%MAIN%" }, "causal", "analysis");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_CAUSAL_SOURCE_SCOPE",
        "Limits causal experiments to the source files or source file + lineno pair "
        "(i.e. <file> or <file>:<line>) matching the provided list of regular "
        "expressions (separated by tab, semi-colon, and/or quotes (single or double))",
        std::string{}, "causal", "analysis");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_CAUSAL_FUNCTION_SCOPE",
        "List of <function> regex entries for causal profiling (separated by tab, "
        "semi-colon, and/or quotes (single or double))",
        std::string{}, "causal", "analysis");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_CAUSAL_BINARY_EXCLUDE",
        "Excludes binaries matching the list of provided regexes from causal experiments "
        "(separated by tab, semi-colon, and/or quotes (single or double))",
        std::string{}, "causal", "analysis");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_CAUSAL_SOURCE_EXCLUDE",
        "Excludes source files or source file + lineno pair (i.e. <file> or "
        "<file>:<line>) matching the list of provided regexes from causal experiments "
        "(separated by tab, semi-colon, and/or quotes (single or double))",
        std::string{}, "causal", "analysis");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_CAUSAL_FUNCTION_EXCLUDE",
        "Excludes functions matching the list of provided regexes from causal "
        "experiments (separated by tab, semi-colon, and/or quotes (single or double))",
        std::string{}, "causal", "analysis");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_CAUSAL_FUNCTION_EXCLUDE_DEFAULTS",
        "This controls adding a series of function exclude regexes to avoid "
        "experimenting on STL implementation functions, etc. which are, "
        "generally, not helpful. Details: excludes demangled function names "
        "starting with '_' or containing '::_M'.",
        true, "causal", "analysis", "advanced");

    // set the defaults
    _config->get_flamegraph_output()     = false;
    _config->get_ctest_notes()           = false;
    _config->get_cout_output()           = false;
    _config->get_file_output()           = true;
    _config->get_json_output()           = true;
    _config->get_tree_output()           = true;
    _config->get_enable_signal_handler() = true;
    _config->get_collapse_processes()    = false;
    _config->get_collapse_threads()      = false;
    _config->get_stack_clearing()        = false;
    _config->get_time_output()           = true;
    _config->get_timing_precision()      = 6;
    _config->get_max_thread_bookmarks()  = 1;
    _config->get_timing_units()          = "sec";
    _config->get_memory_units()          = "MB";

    // settings native to timemory but critically and/or extensively used by omnitrace
    auto _add_omnitrace_category = [&_config](auto itr) {
        if(itr != _config->end())
        {
            auto _categories = itr->second->get_categories();
            _categories.emplace("omnitrace");
            _categories.emplace("librocprof-sys");
            itr->second->set_categories(_categories);
        }
    };

    _add_omnitrace_category(_config->find("OMNITRACE_CONFIG_FILE"));
    _add_omnitrace_category(_config->find("OMNITRACE_DEBUG"));
    _add_omnitrace_category(_config->find("OMNITRACE_VERBOSE"));
    _add_omnitrace_category(_config->find("OMNITRACE_TIME_OUTPUT"));
    _add_omnitrace_category(_config->find("OMNITRACE_OUTPUT_PREFIX"));
    _add_omnitrace_category(_config->find("OMNITRACE_OUTPUT_PATH"));

    auto _add_advanced_category = [&_config](const std::string& _name) {
        auto itr = _config->find(_name);
        if(itr != _config->end())
        {
            auto _categories = itr->second->get_categories();
            _categories.emplace("advanced");
            itr->second->set_categories(_categories);
        }
        else
        {
            if(_config->get<bool>("OMNITRACE_CI"))
            {
                OMNITRACE_THROW("Error! Setting '%s' not found!", _name.c_str());
            }
        }
    };

    _add_advanced_category("OMNITRACE_CPU_AFFINITY");
    _add_advanced_category("OMNITRACE_COUT_OUTPUT");
    _add_advanced_category("OMNITRACE_FILE_OUTPUT");
    _add_advanced_category("OMNITRACE_JSON_OUTPUT");
    _add_advanced_category("OMNITRACE_TREE_OUTPUT");
    _add_advanced_category("OMNITRACE_TEXT_OUTPUT");
    _add_advanced_category("OMNITRACE_DIFF_OUTPUT");
    _add_advanced_category("OMNITRACE_DEBUG");
    _add_advanced_category("OMNITRACE_ENABLE_SIGNAL_HANDLER");
    _add_advanced_category("OMNITRACE_FLAT_PROFILE");
    _add_advanced_category("OMNITRACE_INPUT_EXTENSIONS");
    _add_advanced_category("OMNITRACE_INPUT_PATH");
    _add_advanced_category("OMNITRACE_INPUT_PREFIX");
    _add_advanced_category("OMNITRACE_MAX_DEPTH");
    _add_advanced_category("OMNITRACE_MAX_WIDTH");
    _add_advanced_category("OMNITRACE_MEMORY_PRECISION");
    _add_advanced_category("OMNITRACE_MEMORY_SCIENTIFIC");
    _add_advanced_category("OMNITRACE_MEMORY_UNITS");
    _add_advanced_category("OMNITRACE_MEMORY_WIDTH");
    _add_advanced_category("OMNITRACE_NETWORK_INTERFACE");
    _add_advanced_category("OMNITRACE_NODE_COUNT");
    _add_advanced_category("OMNITRACE_PAPI_FAIL_ON_ERROR");
    _add_advanced_category("OMNITRACE_PAPI_OVERFLOW");
    _add_advanced_category("OMNITRACE_PAPI_MULTIPLEXING");
    _add_advanced_category("OMNITRACE_PAPI_QUIET");
    _add_advanced_category("OMNITRACE_PAPI_THREADING");
    _add_advanced_category("OMNITRACE_PRECISION");
    _add_advanced_category("OMNITRACE_SCIENTIFIC");
    _add_advanced_category("OMNITRACE_STRICT_CONFIG");
    _add_advanced_category("OMNITRACE_TIMELINE_PROFILE");
    _add_advanced_category("OMNITRACE_SCIENTIFIC");
    _add_advanced_category("OMNITRACE_TIME_FORMAT");
    _add_advanced_category("OMNITRACE_TIMING_PRECISION");
    _add_advanced_category("OMNITRACE_TIMING_SCIENTIFIC");
    _add_advanced_category("OMNITRACE_TIMING_UNITS");
    _add_advanced_category("OMNITRACE_TIMING_WIDTH");
    _add_advanced_category("OMNITRACE_WIDTH");
    _add_advanced_category("OMNITRACE_COLLAPSE_THREADS");
    _add_advanced_category("OMNITRACE_COLLAPSE_PROCESSES");

#if defined(TIMEMORY_USE_PAPI)
    int _paranoid = 2;
    {
        std::ifstream _fparanoid{ "/proc/sys/kernel/perf_event_paranoid" };
        if(_fparanoid) _fparanoid >> _paranoid;
    }

    auto  _cap_status        = timemory::linux::capability::cap_read(process::get_id());
    auto* _cap_data          = &_cap_status.effective;
    bool  _has_cap_sys_admin = false;
    for(auto itr : timemory::linux::capability::cap_decode(*_cap_data))
        if(itr == CAP_SYS_ADMIN) _has_cap_sys_admin = true;

    if(_paranoid > 2 && !_has_cap_sys_admin)
    {
        OMNITRACE_BASIC_VERBOSE(0,
                                "/proc/sys/kernel/perf_event_paranoid has a value of %i. "
                                "Disabling PAPI (requires a value <= 2)...\n",
                                _paranoid);
        OMNITRACE_BASIC_VERBOSE(
            0, "In order to enable PAPI support, run 'echo N | sudo tee "
               "/proc/sys/kernel/perf_event_paranoid' where N is <= 2\n");
        trait::runtime_enabled<comp::papi_config>::set(false);
        trait::runtime_enabled<comp::papi_common<void>>::set(false);
        trait::runtime_enabled<comp::papi_array_t>::set(false);
        trait::runtime_enabled<comp::papi_vector>::set(false);
        trait::runtime_enabled<comp::cpu_roofline_flops>::set(false);
        trait::runtime_enabled<comp::cpu_roofline_dp_flops>::set(false);
        trait::runtime_enabled<comp::cpu_roofline_sp_flops>::set(false);
        _config->get_papi_events() = std::string{};
    }
    else
    {
        auto _papi_events = _config->find("OMNITRACE_PAPI_EVENTS");
        _add_omnitrace_category(_papi_events);
        std::vector<std::string> _papi_choices = {};
        for(auto itr : tim::papi::available_events_info())
        {
            if(itr.available()) _papi_choices.emplace_back(itr.symbol());
        }
        _papi_events->second->set_choices(_papi_choices);
    }
#else
    _config->find("OMNITRACE_PAPI_EVENTS")->second->set_hidden(true);
    _config->get_papi_quiet() = true;
#endif

    // always initialize timemory because gotcha wrappers are always used
    auto _cmd     = tim::read_command_line(process::get_id());
    auto _cmd_env = tim::get_env<std::string>("OMNITRACE_COMMAND_LINE", "");
    if(!_cmd_env.empty()) _cmd = tim::delimit(_cmd_env, " ");
    auto _exe          = (_cmd.empty()) ? "exe" : _cmd.front();
    get_exe_realpath() = filepath::realpath(_exe, nullptr, false);
    auto _pos          = _exe.find_last_of('/');
    if(_pos < _exe.length() - 1) _exe = _exe.substr(_pos + 1);
    get_exe_name() = _exe;
    _config->set_tag(_exe);

    bool _found_sep = false;
    for(const auto& itr : _cmd)
    {
        if(itr == "--") _found_sep = true;
    }
    if(!_found_sep && _cmd.size() > 1) _cmd.insert(_cmd.begin() + 1, "--");

    auto _pid       = getpid();
    auto _ppid      = getppid();
    auto _proc      = mproc::get_concurrent_processes(_ppid);
    bool _main_proc = (_proc.size() < 2 || *_proc.begin() == _pid);

    for(auto&& itr :
        tim::delimit(_config->get<std::string>("OMNITRACE_CONFIG_FILE"), ";:"))
    {
        if(_config->get_suppress_config()) continue;

        OMNITRACE_BASIC_VERBOSE(1, "Reading config file %s\n", itr.c_str());
        if(_config->read(itr) && _main_proc &&
           ((_config->get<bool>("OMNITRACE_CI") && settings::verbose() >= 0) ||
            settings::verbose() >= 1 || settings::debug()))
        {
            auto              fitr = settings::format(itr, _config->get_tag());
            std::ifstream     _in{ fitr };
            std::stringstream _iss{};
            while(_in)
            {
                std::string _line{};
                getline(_in, _line);
                _iss << _line << "\n";
            }
            if(!_iss.str().empty())
            {
                OMNITRACE_BASIC_VERBOSE(1, "config file '%s':\n%s\n", fitr.c_str(),
                                        _iss.str().c_str());
            }
        }
    }

    settings::suppress_config() = true;

    if(get_env("OMNITRACE_MONOCHROME", _config->get<bool>("OMNITRACE_MONOCHROME")))
        tim::log::monochrome() = true;

    if(_init)
    {
        using argparser_t = tim::argparse::argument_parser;
        argparser_t _parser{ _exe };
        tim::timemory_init(_cmd, _parser, "rocprofsys-");
    }

#if !defined(OMNITRACE_USE_MPI) && !defined(OMNITRACE_USE_MPI_HEADERS)
    set_setting_value("OMNITRACE_USE_MPIP", false);
#endif

    _config->get_global_components() =
        _config->get<std::string>("OMNITRACE_TIMEMORY_COMPONENTS");

    auto _combine_perfetto_traces = _config->find("OMNITRACE_PERFETTO_COMBINE_TRACES");
    if(!_combine_perfetto_traces->second->get_environ_updated() &&
       _combine_perfetto_traces->second->get_config_updated())
    {
        _combine_perfetto_traces->second->set(_config->get<bool>("collapse_processes"));
    }

    handle_deprecated_setting("OMNITRACE_ROCM_SMI_DEVICES", "OMNITRACE_SAMPLING_GPUS");
    handle_deprecated_setting("OMNITRACE_USE_THREAD_SAMPLING",
                              "OMNITRACE_USE_PROCESS_SAMPLING");
    handle_deprecated_setting("OMNITRACE_OUTPUT_FILE", "OMNITRACE_PERFETTO_FILE");
    handle_deprecated_setting("OMNITRACE_USE_PERFETTO", "OMNITRACE_TRACE");
    handle_deprecated_setting("OMNITRACE_USE_TIMEMORY", "OMNITRACE_PROFILE");

    scope::get_fields()[scope::flat::value]     = _config->get_flat_profile();
    scope::get_fields()[scope::timeline::value] = _config->get_timeline_profile();

    settings::suppress_parsing()  = true;
    settings::use_output_suffix() = _config->get<bool>("OMNITRACE_USE_PID");
    if(settings::use_output_suffix())
        settings::default_process_suffix() = process::get_id();
#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    if(tim::dmp::is_initialized()) settings::default_process_suffix() = tim::dmp::rank();
#endif

    auto _dl_verbose = _config->find("OMNITRACE_DL_VERBOSE");
    if(_dl_verbose->second->get_config_updated())
        tim::set_env(std::string{ _dl_verbose->first }, _dl_verbose->second->as_string(),
                     0);

    if(_config->get_papi_events().empty())
    {
        trait::runtime_enabled<comp::papi_config>::set(false);
        trait::runtime_enabled<comp::papi_common<void>>::set(false);
        trait::runtime_enabled<comp::papi_array_t>::set(false);
        trait::runtime_enabled<comp::papi_vector>::set(false);
    }

    configure_mode_settings(_config);
    configure_signal_handler(_config);
    configure_disabled_settings(_config);

    OMNITRACE_BASIC_VERBOSE(2, "configuration complete\n");

    _settings_are_configured() = true;
}

void
configure_mode_settings(const std::shared_ptr<settings>& _config)
{
    auto _set = [](const std::string& _name, bool _v) {
        if(!set_setting_value(_name, _v))
        {
            OMNITRACE_BASIC_VERBOSE(
                4, "[configure_mode_settings] No configuration setting named '%s'...\n",
                _name.data());
        }
        else
        {
            bool _changed = get_setting_value<bool>(_name).value_or(!_v) != _v;
            OMNITRACE_BASIC_VERBOSE(
                1 && _changed,
                "[configure_mode_settings] Overriding %s to %s in %s mode...\n",
                _name.c_str(), JOIN("", std::boolalpha, _v).c_str(),
                std::to_string(get_mode()).c_str());
        }
    };

    auto _use_causal = get_setting_value<bool>("OMNITRACE_USE_CAUSAL");
    if(_use_causal && *_use_causal) set_env("OMNITRACE_MODE", "causal", 1);

    if(get_mode() == Mode::Coverage)
    {
        set_default_setting_value("OMNITRACE_USE_CODE_COVERAGE", true);
        _set("OMNITRACE_TRACE", false);
        _set("OMNITRACE_PROFILE", false);
        _set("OMNITRACE_USE_CAUSAL", false);
        _set("OMNITRACE_USE_ROCM_SMI", false);
        _set("OMNITRACE_USE_ROCTRACER", false);
        _set("OMNITRACE_USE_ROCPROFILER", false);
        _set("OMNITRACE_USE_KOKKOSP", false);
        _set("OMNITRACE_USE_RCCLP", false);
        _set("OMNITRACE_USE_OMPT", false);
        _set("OMNITRACE_USE_SAMPLING", false);
        _set("OMNITRACE_USE_PROCESS_SAMPLING", false);
    }
    else if(get_mode() == Mode::Causal)
    {
        _set("OMNITRACE_USE_CAUSAL", true);
        _set("OMNITRACE_TRACE", false);
        _set("OMNITRACE_PROFILE", false);
        _set("OMNITRACE_USE_SAMPLING", false);
        _set("OMNITRACE_USE_PROCESS_SAMPLING", false);
    }
    else if(get_mode() == Mode::Sampling)
    {
        set_default_setting_value("OMNITRACE_USE_SAMPLING", true);
        set_default_setting_value("OMNITRACE_USE_PROCESS_SAMPLING", true);
    }

    if(gpu::device_count() == 0)
    {
#if OMNITRACE_HIP_VERSION > 0
        OMNITRACE_BASIC_VERBOSE(1, "No HIP devices were found: disabling roctracer, "
                                   "rocprofiler, and rocm_smi...\n");
#endif
        _set("OMNITRACE_USE_ROCPROFILER", false);
        _set("OMNITRACE_USE_ROCTRACER", false);
        _set("OMNITRACE_USE_ROCM_SMI", false);
    }

    if(_config->get<bool>("OMNITRACE_USE_KOKKOSP"))
    {
        auto _current_kokkosp_lib = tim::get_env<std::string>("KOKKOS_PROFILE_LIBRARY");
        if(_current_kokkosp_lib.find("librocprof-sys-dl.so") == std::string::npos &&
           _current_kokkosp_lib.find("librocprof-sys.so") == std::string::npos)
        {
            auto        _force   = 0;
            std::string _message = {};
            if(std::regex_search(_current_kokkosp_lib, std::regex{ "libtimemory\\." }))
            {
                _force = 1;
                _message =
                    JOIN("", " (forced. Previous value: '", _current_kokkosp_lib, "')");
            }
            OMNITRACE_BASIC_VERBOSE_F(1, "Setting KOKKOS_PROFILE_LIBRARY=%s%s\n",
                                      "librocprof-sys.so", _message.c_str());
            tim::set_env("KOKKOS_PROFILE_LIBRARY", "librocprof-sys.so", _force);
        }
    }

    // recycle all subsequent thread ids
    threading::recycle_ids() = tim::get_env<bool>(
        "OMNITRACE_RECYCLE_TIDS", !_config->get<bool>("OMNITRACE_USE_SAMPLING"));

    if(!_config->get_enabled())
    {
        _set("OMNITRACE_USE_TRACE", false);
        _set("OMNITRACE_PROFILE", false);
        _set("OMNITRACE_USE_CAUSAL", false);
        _set("OMNITRACE_USE_ROCM_SMI", false);
        _set("OMNITRACE_USE_ROCTRACER", false);
        _set("OMNITRACE_USE_ROCPROFILER", false);
        _set("OMNITRACE_USE_KOKKOSP", false);
        _set("OMNITRACE_USE_RCCLP", false);
        _set("OMNITRACE_USE_OMPT", false);
        _set("OMNITRACE_USE_SAMPLING", false);
        _set("OMNITRACE_USE_PROCESS_SAMPLING", false);
        _set("OMNITRACE_USE_CODE_COVERAGE", false);
        set_setting_value("OMNITRACE_TIMEMORY_COMPONENTS", std::string{});
        set_setting_value("OMNITRACE_PAPI_EVENTS", std::string{});
    }
}

namespace
{
using signal_settings = tim::signals::signal_settings;
using sys_signal      = tim::signals::sys_signal;

std::atomic<signal_handler_t>&
get_signal_handler()
{
    static auto _v = std::atomic<signal_handler_t>{ nullptr };
    return _v;
}

void
omnitrace_exit_action(int nsig)
{
    tim::signals::block_signals(get_sampling_signals(),
                                tim::signals::sigmask_scope::process);
    OMNITRACE_BASIC_PRINT("Finalizing after signal %i :: %s\n", nsig,
                          signal_settings::str(static_cast<sys_signal>(nsig)).c_str());
    auto _handler = get_signal_handler().load();
    if(_handler) (*_handler)();
    kill(process::get_id(), nsig);
}

void
omnitrace_trampoline_handler(int _v)
{
    if(get_verbose_env() >= 1)
    {
        ::omnitrace::debug::flush();
        ::omnitrace::debug::lock _debug_lk{};
        OMNITRACE_FPRINTF_STDERR_COLOR(warning);
        fprintf(::omnitrace::debug::get_file(),
                "signal %i ignored (OMNITRACE_IGNORE_DYNINST_TRAMPOLINE=ON)\n", _v);
        ::omnitrace::debug::flush();
        timemory_print_demangled_backtrace<64>();
    }
}
}  // namespace

signal_handler_t
set_signal_handler(signal_handler_t _func)
{
    if(_func)
    {
        auto _handler = get_signal_handler().load(std::memory_order_relaxed);
        if(get_signal_handler().compare_exchange_strong(_handler, _func,
                                                        std::memory_order_relaxed))
        {
            return _handler;
        }
        else
        {
            _handler = get_signal_handler().load(std::memory_order_seq_cst);
            get_signal_handler().store(_func);
            return _handler;
        }
    }

    return get_signal_handler().load();
}

void
configure_signal_handler(const std::shared_ptr<settings>& _config)
{
    auto _ignore_dyninst_trampoline =
        tim::get_env("OMNITRACE_IGNORE_DYNINST_TRAMPOLINE", false);
    // this is how dyninst looks up the env variable
    static auto _dyninst_trampoline_signal =
        getenv("DYNINST_SIGNAL_TRAMPOLINE_SIGILL") ? SIGILL : SIGTRAP;

    if(_config->get_enable_signal_handler())
    {
        tim::signals::disable_signal_detection();
        signal_settings::enable(sys_signal::Interrupt);
        signal_settings::set_exit_action(omnitrace_exit_action);
        signal_settings::check_environment();
        auto default_signals = signal_settings::get_default();
        for(const auto& itr : default_signals)
            signal_settings::enable(itr);
        if(_ignore_dyninst_trampoline)
            signal_settings::disable(static_cast<sys_signal>(_dyninst_trampoline_signal));
        auto enabled_signals = signal_settings::get_enabled();
        tim::signals::enable_signal_detection(enabled_signals);
    }

    if(_ignore_dyninst_trampoline)
    {
        struct sigaction _action;
        sigemptyset(&_action.sa_mask);
        _action.sa_flags   = {};
        _action.sa_handler = omnitrace_trampoline_handler;
        sigaction(_dyninst_trampoline_signal, &_action, nullptr);
    }
}

bool
get_use_sampling_overflow()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_OVERFLOW");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_sampling_realtime()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_REALTIME");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_sampling_cputime()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_CPUTIME");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

std::set<int> get_sampling_signals(int64_t)
{
    auto _v = std::set<int>{};
    if(get_use_causal())
    {
        _v.emplace(get_sampling_cputime_signal());
        _v.emplace(get_sampling_realtime_signal());
    }
    else
    {
        if(get_use_sampling() && !get_use_sampling_cputime() &&
           !get_use_sampling_realtime() && !get_use_sampling_overflow())
        {
            OMNITRACE_VERBOSE_F(1, "sampling enabled by cputime/realtime/overflow not "
                                   "specified. defaulting to cputime...\n");
            set_setting_value("OMNITRACE_SAMPLING_CPUTIME", true);
        }

        if(get_use_sampling_cputime()) _v.emplace(get_sampling_cputime_signal());
        if(get_use_sampling_realtime()) _v.emplace(get_sampling_realtime_signal());
        if(get_use_sampling_overflow()) _v.emplace(get_sampling_overflow_signal());
    }

    return _v;
}

void
configure_disabled_settings(const std::shared_ptr<settings>& _config)
{
    auto _handle_use_option = [_config](const std::string& _opt,
                                        const std::string& _category) {
        if(!_config->get<bool>(_opt))
        {
            auto _disabled = _config->disable_category(_category);
            _config->enable(_opt);
            for(auto&& itr : _disabled)
                OMNITRACE_BASIC_VERBOSE(3, "[%s=OFF]    disabled option :: '%s'\n",
                                        _opt.c_str(), itr.c_str());
            return false;
        }
        auto _enabled = _config->enable_category(_category);
        for(auto&& itr : _enabled)
            OMNITRACE_BASIC_VERBOSE(3, "[%s=ON]      enabled option :: '%s'\n",
                                    _opt.c_str(), itr.c_str());
        return true;
    };

    _handle_use_option("OMNITRACE_USE_SAMPLING", "sampling");
    _handle_use_option("OMNITRACE_USE_PROCESS_SAMPLING", "process_sampling");
    _handle_use_option("OMNITRACE_USE_CAUSAL", "causal");
    _handle_use_option("OMNITRACE_USE_KOKKOSP", "kokkos");
    _handle_use_option("OMNITRACE_USE_TRACE", "perfetto");
    _handle_use_option("OMNITRACE_PROFILE", "timemory");
    _handle_use_option("OMNITRACE_USE_OMPT", "ompt");
    _handle_use_option("OMNITRACE_USE_RCCLP", "rcclp");
    _handle_use_option("OMNITRACE_USE_ROCM_SMI", "rocm_smi");
    _handle_use_option("OMNITRACE_USE_ROCTRACER", "roctracer");
    _handle_use_option("OMNITRACE_USE_ROCPROFILER", "rocprofiler");

#if !defined(OMNITRACE_USE_ROCTRACER) || OMNITRACE_USE_ROCTRACER == 0
    _config->find("OMNITRACE_USE_ROCTRACER")->second->set_hidden(true);
    for(const auto& itr : _config->disable_category("roctracer"))
        _config->find(itr)->second->set_hidden(true);
#endif

#if !defined(OMNITRACE_USE_ROCPROFILER) || OMNITRACE_USE_ROCPROFILER == 0
    _config->find("OMNITRACE_USE_ROCPROFILER")->second->set_hidden(true);
    for(const auto& itr : _config->disable_category("rocprofiler"))
        _config->find(itr)->second->set_hidden(true);
#endif

#if !defined(OMNITRACE_USE_ROCM_SMI) || OMNITRACE_USE_ROCM_SMI == 0
    _config->find("OMNITRACE_USE_ROCM_SMI")->second->set_hidden(true);
    for(const auto& itr : _config->disable_category("rocm_smi"))
        _config->find(itr)->second->set_hidden(true);
#endif

#if defined(OMNITRACE_USE_OMPT) || OMNITRACE_USE_OMPT == 0
    _config->find("OMNITRACE_USE_OMPT")->second->set_hidden(true);
    for(const auto& itr : _config->disable_category("ompt"))
        _config->find(itr)->second->set_hidden(true);
#endif

#if !defined(TIMEMORY_USE_MPI) || TIMEMORY_USE_MPI == 0
    _config->disable("OMNITRACE_PERFETTO_COMBINE_TRACES");
    _config->disable("OMNITRACE_COLLAPSE_PROCESSES");
    _config->find("OMNITRACE_PERFETTO_COMBINE_TRACES")->second->set_hidden(true);
    _config->find("OMNITRACE_COLLAPSE_PROCESSES")->second->set_hidden(true);
#endif

    _config->disable_category("throttle");

    // user bundle components
    _config->disable("components");
    _config->disable("global_components");
    _config->disable("ompt_components");
    _config->disable("kokkos_components");
    _config->disable("trace_components");
    _config->disable("profiler_components");

    // miscellaneous
    _config->disable("destructor_report");
    _config->disable("stack_clearing");
    _config->disable("add_secondary");

    // output fields
    _config->disable("auto_output");
    _config->disable("file_output");
    _config->disable("plot_output");
    _config->disable("dart_output");
    _config->disable("flamegraph_output");
    _config->disable("separator_freq");

    // exclude some timemory settings which are not relevant to omnitrace
    //  exact matches, e.g. OMNITRACE_BANNER
    std::string _hidden_exact_re =
        "^OMNITRACE_(BANNER|DESTRUCTOR_REPORT|COMPONENTS|(GLOBAL|MPIP|NCCLP|OMPT|"
        "PROFILER|TRACE|KOKKOS)_COMPONENTS|PYTHON_EXE|PAPI_ATTACH|PLOT_OUTPUT|SEPARATOR_"
        "FREQ|STACK_CLEARING|TARGET_PID|THROTTLE_(COUNT|VALUE)|(AUTO|FLAMEGRAPH)_OUTPUT|"
        "(ENABLE|DISABLE)_ALL_SIGNALS|ALLOW_SIGNAL_HANDLER|CTEST_NOTES|INSTRUCTION_"
        "ROOFLINE|ADD_SECONDARY|MAX_THREAD_BOOKMARKS)$";

    //  leading matches, e.g. OMNITRACE_MPI_[A-Z_]+
    std::string _hidden_begin_re =
        "^OMNITRACE_(ERT|DART|MPI|UPCXX|ROOFLINE|CUDA|NVTX|CUPTI)_[A-Z_]+$";

    auto _hidden_exact = std::set<std::string>{};

#if !defined(TIMEMORY_USE_CRAYPAT)
    _hidden_exact.emplace("OMNITRACE_CRAYPAT");
#endif

    for(const auto& itr : *_config)
    {
        auto _v = itr.second->get_env_name();
        if(_hidden_exact.count(_v) > 0 ||
           std ::regex_match(_v, std::regex{ _hidden_exact_re }) ||
           std::regex_match(_v, std::regex{ _hidden_begin_re }))
        {
            itr.second->set_enabled(false);
            itr.second->set_hidden(true);
        }
    }
}

void
handle_deprecated_setting(const std::string& _old, const std::string& _new, int _verbose)
{
    auto _config      = settings::shared_instance();
    auto _old_setting = _config->find(_old);
    auto _new_setting = _config->find(_new);

    if(_old_setting == _config->end()) return;

    OMNITRACE_CI_THROW(_new_setting == _config->end(),
                       "New configuration setting not found: '%s'", _new.c_str());

    if(_old_setting->second->get_environ_updated() ||
       _old_setting->second->get_config_updated())
    {
        auto _separator = [_verbose]() {
            std::array<char, 79> _v = {};
            _v.fill('=');
            _v.back() = '\0';
            OMNITRACE_BASIC_VERBOSE(_verbose, "#%s#\n", _v.data());
        };
        _separator();
        OMNITRACE_BASIC_VERBOSE(_verbose, "#\n");
        OMNITRACE_BASIC_VERBOSE(_verbose, "# DEPRECATION NOTICE:\n");
        OMNITRACE_BASIC_VERBOSE(_verbose, "#   %s is deprecated!\n", _old.c_str());
        OMNITRACE_BASIC_VERBOSE(_verbose, "#   Use %s instead!\n", _new.c_str());

        if(!_new_setting->second->get_environ_updated() &&
           !_new_setting->second->get_config_updated())
        {
            auto _before = _new_setting->second->as_string();
            _new_setting->second->parse(_old_setting->second->as_string());
            auto _after = _new_setting->second->as_string();

            if(_before != _after)
            {
                std::string _cause =
                    (_old_setting->second->get_environ_updated()) ? "environ" : "config";
                OMNITRACE_BASIC_VERBOSE(_verbose, "#\n");
                OMNITRACE_BASIC_VERBOSE(_verbose, "# %s :: '%s' -> '%s'\n", _new.c_str(),
                                        _before.c_str(), _after.c_str());
                OMNITRACE_BASIC_VERBOSE(_verbose, "#   via %s (%s)\n", _old.c_str(),
                                        _cause.c_str());
            }
        }

        OMNITRACE_BASIC_VERBOSE(_verbose, "#\n");
        _separator();
    }
}

void
print_banner(std::ostream& _os)
{
    static const char* _banner = R"banner(

      ______   .___  ___. .__   __.  __  .___________..______          ___       ______  _______
     /  __  \  |   \/   | |  \ |  | |  | |           ||   _  \        /   \     /      ||   ____|
    |  |  |  | |  \  /  | |   \|  | |  | `---|  |----`|  |_)  |      /  ^  \   |  ,----'|  |__
    |  |  |  | |  |\/|  | |  . `  | |  |     |  |     |      /      /  /_\  \  |  |     |   __|
    |  `--'  | |  |  |  | |  |\   | |  |     |  |     |  |\  \----./  _____  \ |  `----.|  |____
     \______/  |__|  |__| |__| \__| |__|     |__|     | _| `._____/__/     \__\ \______||_______|

    )banner";

    std::stringstream _version_info{};
    _version_info << "omnitrace v" << OMNITRACE_VERSION_STRING;

    namespace join = ::timemory::join;

    // assemble the list of properties
    auto _generate_properties =
        [](std::initializer_list<std::pair<std::string, std::string>>&& _data) {
            auto _property_info = std::vector<std::string>{};
            _property_info.reserve(_data.size());
            for(const auto& itr : _data)
            {
                if(!itr.second.empty())
                    _property_info.emplace_back(
                        itr.first.empty() ? itr.second
                                          : join::join(": ", itr.first, itr.second));
            }
            return _property_info;
        };

    auto _properties =
        _generate_properties({ { "rev", OMNITRACE_GIT_REVISION },
                               { "tag", OMNITRACE_GIT_DESCRIBE },
                               { "", OMNITRACE_LIBRARY_ARCH },
                               { "compiler", OMNITRACE_COMPILER_STRING },
                               { "rocm", OMNITRACE_HIP_VERSION_COMPAT_STRING } });

    // <NAME> <VERSION> (<PROPERTIES>)
    if(!_properties.empty())
        _version_info << join::join(join::array_config{ ", ", " (", ")" }, _properties);

    tim::log::stream(_os, tim::log::color::info()) << _banner << _version_info.str();
    _os << std::endl;
}

void
print_settings(
    std::ostream&                                                                _ros,
    std::function<bool(const std::string_view&, const std::set<std::string>&)>&& _filter)
{
    OMNITRACE_CONDITIONAL_BASIC_PRINT(true, "configuration:\n");

    std::stringstream _os{};

    bool _print_desc = get_debug() || tim::get_env("OMNITRACE_SETTINGS_DESC", false);
    bool _md         = tim::get_env<bool>("OMNITRACE_SETTINGS_DESC_MARKDOWN", false);

    constexpr size_t nfields = 3;
    using str_array_t        = std::array<std::string, nfields>;
    std::vector<str_array_t>    _data{};
    std::array<size_t, nfields> _widths{};
    _widths.fill(0);
    for(const auto& itr : *get_config())
    {
        if(itr.second->get_hidden()) continue;
        if(!itr.second->get_enabled()) continue;
        if(_filter(itr.first, itr.second->get_categories()))
        {
            auto _disp = itr.second->get_display(std::ios::boolalpha);
            _data.emplace_back(str_array_t{ _disp.at("env_name"), _disp.at("value"),
                                            _disp.at("description") });
            for(size_t i = 0; i < nfields; ++i)
            {
                size_t _wextra = (_md && i < 2) ? 2 : 0;
                _widths.at(i)  = std::max<size_t>(_widths.at(i),
                                                 _data.back().at(i).length() + _wextra);
            }
        }
    }

    std::sort(_data.begin(), _data.end(), [](const auto& lhs, const auto& rhs) {
        auto _npos = std::string::npos;
        // OMNITRACE_CONFIG_FILE always first
        if(lhs.at(0) == "OMNITRACE_MODE") return true;
        if(rhs.at(0) == "OMNITRACE_MODE") return false;
        // OMNITRACE_CONFIG_FILE always second
        if(lhs.at(0).find("OMNITRACE_CONFIG") != _npos) return true;
        if(rhs.at(0).find("OMNITRACE_CONFIG") != _npos) return false;
        // OMNITRACE_USE_* prioritized
        auto _lhs_use = lhs.at(0).find("OMNITRACE_USE_");
        auto _rhs_use = rhs.at(0).find("OMNITRACE_USE_");
        if(_lhs_use != _rhs_use && _lhs_use < _rhs_use) return true;
        if(_lhs_use != _rhs_use && _lhs_use > _rhs_use) return false;
        // alphabetical sort
        return lhs.at(0) < rhs.at(0);
    });

    auto tot_width = std::accumulate(_widths.begin(), _widths.end(), 0);
    if(!_print_desc) tot_width -= _widths.back() + 4;

    size_t _spacer_extra = 9;
    if(!_md)
        _spacer_extra += 2;
    else if(_md && _print_desc)
        _spacer_extra -= 1;
    std::stringstream _spacer{};
    _spacer.fill('-');
    _spacer << "#" << std::setw(tot_width + _spacer_extra) << ""
            << "#";
    _os << _spacer.str() << "\n";
    for(const auto& itr : _data)
    {
        _os << ((_md) ? "| " : "# ");
        for(size_t i = 0; i < nfields; ++i)
        {
            switch(i)
            {
                case 0: _os << std::left; break;
                case 1: _os << std::left; break;
                case 2: _os << std::left; break;
            }
            if(_md)
            {
                std::stringstream _ss{};
                _ss.setf(_os.flags());
                std::string _extra = (i < 2) ? "`" : "";
                _ss << _extra << itr.at(i) << _extra;
                _os << std::setw(_widths.at(i)) << _ss.str() << " | ";
                if(!_print_desc && i == 1) break;
            }
            else
            {
                _os << std::setw(_widths.at(i)) << itr.at(i) << " ";
                if(!_print_desc && i == 1) break;
                switch(i)
                {
                    case 0: _os << "= "; break;
                    case 1: _os << "[ "; break;
                    case 2: _os << "]"; break;
                }
            }
        }
        _os << ((_md) ? "\n" : "  #\n");
    }

    _os << _spacer.str() << "\n";

    tim::log::stream(_ros, tim::log::color::info()) << _os.str();
    _ros << std::flush;
}

void
print_settings(bool _include_env)
{
    if(dmp::rank() > 0) return;

    // generic filter for filtering relevant options
    auto _is_omnitrace_option = [](const auto& _v, const auto&) {
        return (_v.find("OMNITRACE_") == 0);
    };

    if(_include_env)
    {
        std::cerr << tim::log::info;
        tim::print_env(std::cerr, [_is_omnitrace_option](const std::string& _v) {
            auto _is_omni_opt = _is_omnitrace_option(_v, std::set<std::string>{});
            if(settings::verbose() >= 2 || settings::debug()) return _is_omni_opt;
            return (_is_omni_opt && _v.find("OMNITRACE_SIGNAL_") != 0);
        });
        std::cerr << tim::log::flush;
    }

    print_settings(std::cerr, _is_omnitrace_option);

    fprintf(stderr, "\n");
}

std::string&
get_exe_name()
{
    static std::string _v = {};
    return _v;
}

std::string&
get_exe_realpath()
{
    static std::string _v = []() {
        auto _cmd_line = tim::read_command_line(process::get_id());
        if(!_cmd_line.empty())
            return filepath::realpath(_cmd_line.front(), nullptr, false);
        return std::string{};
    }();
    return _v;
}

std::string
get_config_file()
{
    static auto _v = get_config()->find("OMNITRACE_CONFIG_FILE");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

Mode
get_mode()
{
    if(!settings_are_configured())
    {
        auto _mode = tim::get_env_choice<std::string>(
            "OMNITRACE_MODE", "trace", { "trace", "sampling", "causal", "coverage" });
        if(_mode == "sampling")
            return Mode::Sampling;
        else if(_mode == "causal")
            return Mode::Causal;
        else if(_mode == "coverage")
            return Mode::Coverage;
        return Mode::Trace;
    }
    static auto _m =
        std::unordered_map<std::string_view, Mode>{ { "trace", Mode::Trace },
                                                    { "causal", Mode::Causal },
                                                    { "sampling", Mode::Sampling },
                                                    { "coverage", Mode::Coverage } };
    static auto _v = get_config()->find("OMNITRACE_MODE");
    try
    {
        return _m.at(static_cast<tim::tsettings<std::string>&>(*_v->second).get());
    } catch(std::runtime_error& _e)
    {
        auto _mode = static_cast<tim::tsettings<std::string>&>(*_v->second).get();
        std::stringstream _ss{};
        for(const auto& itr : _v->second->get_choices())
            _ss << ", " << itr;
        auto _msg = (_ss.str().length() > 2) ? _ss.str().substr(2) : std::string{};
        OMNITRACE_THROW("[%s] invalid mode %s. Choices: %s\n", __FUNCTION__,
                        _mode.c_str(), _msg.c_str());
    }
    return Mode::Trace;
}

bool&
is_attached()
{
    static bool _v = false;
    return _v;
}

bool&
is_binary_rewrite()
{
    static bool _v = false;
    return _v;
}

bool
get_debug_env()
{
    return (settings_are_configured())
               ? get_debug()
               : tim::get_env<bool>("OMNITRACE_DEBUG", false, false);
}

bool
get_is_continuous_integration()
{
    if(!settings_are_configured())
        return tim::get_env<bool>("OMNITRACE_CI", false, false);
    static auto _v = get_config()->find("OMNITRACE_CI");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_debug_init()
{
    return tim::get_env<bool>("OMNITRACE_DEBUG_INIT", get_debug_env());
}

bool
get_debug_finalize()
{
    return tim::get_env<bool>("OMNITRACE_DEBUG_FINALIZE", false);
}

bool
get_debug()
{
    static auto _v = get_config()->find("OMNITRACE_DEBUG");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_debug_sampling()
{
    static bool _v =
        tim::get_env<bool>("OMNITRACE_DEBUG_SAMPLING",
                           (settings_are_configured() ? get_debug() : get_debug_env()));
    return _v;
}

int
get_verbose_env()
{
    return (settings_are_configured()) ? get_verbose()
                                       : tim::get_env<int>("OMNITRACE_VERBOSE", 0, false);
}

int
get_verbose()
{
    static auto _v = get_config()->find("OMNITRACE_VERBOSE");
    return static_cast<tim::tsettings<int>&>(*_v->second).get();
}

bool&
get_use_perfetto()
{
    static auto _v = get_config()->find("OMNITRACE_TRACE");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool&
get_use_timemory()
{
    static auto _v = get_config()->find("OMNITRACE_PROFILE");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool&
get_use_causal()
{
    static auto _v = get_config()->find("OMNITRACE_USE_CAUSAL");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_roctracer()
{
#if defined(OMNITRACE_USE_ROCTRACER) && OMNITRACE_USE_ROCTRACER > 0
    static auto _v = get_config()->find("OMNITRACE_USE_ROCTRACER");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool
get_perfetto_roctracer_per_stream()
{
#if defined(OMNITRACE_USE_ROCTRACER) && OMNITRACE_USE_ROCTRACER > 0
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_ROCTRACER_PER_STREAM");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool
get_use_rocprofiler()
{
#if defined(OMNITRACE_USE_ROCPROFILER) && OMNITRACE_USE_ROCPROFILER > 0
    static auto _v = get_config()->find("OMNITRACE_USE_ROCPROFILER");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool
get_use_rocm_smi()
{
#if defined(OMNITRACE_USE_ROCM_SMI) && OMNITRACE_USE_ROCM_SMI > 0
    static auto _v = get_config()->find("OMNITRACE_USE_ROCM_SMI");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool
get_use_roctx()
{
#if defined(OMNITRACE_USE_ROCTRACER) && OMNITRACE_USE_ROCTRACER > 0
    static auto _v = get_config()->find("OMNITRACE_USE_ROCTX");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool&
get_use_sampling()
{
#if defined(TIMEMORY_USE_LIBUNWIND)
    static auto _v = get_config()->find("OMNITRACE_USE_SAMPLING");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    OMNITRACE_THROW(
        "Error! sampling was enabled but omnitrace was not built with libunwind support");
    static bool _v = false;
    return _v;
#endif
}

bool&
get_use_process_sampling()
{
    static auto _v = get_config()->find("OMNITRACE_USE_PROCESS_SAMPLING");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool&
get_use_pid()
{
    static auto _v = get_config()->find("OMNITRACE_USE_PID");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool&
get_use_mpip()
{
    static auto _v = get_config()->find("OMNITRACE_USE_MPIP");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_kokkosp()
{
    static auto _v = get_config()->find("OMNITRACE_USE_KOKKOSP");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_kokkosp_kernel_logger()
{
    static auto _v = get_config()->find("OMNITRACE_KOKKOSP_KERNEL_LOGGER");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_ompt()
{
#if defined(TIMEMORY_USE_OMPT)
    static auto _v = get_config()->find("OMNITRACE_USE_OMPT");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool
get_use_code_coverage()
{
    static auto _v = get_config()->find("OMNITRACE_USE_CODE_COVERAGE");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_rcclp()
{
    static auto _v = get_config()->find("OMNITRACE_USE_RCCLP");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

size_t
get_num_threads_hint()
{
    static auto _v = get_config()->find("OMNITRACE_NUM_THREADS_HINT");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

bool
get_sampling_keep_internal()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_KEEP_INTERNAL");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

int
get_sampling_overflow_signal()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_OVERFLOW_SIGNAL");
    return static_cast<tim::tsettings<int>&>(*_v->second).get();
}

int
get_sampling_realtime_signal()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_REALTIME_SIGNAL");
    return static_cast<tim::tsettings<int>&>(*_v->second).get();
}

int
get_sampling_cputime_signal()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_CPUTIME_SIGNAL");
    return static_cast<tim::tsettings<int>&>(*_v->second).get();
}

bool
get_trace_hip_api()
{
    static auto _v = get_config()->find("OMNITRACE_ROCTRACER_HIP_API");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_trace_hip_activity()
{
    static auto _v = get_config()->find("OMNITRACE_ROCTRACER_HIP_ACTIVITY");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_trace_hsa_api()
{
    static auto _v = get_config()->find("OMNITRACE_ROCTRACER_HSA_API");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_trace_hsa_activity()
{
    static auto _v = get_config()->find("OMNITRACE_ROCTRACER_HSA_ACTIVITY");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

size_t
get_perfetto_shmem_size_hint()
{
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_SHMEM_SIZE_HINT_KB");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

size_t
get_perfetto_buffer_size()
{
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_BUFFER_SIZE_KB");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

bool
get_perfetto_combined_traces()
{
#if defined(TIMEMORY_USE_MPI) && TIMEMORY_USE_MPI > 0
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_COMBINE_TRACES");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

std::string
get_perfetto_fill_policy()
{
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_FILL_POLICY");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

namespace
{
auto
get_category_config()
{
    using strset_t = std::set<std::string>;

    static auto _v = []() {
        auto _avail = get_available_categories<strset_t>();
        auto _parse = [&_avail](const auto& _setting) {
            auto _ret = strset_t{};
            for(auto itr : tim::delimit(
                    static_cast<tim::tsettings<std::string>&>(*_setting->second).get(),
                    " ,;:\n\t"))
            {
                if(_avail.count(itr) > 0) _ret.emplace(itr);
            }
            return _ret;
        };

        auto _enabled  = _parse(get_config()->find("OMNITRACE_ENABLE_CATEGORIES"));
        auto _disabled = _parse(get_config()->find("OMNITRACE_DISABLE_CATEGORIES"));

        if(_enabled.empty() && _disabled.empty())
        {
            _enabled = _avail;
        }
        else if(_enabled.empty() && !_disabled.empty())
        {
            for(auto itr : _avail)
            {
                if(_disabled.count(itr) == 0) _enabled.emplace(itr);
            }
        }
        else if(!_enabled.empty() && _disabled.empty())
        {
            for(auto itr : _avail)
            {
                if(_enabled.count(itr) == 0) _disabled.emplace(itr);
            }
        }
        else
        {
            OMNITRACE_ABORT("Error! Conflicting options OMNITRACE_ENABLE_CATEGORIES and "
                            "OMNITRACE_DISABLE_CATEGORIES were both provided.");
        }

        OMNITRACE_CI_THROW(_enabled.size() + _disabled.size() != _avail.size(),
                           "Error! Internal error for categories: %zu (enabled) + %zu "
                           "(disabled) != %zu (total)\n",
                           _enabled.size(), _disabled.size(), _avail.size());

        return std::make_pair(_enabled, _disabled);
    }();

    return _v;
}
}  // namespace
std::set<std::string>
get_enabled_categories()
{
    return get_category_config().first;
}

std::set<std::string>
get_disabled_categories()
{
    return get_category_config().second;
}

bool
get_perfetto_annotations()
{
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_ANNOTATIONS");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

uint64_t
get_thread_pool_size()
{
    static uint64_t _v = get_config()->get<uint64_t>("OMNITRACE_THREAD_POOL_SIZE");
    return _v;
}

std::string
get_trace_hsa_api_types()
{
    static std::string _v =
        get_config()->get<std::string>("OMNITRACE_ROCTRACER_HSA_API_TYPES");
    return _v;
}

std::string&
get_perfetto_backend()
{
    // select inprocess, system, or both (i.e. all)
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_BACKEND");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

std::string
get_perfetto_output_filename()
{
    static auto _v       = get_config()->find("OMNITRACE_PERFETTO_FILE");
    auto        _val     = static_cast<tim::tsettings<std::string>&>(*_v->second).get();
    auto        _pos_dir = _val.find_last_of('/');
    auto        _dir     = std::string{};
    auto        _ext     = std::string{ "proto" };
    if(_pos_dir != std::string::npos)
    {
        _dir = _val.substr(0, _pos_dir + 1);
        _val = _val.substr(_pos_dir + 1);
    }
    auto _pos_ext = _val.find_last_of('.');
    if(_pos_ext + 1 < _val.length())
    {
        _ext = _val.substr(_pos_ext + 1);
        _val = _val.substr(0, _pos_ext);
    }

    auto _cfg = settings::compose_filename_config{ settings::use_output_suffix(),
                                                   settings::default_process_suffix(),
                                                   false, _dir };
    _val      = settings::compose_output_filename(_val, _ext, _cfg);
    if(!_val.empty() && _val.at(0) != '/')
        return settings::format(JOIN('/', "%env{PWD}%", _val), get_config()->get_tag());
    return _val;
}

double
get_sampling_freq()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_FREQ");
    return static_cast<tim::tsettings<double>&>(*_v->second).get();
}

double
get_sampling_cputime_freq()
{
    static auto _v   = get_config()->find("OMNITRACE_SAMPLING_CPUTIME_FREQ");
    auto&       _val = static_cast<tim::tsettings<double>&>(*_v->second).get();
    if(_val <= 0.0) _val = get_sampling_freq();
    return _val;
}

double
get_sampling_realtime_freq()
{
    static auto _v   = get_config()->find("OMNITRACE_SAMPLING_REALTIME_FREQ");
    auto&       _val = static_cast<tim::tsettings<double>&>(*_v->second).get();
    if(_val <= 0.0) _val = get_sampling_freq();
    return _val;
}

double
get_sampling_overflow_freq()
{
    static auto _v   = get_config()->find("OMNITRACE_SAMPLING_OVERFLOW_FREQ");
    auto&       _val = static_cast<tim::tsettings<double>&>(*_v->second).get();
    if(_val <= 0.0) _val = get_sampling_freq();
    return _val;
}

double
get_sampling_delay()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_DELAY");
    return static_cast<tim::tsettings<double>&>(*_v->second).get();
}

double
get_sampling_cputime_delay()
{
    static auto _v   = get_config()->find("OMNITRACE_SAMPLING_CPUTIME_DELAY");
    auto&       _val = static_cast<tim::tsettings<double>&>(*_v->second).get();
    if(_val <= 0.0) _val = get_sampling_delay();
    return _val;
}

double
get_sampling_realtime_delay()
{
    static auto _v   = get_config()->find("OMNITRACE_SAMPLING_REALTIME_DELAY");
    auto&       _val = static_cast<tim::tsettings<double>&>(*_v->second).get();
    if(_val <= 0.0) _val = get_sampling_delay();
    return _val;
}

double
get_sampling_duration()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_DURATION");
    return static_cast<tim::tsettings<double>&>(*_v->second).get();
}

std::string
get_sampling_cpus()
{
    auto _v = get_config()->find("OMNITRACE_SAMPLING_CPUS");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

std::set<int64_t>
get_sampling_tids()
{
    auto _v = get_config()->find("OMNITRACE_SAMPLING_TIDS");
    return parse_numeric_range<>(
        static_cast<tim::tsettings<std::string>&>(*_v->second).get(), "thread IDs", 1L);
}

std::set<int64_t>
get_sampling_cputime_tids()
{
    auto _v = get_config()->find("OMNITRACE_SAMPLING_CPUTIME_TIDS");
    return parse_numeric_range<>(
        static_cast<tim::tsettings<std::string>&>(*_v->second).get(), "thread IDs", 1L);
}

std::set<int64_t>
get_sampling_realtime_tids()
{
    auto _v = get_config()->find("OMNITRACE_SAMPLING_REALTIME_TIDS");
    return parse_numeric_range<>(
        static_cast<tim::tsettings<std::string>&>(*_v->second).get(), "thread IDs", 1L);
}

std::set<int64_t>
get_sampling_overflow_tids()
{
    auto _v = get_config()->find("OMNITRACE_SAMPLING_OVERFLOW_TIDS");
    return parse_numeric_range<>(
        static_cast<tim::tsettings<std::string>&>(*_v->second).get(), "thread IDs", 1L);
}

bool
get_sampling_include_inlines()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_INCLUDE_INLINES");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

size_t
get_sampling_allocator_size()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_ALLOCATOR_SIZE");
    return std::max<size_t>(static_cast<tim::tsettings<size_t>&>(*_v->second).get(), 1);
}

double
get_process_sampling_freq()
{
    static auto _v = get_config()->find("OMNITRACE_PROCESS_SAMPLING_FREQ");
    auto        _val =
        std::min<double>(static_cast<tim::tsettings<double>&>(*_v->second).get(), 1000.0);
    if(_val < 1.0e-9) return std::min<double>(get_sampling_freq(), 100.0);
    return _val;
}

double
get_process_sampling_duration()
{
    static auto _v = get_config()->find("OMNITRACE_PROCESS_SAMPLING_DURATION");
    return static_cast<tim::tsettings<double>&>(*_v->second).get();
}

std::string
get_sampling_gpus()
{
#if defined(OMNITRACE_USE_ROCM_SMI) && OMNITRACE_USE_ROCM_SMI > 0
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_GPUS");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
#else
    return std::string{};
#endif
}

bool
get_trace_thread_locks()
{
    static auto _v = get_config()->find("OMNITRACE_TRACE_THREAD_LOCKS");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

std::string
get_rocm_events()
{
    static auto _v = get_config()->find("OMNITRACE_ROCM_EVENTS");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

bool
get_trace_thread_rwlocks()
{
    static auto _v = get_config()->find("OMNITRACE_TRACE_THREAD_RW_LOCKS");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_trace_thread_spin_locks()
{
    static auto _v = get_config()->find("OMNITRACE_TRACE_THREAD_SPIN_LOCKS");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_trace_thread_barriers()
{
    static auto _v = get_config()->find("OMNITRACE_TRACE_THREAD_BARRIERS");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_trace_thread_join()
{
    static auto _v = get_config()->find("OMNITRACE_TRACE_THREAD_JOIN");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_debug_tid()
{
    static auto _vlist = parse_numeric_range<int64_t, std::unordered_set<int64_t>>(
        tim::get_env<std::string>("OMNITRACE_DEBUG_TIDS", ""), "debug tids", 1L);
    static thread_local bool _v =
        _vlist.empty() || _vlist.count(tim::threading::get_id()) > 0;
    return _v;
}

bool
get_debug_pid()
{
    static auto _vlist = parse_numeric_range<int64_t, std::unordered_set<int64_t>>(
        tim::get_env<std::string>("OMNITRACE_DEBUG_PIDS", ""), "debug pids", 1L);
    static bool _v = _vlist.empty() || _vlist.count(tim::process::get_id()) > 0 ||
                     _vlist.count(dmp::rank()) > 0;
    return _v;
}

bool
get_use_tmp_files()
{
    static auto _v = get_config()->find("OMNITRACE_USE_TEMPORARY_FILES");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

std::string
get_tmpdir()
{
    static auto _v = get_config()->find("OMNITRACE_TMPDIR");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

tmp_file::tmp_file(std::string _v)
: filename{ std::move(_v) }
{}

tmp_file::~tmp_file()
{
    close();
    remove();
}

bool
tmp_file::open(std::ios::openmode _mode)
{
    OMNITRACE_BASIC_VERBOSE(2, "Opening temporary file '%s'...\n", filename.c_str());

    if(!filepath::exists(filename))
    {
        // if the filepath does not exist, open in out mode to create it
        std::ofstream _ofs{};
        filepath::open(_ofs, filename);
    }

    stream.open(filename, _mode);

    return (stream.is_open() && stream.good());
}

bool
tmp_file::fopen(const char* _mode)
{
    OMNITRACE_BASIC_VERBOSE(2, "Opening temporary file '%s'...\n", filename.c_str());

    if(!filepath::exists(filename))
    {
        // if the filepath does not exist, open in out mode to create it
        std::ofstream _ofs{};
        filepath::open(_ofs, filename);
    }

    file = filepath::fopen(filename, _mode);
    if(file) fd = ::fileno(file);

    return (file != nullptr && fd > 0);
}

bool
tmp_file::flush()
{
    if(stream.is_open())
    {
        stream.flush();
    }
    else if(file != nullptr)
    {
        int _ret = fflush(file);
        int _cnt = 0;
        while(_ret == EAGAIN || _ret == EINTR)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
            _ret = fflush(file);
            if(++_cnt > 10) break;
        }
        return (_ret == 0);
    }

    return true;
}

bool
tmp_file::close()
{
    flush();

    if(stream.is_open())
    {
        stream.close();
        return !stream.is_open();
    }
    else if(file != nullptr)
    {
        auto _ret = fclose(file);
        if(_ret == 0)
        {
            file = nullptr;
            fd   = -1;
        }
        return (_ret == 0);
    }

    return true;
}

bool
tmp_file::remove()
{
    close();
    if(filepath::exists(filename))
    {
        OMNITRACE_BASIC_VERBOSE(2, "Removing temporary file '%s'...\n", filename.c_str());
        auto _ret = ::remove(filename.c_str());
        return (_ret == 0);
    }

    return true;
}

tmp_file::operator bool() const
{
    return (stream.is_open() && stream.good()) || (file != nullptr && fd > 0);
}

std::shared_ptr<tmp_file>
get_tmp_file(std::string _basename, std::string _ext)
{
    if(!get_use_tmp_files()) return std::shared_ptr<tmp_file>{};

    static auto _existing_files =
        std::unordered_map<std::string, std::shared_ptr<tmp_file>>{};
    static std::mutex            _mutex{};
    std::unique_lock<std::mutex> _lk{ _mutex };

    cfg_fini_callbacks.emplace_back([]() {
        for(auto itr : _existing_files)
        {
            if(itr.second)
            {
                itr.second->close();
                itr.second->remove();
                itr.second.reset();
            }
        }
        _existing_files.clear();
    });

    auto _cfg          = settings::compose_filename_config{};
    _cfg.use_suffix    = true;
    _cfg.suffix        = "%pid%";
    _cfg.explicit_path = get_tmpdir();
    _cfg.subdirectory  = JOIN('/', settings::output_path(), "%ppid%", "");
    auto _fname =
        settings::compose_output_filename(std::move(_basename), std::move(_ext), _cfg);

    if(_fname.empty() || _fname.front() != '/')
    {
        OMNITRACE_THROW("Error! temporary file '%s' (based on '%s.%s') is either empty "
                        "or is not an absolute path",
                        _fname.c_str(), _basename.c_str(), _ext.c_str());
    }
    auto itr = _existing_files.find(_fname);
    if(itr != _existing_files.end()) return itr->second;

    auto _v = std::make_shared<tmp_file>(_fname);
    _existing_files.emplace(_fname, std::move(_v));
    return _existing_files.at(_fname);
}

CausalBackend
get_causal_backend()
{
    static auto _m = std::unordered_map<std::string_view, CausalBackend>{
        { "auto", CausalBackend::Auto },
        { "perf", CausalBackend::Perf },
        { "timer", CausalBackend::Timer },
    };

    auto _v = get_config()->find("OMNITRACE_CAUSAL_BACKEND");
    try
    {
        return _m.at(static_cast<tim::tsettings<std::string>&>(*_v->second).get());
    } catch(std::runtime_error& _e)
    {
        auto _mode = static_cast<tim::tsettings<std::string>&>(*_v->second).get();
        OMNITRACE_THROW("[%s] invalid causal backend %s. Choices: %s\n", __FUNCTION__,
                        _mode.c_str(),
                        timemory::join::join(timemory::join::array_config{ ", ", "", "" },
                                             _v->second->get_choices())
                            .c_str());
    }
    return CausalBackend::Auto;
}

CausalMode
get_causal_mode()
{
    if(!settings_are_configured())
    {
        auto _mode = tim::get_env_choice<std::string>("OMNITRACE_CAUSAL_MODE", "function",
                                                      { "line", "function" });
        if(_mode == "line") return CausalMode::Line;
        return CausalMode::Function;
    }
    static auto _causal_mode = []() {
        auto _m = std::unordered_map<std::string_view, CausalMode>{
            { "line", CausalMode::Line },
            { "func", CausalMode::Function },
            { "function", CausalMode::Function }
        };
        auto _v = get_config()->find("OMNITRACE_CAUSAL_MODE");
        try
        {
            return _m.at(static_cast<tim::tsettings<std::string>&>(*_v->second).get());
        } catch(std::runtime_error& _e)
        {
            auto _mode = static_cast<tim::tsettings<std::string>&>(*_v->second).get();
            OMNITRACE_THROW(
                "[%s] invalid causal mode %s. Choices: %s\n", __FUNCTION__, _mode.c_str(),
                timemory::join::join(timemory::join::array_config{ ", ", "", "" },
                                     _v->second->get_choices())
                    .c_str());
        }
        return CausalMode::Function;
    }();
    return _causal_mode;
}

bool
get_causal_end_to_end()
{
    static auto _v = get_config()->find("OMNITRACE_CAUSAL_END_TO_END");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

std::vector<int64_t>
get_causal_fixed_speedup()
{
    static auto _v = get_config()->find("OMNITRACE_CAUSAL_FIXED_SPEEDUP");
    return parse_numeric_range<int64_t, std::vector<int64_t>>(
        static_cast<tim::tsettings<std::string>&>(*_v->second).get(),
        "causal fixed speedup", 5L);
}

std::string
get_causal_output_filename()
{
    static auto _v     = get_config()->find("OMNITRACE_CAUSAL_FILE");
    auto        _fname = static_cast<tim::tsettings<std::string>&>(*_v->second).get();
    for(auto&& itr : std::initializer_list<std::string>{ ".txt", ".json", ".xml" })
    {
        auto _pos = _fname.find(itr);
        // if extension is found at end of string, remove
        if(_pos != std::string::npos && (_pos + itr.length()) == _fname.length())
            _fname = _fname.substr(0, _fname.length() - itr.length());
    }
    return _fname;
}

namespace
{
std::vector<std::string>
format_causal_scopes(std::vector<std::string> _value, const std::string& _tag)
{
    const auto _config   = get_config();
    const auto _main_re  = std::regex{ "(^|[^a-zA-Z])(MAIN|%MAIN%)($|[^a-zA-Z])" };
    const auto _space_re = std::regex{ "^([ ]*)(.*)([ ]*)$" };
    for(auto& itr : _value)
    {
        // replace any output/input keys, e.g. %argv0%
        itr = settings::format(itr, _tag);
        // replace MAIN or %MAIN% with (<exe_basename>|<exe_realpath>)
        if(std::regex_search(itr, _main_re))
        {
            itr = std::regex_replace(
                itr, _main_re,
                join("", "$1", "(", get_exe_name(), "|", get_exe_realpath(), ")", "$3"));
        }
        // trim leading and trailing spaces since we didn't delimit spaces
        if(std::regex_search(itr, _space_re))
            itr = std::regex_replace(itr, _space_re, "$2");
    }
    return _value;
}
}  // namespace

std::vector<std::string>
get_causal_binary_scope()
{
    auto&&      _config = get_config();
    static auto _v      = _config->find("OMNITRACE_CAUSAL_BINARY_SCOPE");
    return format_causal_scopes(
        tim::delimit(static_cast<tim::tsettings<std::string>&>(*_v->second).get(),
                     "\t\"';"),
        _config->get_tag());
}

std::vector<std::string>
get_causal_source_scope()
{
    static auto _v = get_config()->find("OMNITRACE_CAUSAL_SOURCE_SCOPE");
    return tim::delimit(static_cast<tim::tsettings<std::string>&>(*_v->second).get(),
                        "\t\"';");
}

std::vector<std::string>
get_causal_function_scope()
{
    static auto _v = get_config()->find("OMNITRACE_CAUSAL_FUNCTION_SCOPE");
    return tim::delimit(static_cast<tim::tsettings<std::string>&>(*_v->second).get(),
                        "\t\"';");
}

std::vector<std::string>
get_causal_binary_exclude()
{
    auto&&      _config = get_config();
    static auto _v      = _config->find("OMNITRACE_CAUSAL_BINARY_EXCLUDE");
    return format_causal_scopes(
        tim::delimit(static_cast<tim::tsettings<std::string>&>(*_v->second).get(),
                     "\t\"';"),
        _config->get_tag());
}

std::vector<std::string>
get_causal_source_exclude()
{
    static auto _v = get_config()->find("OMNITRACE_CAUSAL_SOURCE_EXCLUDE");
    return tim::delimit(static_cast<tim::tsettings<std::string>&>(*_v->second).get(),
                        "\t\"';");
}

std::vector<std::string>
get_causal_function_exclude()
{
    static auto _v = get_config()->find("OMNITRACE_CAUSAL_FUNCTION_EXCLUDE");
    return tim::delimit(static_cast<tim::tsettings<std::string>&>(*_v->second).get(),
                        "\t\"';");
}
}  // namespace config
}  // namespace omnitrace
