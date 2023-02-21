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

#include "api.hpp"

#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/perfetto.hpp"
#include "core/perfetto_fwd.hpp"
#include "core/state.hpp"
#include "library/components/fork_gotcha.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"

#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>

#include <unistd.h>

namespace omnitrace
{
namespace component
{
void
fork_gotcha::configure()
{
    fork_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(fork_gotcha_t, 0, fork);
    };
}

void
fork_gotcha::audit(const gotcha_data_t&, audit::incoming)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    tim::set_env("OMNITRACE_PRELOAD", "0", 1);
    tim::set_env("OMNITRACE_ROOT_PROCESS", process::get_id(), 0);
    omnitrace_reset_preload_hidden();
    OMNITRACE_BASIC_VERBOSE(0, "fork() called on PID %i (rank: %i), TID %li\n",
                            process::get_id(), dmp::rank(), threading::get_id());
    OMNITRACE_BASIC_DEBUG(
        "Warning! Calling fork() within an OpenMPI application using libfabric "
        "may result is segmentation fault\n");
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(get_debug_env(), 16);

    if(config::get_use_perfetto())
    {
        OMNITRACE_BASIC_VERBOSE(1, "Stopping perfetto tracing...\n");
        omnitrace::perfetto::stop();
    }
}

void
fork_gotcha::audit(const gotcha_data_t&, audit::outgoing, pid_t _pid)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    if(_pid != 0)
    {
        OMNITRACE_BASIC_VERBOSE(0, "fork() called on PID %i created PID %i\n", getppid(),
                                _pid);

        if(config::get_use_perfetto())
        {
            OMNITRACE_BASIC_VERBOSE(1, "Resuming perfetto tracing...\n");
            omnitrace::perfetto::start();
        }
    }
    else
    {
        OMNITRACE_REQUIRE(is_child_process())
            << "Error! child process " << process::get_id()
            << " believes it is the root process " << get_root_process_id() << "\n";

        settings::enabled() = false;
        settings::verbose() = -127;
        settings::debug()   = false;
        omnitrace::sampling::shutdown();
        omnitrace::categories::shutdown();
        omnitrace::get_perfetto_session().reset();
        set_thread_state(::omnitrace::ThreadState::Disabled);
    }

    if(!settings::use_output_suffix())
    {
        settings::use_output_suffix()      = true;
        settings::default_process_suffix() = process::get_id();

        OMNITRACE_BASIC_VERBOSE(
            0, "call to fork() enables using an output suffix. PID %i will use %i\n",
            process::get_id(), process::get_id());
    }
}
}  // namespace component
}  // namespace omnitrace
