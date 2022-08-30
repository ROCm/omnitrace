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

#include "library/causal/experiment.hpp"
#include "library/causal/data.hpp"
#include "library/causal/progress_point.hpp"
#include "library/config.hpp"

#include <timemory/components/timing/backends.hpp>

namespace omnitrace
{
namespace causal
{
std::string
experiment::label()
{
    return "casual_experiment";
}

std::string
experiment::description()
{
    return "Records an experiment for casual profiling";
}

void
experiment::start()
{
    // sampling period in nanoseconds
    auto _sampling_period = (1.0 / config::get_sampling_freq()) * units::sec;
    active                = sample_enabled();
    virtual_speedup       = (active) ? sample_virtual_speedup() : 0;
    selection             = sample_progress_stack();
    start_time            = tim::get_clock_real_now<uint64_t, std::nano>();
    duration              = 100 * static_cast<uint64_t>(_sampling_period);

    progress_point::setup();
}

void
experiment::stop()
{
    if(tim::get_clock_real_now<uint64_t, std::nano>() < start_time + duration) return;
    progress = progress_point::get();
}
}  // namespace causal
}  // namespace omnitrace
