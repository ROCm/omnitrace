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

#pragma once

#include "library/defines.hpp"
#include "library/thread_data.hpp"
#include "library/timemory.hpp"

#include <timemory/backends/hardware_counters.hpp>
#include <timemory/macros.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/macros.hpp>

#include <array>
#include <atomic>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

#ifndef OMNITRACE_MAX_COUNTERS
#    define OMNITRACE_MAX_COUNTERS 25
#endif

#ifndef OMNITRACE_ROCM_LOOK_AHEAD
#    define OMNITRACE_ROCM_LOOK_AHEAD 128
#endif

#ifndef OMNITRACE_MAX_ROCM_QUEUES
#    define OMNITRACE_MAX_ROCM_QUEUES OMNITRACE_MAX_THREADS
#endif

namespace omnitrace
{
namespace rocprofiler
{
using metric_type     = unsigned long long;
using info_entry_t    = ::tim::hardware_counters::info;
using feature_value_t = std::variant<uint32_t, float, uint64_t, double>;

struct rocm_counter
{
    std::array<metric_type, OMNITRACE_MAX_COUNTERS> counters;
};

struct rocm_event
{
    using value_type = feature_value_t;

    uint32_t                      device_id      = 0;
    uint32_t                      thread_id      = 0;
    uint32_t                      queue_id       = 0;
    metric_type                   entry          = 0;
    metric_type                   exit           = 0;
    std::string                   name           = {};
    std::vector<std::string_view> feature_names  = {};
    std::vector<feature_value_t>  feature_values = {};

    rocm_event() = default;
    rocm_event(uint32_t _dev, uint32_t _thr, uint32_t _queue, std::string _event_name,
               metric_type begin, metric_type end, uint32_t _feature_count,
               void* _features);

    std::string as_string() const;

    friend std::ostream& operator<<(std::ostream& _os, const rocm_event& _v)
    {
        return (_os << _v.as_string());
    }

    friend bool operator<(const rocm_event& _lhs, const rocm_event& _rhs)
    {
        return std::tie(_lhs.device_id, _lhs.queue_id, _lhs.entry, _lhs.thread_id) <
               std::tie(_rhs.device_id, _rhs.queue_id, _rhs.entry, _rhs.thread_id);
    }
};

using rocm_data_t       = std::vector<rocm_event>;
using rocm_data_tracker = comp::data_tracker<feature_value_t, rocm_event>;

unique_ptr_t<rocm_data_t>&
rocm_data(int64_t _tid = threading::get_id());

std::map<uint32_t, std::vector<std::string_view>>
get_data_labels();

void
rocm_initialize();

void
rocm_cleanup();

bool&
is_setup();

void
post_process();

std::vector<info_entry_t>
rocm_metrics();

#if !defined(OMNITRACE_USE_ROCPROFILER) || OMNITRACE_USE_ROCPROFILER == 0
inline void
post_process()
{}
#endif

}  // namespace rocprofiler
}  // namespace omnitrace
