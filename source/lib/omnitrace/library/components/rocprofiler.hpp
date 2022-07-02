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

#include <atomic>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <tuple>
#include <unistd.h>
#include <utility>
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
using metric_type  = unsigned long long;
using info_entry_t = std::tuple<std::string, std::string, std::string>;

struct RocmCounter
{
    metric_type counters[OMNITRACE_MAX_COUNTERS];
};

struct RocmEvent
{
    struct RocmCounter entry  = {};
    struct RocmCounter exit   = {};
    std::string        name   = {};
    int                taskid = 0;

    RocmEvent() = default;
    RocmEvent(std::string _event_name, metric_type begin, metric_type end, int _task_id)
    : name(std::move(_event_name))
    , taskid(_task_id)
    {
        entry.counters[0] = begin;
        exit.counters[0]  = end;
    }

    void printEvent()
    {
        std::cout << name << " Task: " << taskid << ", \t\tEntry: " << entry.counters[0]
                  << " , Exit = " << exit.counters[0];
    }

    bool appearsBefore(const RocmEvent& other_event) const
    {
        // both entry and exit of my event is before the entry of the other event.
        return ((taskid == other_event.taskid) &&
                (entry.counters[0] < other_event.entry.counters[0]) &&
                (exit.counters[0] < other_event.entry.counters[0]));
    }

    friend bool operator<(const RocmEvent& _lhs, const RocmEvent& _rhs)
    {
        return (_lhs.entry.counters[0] < _rhs.entry.counters[0]);
    }
};

extern std::vector<RocmEvent> RocmList;

extern void
rocm_initialize();

extern void
rocm_cleanup();

extern void
process_rocm_events(RocmEvent e);

extern void
process_rocm_events(RocmEvent e);

extern int
get_initialized_queues(int queue_id);

extern void
set_initialized_queues(int queue_id, int value);

extern double
metric_set_synchronized_gpu_timestamp(int tid, double value);

extern void
add_metadata_for_task(const char* key, int value, int taskid);

extern bool
check_timestamps(unsigned long long last_timestamp, unsigned long long current_timestamp,
                 const char* debug_str, int taskid);
extern void
publish_event(RocmEvent event);

extern void
process_rocm_events(RocmEvent e);

extern void
flush_rocm_events_if_necessary(int thread_id);

extern metric_type
get_last_timestamp_ns(void);

extern void
set_last_timestamp_ns(metric_type timestamp);

std::map<unsigned, std::vector<info_entry_t>>
rocm_metrics();

uint64_t
trace_get_time_stamp();

void
metric_set_gpu_timestamp(int tid, double value);

void
metadata_task(const char* name, const char* value, int tid);

void
stop_top_level_timer_if_necessary_task(int tid);

#if !defined(OMNITRACE_USE_ROCPROFILER) || OMNITRACE_USE_ROCPROFILER == 0
inline std::map<unsigned, std::vector<info_entry_t>>
rocm_metrics()
{
    return std::map<unsigned, std::vector<info_entry_t>>{};
}

#endif

}  // namespace rocprofiler
}  // namespace omnitrace
