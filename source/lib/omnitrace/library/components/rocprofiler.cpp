/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#include "library/components/rocprofiler.hpp"
#include "library/debug.hpp"
#include "library/rocm.hpp"
#include "library/rocprofiler/hsa_rsrc_factory.hpp"

#include <cstdlib>
#include <mutex>
#include <rocprofiler.h>

#include <timemory/manager.hpp>

#include <atomic>
#include <dlfcn.h>
#include <hsa.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <unistd.h>
#include <vector>

#define PUBLIC_API      __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API  __attribute__((destructor))

namespace omnitrace
{
namespace rocprofiler
{
// Tool is unloaded
volatile bool is_loaded = false;
// Profiling features
// rocprofiler_feature_t* features = nullptr;
// unsigned feature_count = 0;

// Error handler
void
fatal(const std::string& msg)
{
    fflush(stdout);
    fprintf(stderr, "%s\n\n", msg.c_str());
    fflush(stderr);
    abort();
}

// Check returned HSA API status
void
rocm_check_status(hsa_status_t status)
{
    if(status != HSA_STATUS_SUCCESS)
    {
        const char* error_string = nullptr;
        rocprofiler_error_string(&error_string);
        fprintf(stderr, "ERROR: %s\n", error_string);
        abort();
    }
}

// Context stored entry type
struct context_entry_t
{
    bool                        valid;
    hsa_agent_t                 agent;
    rocprofiler_group_t         group;
    rocprofiler_callback_data_t data;
};

// Context callback arg
struct callbacks_arg_t
{
    rocprofiler_pool_t** pools;
};

// Handler callback arg
struct handler_arg_t
{
    rocprofiler_feature_t* features;
    unsigned               feature_count;
};

// Dump stored context entry
void
rocm_dump_context_entry(context_entry_t* entry, rocprofiler_feature_t* features,
                        unsigned feature_count)
{
    static metric_type last_timestamp = get_last_timestamp_ns();

    volatile std::atomic<bool>* valid =
        reinterpret_cast<std::atomic<bool>*>(&entry->valid);
    while(valid->load() == false)
        sched_yield();

    const rocprofiler_dispatch_record_t* record      = entry->data.record;
    std::string                          kernel_name = entry->data.kernel_name;
    if(!record) return;  // there is nothing to do here.

    kernel_name = kernel_name.substr(0, kernel_name.find(" [clone"));
    int queueid = entry->data.queue_id;
    int taskid  = get_initialized_queues(queueid);

    /*if(taskid == -1)
    {  // not initialized
        // OMNITRACE_CREATE_TASK(taskid);
        OMNITRACE_VERBOSE_F(
            0, "rocm_dump_context_entry: associating queueid %d with taskid %d\n",
            queueid, taskid);
        set_initialized_queues(queueid, taskid);
        timestamp = record->dispatch;
        check_timestamps(last_timestamp, timestamp, "NEW QUEUE", taskid);
        last_timestamp = timestamp;
        // Set the timestamp for TAUGPU_TIME:
        metric_set_synchronized_gpu_timestamp(taskid, ((double) timestamp / 1e3));
        // create_top_level_timer_if_necessary_task(taskid);
        // add_metadata_for_task("OMNITRACE_TASK_ID", taskid, taskid);
        // add_metadata_for_task(
        //    "ROCM_GPU_ID",
        //    HsaRsrcFactory::Instance().GetAgentInfo(entry->agent)->dev_index, taskid);
        // add_metadata_for_task("ROCM_QUEUE_ID", entry->data.queue_id, taskid);
        // add_metadata_for_task("ROCM_THREAD_ID", entry->data.thread_id, taskid);
    }*/

    auto _evt = RocmEvent{ kernel_name, record->begin, record->end, taskid };

    OMNITRACE_VERBOSE_F(0, "KERNEL: name: %s, entry: %lu, exit: %lu ...\n",
                        kernel_name.c_str(), record->begin, record->end);

    process_rocm_events(_evt);
    auto timestamp = record->begin;

    last_timestamp = timestamp;
    timestamp      = record->end;
    last_timestamp = timestamp;
    set_last_timestamp_ns(record->complete);

#ifdef DEBUG_PROF
    fprintf(stdout, "kernel symbol(0x%lx) name(\"%s\") tid(%ld) queue-id(%u) gpu-id(%u) ",
            entry->data.kernel_object, kernel_name.c_str(), entry->data.thread_id,
            entry->data.queue_id,
            HsaRsrcFactory::Instance().GetAgentInfo(entry->agent)->dev_index);
    if(record)
        fprintf(stdout, "time(%lu,%lu,%lu,%lu)", record->dispatch, record->begin,
                record->end, record->complete);
    fprintf(stdout, "\n");
    fflush(stdout);
#endif

    rocprofiler_group_t& group = entry->group;
    if(group.context == nullptr)
    {
        fatal("context is nullptr\n");
    }

    if(feature_count > 0)
    {
        hsa_status_t status = rocprofiler_group_get_data(&group);
        rocm_check_status(status);
        status = rocprofiler_get_metrics(group.context);
        rocm_check_status(status);
    }

    for(unsigned i = 0; i < feature_count; ++i)
    {
        const rocprofiler_feature_t* p = &features[i];
        switch(p->data.kind)
        {
            // Output metrics results
            case ROCPROFILER_DATA_KIND_UNINIT: break;
            case ROCPROFILER_DATA_KIND_BYTES: break;
            case ROCPROFILER_DATA_KIND_INT32:
                OMNITRACE_VERBOSE_F(0, ">  %-20s = %12u\n", p->name,
                                    p->data.result_int32);
                break;
            case ROCPROFILER_DATA_KIND_FLOAT:
                OMNITRACE_VERBOSE_F(0, ">  %-20s = %12.3f\n", p->name,
                                    p->data.result_float);
                break;
            case ROCPROFILER_DATA_KIND_DOUBLE:
                OMNITRACE_VERBOSE_F(0, ">  %-20s = %12.3f\n", p->name,
                                    p->data.result_double);
                break;
            case ROCPROFILER_DATA_KIND_INT64:
                OMNITRACE_VERBOSE_F(0, ">  %-20s = %12lu\n", p->name,
                                    p->data.result_int64);
                break;
        }
    }
}

// Profiling completion handler
// Dump and delete the context entry
// Return true if the context was dumped successfully
bool
rocm_context_handler(const rocprofiler_pool_entry_t* entry, void* arg)
{
    // Context entry
    context_entry_t* ctx_entry   = reinterpret_cast<context_entry_t*>(entry->payload);
    handler_arg_t*   handler_arg = reinterpret_cast<handler_arg_t*>(arg);

    rocm::lock_t _lk{ rocm::rocm_mutex, std::defer_lock };
    if(!_lk.owns_lock()) _lk.lock();

    rocm_dump_context_entry(ctx_entry, handler_arg->features, handler_arg->feature_count);

    return false;
}

// Kernel disoatch callback
hsa_status_t
rocm_dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* arg,
                       rocprofiler_group_t* group)
{
    // Passed tool data
    hsa_agent_t agent = callback_data->agent;
    // HSA status
    hsa_status_t status = HSA_STATUS_ERROR;

#if 1
    // Open profiling context
    const unsigned   gpu_id = HsaRsrcFactory::Instance().GetAgentInfo(agent)->dev_index;
    callbacks_arg_t* callbacks_arg = reinterpret_cast<callbacks_arg_t*>(arg);
    rocprofiler_pool_t*      pool  = callbacks_arg->pools[gpu_id];
    rocprofiler_pool_entry_t pool_entry{};
    status = rocprofiler_pool_fetch(pool, &pool_entry);
    rocm_check_status(status);
    // Profiling context entry
    rocprofiler_t*   context = pool_entry.context;
    context_entry_t* entry   = reinterpret_cast<context_entry_t*>(pool_entry.payload);
#else
    // Open profiling context
    // context properties
    context_entry_t*         entry   = new context_entry_t{};
    rocprofiler_t*           context = nullptr;
    rocprofiler_properties_t properties{};
    properties.handler     = rocm_context_handler1;
    properties.handler_arg = (void*) entry;
    status                 = rocprofiler_open(agent, features, feature_count, &context,
                              0 /*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
    rocm_check_status(status);
#endif
    // Get group[0]
    status = rocprofiler_get_group(context, 0, group);
    rocm_check_status(status);

    // Fill profiling context entry
    entry->agent            = agent;
    entry->group            = *group;
    entry->data             = *callback_data;
    entry->data.kernel_name = strdup(callback_data->kernel_name);
    reinterpret_cast<std::atomic<bool>*>(&entry->valid)->store(true);

    return HSA_STATUS_SUCCESS;
}

unsigned
metrics_input(rocprofiler_feature_t** ret)
{
    // OMNITRACE_THROW("%s\n", __FUNCTION__);
    // Profiling feature objects
    const unsigned         feature_count = 6;
    rocprofiler_feature_t* features      = new rocprofiler_feature_t[feature_count];
    memset(features, 0, feature_count * sizeof(rocprofiler_feature_t));

    // PMC events
    features[0].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    features[0].name = "GRBM_COUNT";
    features[1].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    features[1].name = "GRBM_GUI_ACTIVE";
    features[2].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    features[2].name = "GPUBusy";
    features[3].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    features[3].name = "SQ_WAVES";
    features[4].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    features[4].name = "SQ_INSTS_VALU";
    features[5].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    features[5].name = "VALUInsts";
    //  features[6].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    //  features[6].name = "TCC_HIT_sum";
    //  features[7].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    //  features[7].name = "TCC_MISS_sum";
    //  features[8].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    //  features[8].name = "WRITE_SIZE";

    *ret = features;
    return feature_count;
}

void
rocm_initialize()
{
    // Available GPU agents
    const unsigned gpu_count = HsaRsrcFactory::Instance().GetCountOfGpuAgents();

    // Getting profiling features
    rocprofiler_feature_t* features = nullptr;
    // TAU doesn't support features or metrics yet! SSS
    unsigned feature_count = metrics_input(&features);
    // unsigned feature_count = 0;

    // Handler arg
    handler_arg_t* handler_arg = new handler_arg_t{};
    handler_arg->features      = features;
    handler_arg->feature_count = feature_count;

    // Context properties
    rocprofiler_pool_properties_t properties{};
    properties.num_entries   = 100;
    properties.payload_bytes = sizeof(context_entry_t);
    properties.handler       = rocm_context_handler;
    properties.handler_arg   = handler_arg;

    // Adding dispatch observer
    callbacks_arg_t* callbacks_arg = new callbacks_arg_t{};
    callbacks_arg->pools           = new rocprofiler_pool_t*[gpu_count];
    for(unsigned gpu_id = 0; gpu_id < gpu_count; gpu_id++)
    {
        // Getting GPU device info
        const AgentInfo* agent_info = nullptr;
        if(HsaRsrcFactory::Instance().GetGpuAgentInfo(gpu_id, &agent_info) == false)
        {
            fprintf(stderr, "GetGpuAgentInfo failed\n");
            abort();
        }

        // Open profiling pool
        rocprofiler_pool_t* pool = nullptr;
        hsa_status_t        status =
            rocprofiler_pool_open(agent_info->dev_id, features, feature_count, &pool,
                                  0 /*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
        rocm_check_status(status);
        callbacks_arg->pools[gpu_id] = pool;
    }

    rocprofiler_queue_callbacks_t callbacks_ptrs{};
    callbacks_ptrs.dispatch = rocm_dispatch_callback;
    int err = rocprofiler_set_queue_callbacks(callbacks_ptrs, callbacks_arg);
    OMNITRACE_VERBOSE_F(0, "err=%d, rocprofiler_set_queue_callbacks\n", err);
}

void
rocm_cleanup()
{
    // Unregister dispatch callback
    rocprofiler_remove_queue_callbacks();
    // CLose profiling pool
#if 0
  hsa_status_t status = rocprofiler_pool_flush(pool);
  rocm_check_status(status);
  status = rocprofiler_pool_close(pool);
  rocm_check_status(status);
#endif
}

std::vector<RocmEvent> RocmList;
namespace
{
metric_type        last_timestamp_published = 0;
metric_type        last_timestamp_ns        = 0L;
unsigned long long offset_timestamp         = 0L;
}  // namespace

static auto&
get_initialized_queues()
{
    static auto _v = []() {
        auto _tmp = std::array<int, OMNITRACE_MAX_ROCM_QUEUES>{};
        _tmp.fill(-1);
        return _tmp;
    }();
    return _v;
}

// Dispatch callbacks and context handlers synchronization
// Tool is unloaded

// access tau_initialized_queues through get and set access functions
int
get_initialized_queues(int queue_id)
{
    return get_initialized_queues().at(queue_id);
}

void
set_initialized_queues(int queue_id, int value)
{
    get_initialized_queues().at(queue_id) = value;
}

double
metric_set_synchronized_gpu_timestamp(int /*tid*/, double value)
{
    // printf("value = %f\n", value);
    if(offset_timestamp == 0L)
    {
        // offset_timestamp = trace_get_time_stamp() - ((double) value);
    }
    // metric_set_gpu_timestamp(tid, offset_timestamp + value);
    OMNITRACE_VERBOSE_F(0, "metric_set_gpu_timestamp = %llu + %f = %f\n",
                        offset_timestamp, value, offset_timestamp + value);
    return offset_timestamp + value;
}

// access last_timestamp_ns through get and set access functions
metric_type
get_last_timestamp_ns()
{
    return last_timestamp_ns;
}

void
set_last_timestamp_ns(metric_type timestamp)
{
    last_timestamp_ns = timestamp;
}

void
add_metadata_for_task(const char* key, int value, int taskid)
{
    char buf[1024];
    sprintf(buf, "%d", value);
    // metadata_task(key, buf, taskid);
    OMNITRACE_VERBOSE_F(0, "Adding Metadata: %s, %d, for task %d\n", key, value, taskid);
}

bool
is_thread_id_rocm_task(int thread_id)
{
    // Just for checking!
    for(int i = 0; i < OMNITRACE_MAX_ROCM_QUEUES; i++)
    {
        if(get_initialized_queues(i) == thread_id) return true;
    }
    return false;
}

bool
check_timestamps(unsigned long long last_timestamp, unsigned long long current_timestamp,
                 const char* debug_str, int taskid)
{
    OMNITRACE_VERBOSE_F(0,
                        "Taskid<%d>: check_timestamps: Checking last_timestamp = %llu, "
                        "current_timestamp = %llu at %s\n",
                        taskid, last_timestamp, current_timestamp, debug_str);
    if(last_timestamp > current_timestamp)
    {
        OMNITRACE_VERBOSE_F(
            0,
            "Taskid<%d>: check_timestamps: Timestamps are not monotonically increasing! "
            "last_timestamp = %llu, current_timestamp = %llu at %s\n",
            taskid, last_timestamp, current_timestamp, debug_str);
        return false;
    }
    else
        return true;
}

void
publish_event(RocmEvent event)
{
    metric_type timestamp = 0;

    // First the entry
    if(event.entry.counters[0] < last_timestamp_published)
    {
        OMNITRACE_VERBOSE_F(
            0,
            "ERROR: publish_event: Event to be published has a timestamp = "
            "%llu that is earlier than the last published timestamp %llu\n",
            event.entry.counters[0], last_timestamp_published);
        OMNITRACE_VERBOSE_F(
            0,
            "ERROR: please re-configure TAU with -useropt=-DOMNITRACE_ROCM_LOOK_AHEAD=16 "
            "(or some higher value) for a window of events to sort, make install and "
            "retry the command\n");
        OMNITRACE_VERBOSE_F(0, "Ignoring this event:\n");
        return;
    }

    timestamp = event.entry.counters[0];  // Using first element of array
    metric_set_synchronized_gpu_timestamp(
        event.taskid, ((double) timestamp / 1e3));  // convert to microseconds
    // OMNITRACE_START_TASK(event.name.c_str(), event.taskid);
    OMNITRACE_VERBOSE_F(0, "Started event %s on task %d timestamp = %llu \n",
                        event.name.c_str(), event.taskid, timestamp);

    // then the exit
    timestamp                = event.exit.counters[0];  // Using first element of array
    last_timestamp_published = timestamp;
    metric_set_synchronized_gpu_timestamp(
        event.taskid, ((double) timestamp / 1e3));  // convert to microseconds
    // OMNITRACE_STOP_TASK(event.name.c_str(), event.taskid);
    OMNITRACE_VERBOSE_F(0, "Stopped event %s on task %d timestamp = %llu \n",
                        event.name.c_str(), event.taskid, timestamp);
}

void
process_rocm_events(RocmEvent _evt)
{
    RocmList.emplace_back(_evt);

    std::sort(RocmList.begin(), RocmList.end());

    if(RocmList.size() < OMNITRACE_ROCM_LOOK_AHEAD)
    {
        return;  // don't do anything else
    }
    else
    {
        // There are elements in the list. What are they?
        // publish_event(RocmList.front());
        // RocmList.pop_front();
        // RocmList.erase(RocmList.begin());
    }
}

extern void
flush_rocm_events_if_necessary(int thread_id)
{
    static bool reentrant_flag = false;
    if(reentrant_flag)
    {
        OMNITRACE_VERBOSE_F(
            0, "flush_rocm_events_if_necessary: reentrant_flag = true, tid = %d\n",
            thread_id);
        return;
    }
    if(is_thread_id_rocm_task(thread_id))
    {
        reentrant_flag = true;
    }
    else
    {
        return;
    }

    if(RocmList.empty()) return;
    std::sort(RocmList.begin(), RocmList.end());
    while(!RocmList.empty())
    {
        publish_event(RocmList.front());
        RocmList.erase(RocmList.begin());
    }

    //  stop_top_level_timer_if_necessary();

    for(int i = 0; i < OMNITRACE_MAX_ROCM_QUEUES; i++)
    {
        if(get_initialized_queues(i) != -1)
        {
            // RtsLayer::LockDB();
            if(get_initialized_queues(i) != -1)
            {  // contention. Is it still -1?
                OMNITRACE_VERBOSE_F(0, "Closing thread id: %d last timestamp = %llu\n",
                                    get_initialized_queues(i), last_timestamp_ns);
                metric_set_synchronized_gpu_timestamp(
                    i,
                    ((double) last_timestamp_ns / 1e3));  // convert to microseconds
                // stop_top_level_timer_if_necessary_task(get_initialized_queues(i));
                set_initialized_queues(i, -1);
            }
            // RtsLayer::UnLockDB();
        }
    }
    reentrant_flag = false; /* safe */
}
}  // namespace rocprofiler
}  // namespace omnitrace
