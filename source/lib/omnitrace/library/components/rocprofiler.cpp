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
#include "library/common.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/rocm.hpp"
#include "library/rocprofiler/hsa_rsrc_factory.hpp"

#include <timemory/backends/hardware_counters.hpp>
#include <timemory/manager.hpp>

#include <rocprofiler.h>

#include <atomic>
#include <cstdlib>
#include <dlfcn.h>
#include <hsa.h>
#include <iostream>
#include <mutex>
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
        OMNITRACE_PRINT_F("ERROR: %s\n", error_string);
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
    // static metric_type last_timestamp = get_last_timestamp_ns();

    volatile std::atomic<bool>* valid =
        reinterpret_cast<std::atomic<bool>*>(&entry->valid);
    while(valid->load() == false)
        sched_yield();

    const rocprofiler_dispatch_record_t* record = entry->data.record;
    if(!record) return;  // there is nothing to do here.

    int  queueid     = entry->data.queue_id;
    int  taskid      = get_initialized_queues(queueid);
    auto kernel_name = std::string{ entry->data.kernel_name };
    auto _pos        = kernel_name.find_last_of(')');
    if(_pos != std::string::npos) kernel_name = kernel_name.substr(0, _pos + 1);

    // free(entry->data.kernel_name);

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
    // auto timestamp = record->begin;

    // last_timestamp = timestamp;
    // timestamp      = record->end;
    // last_timestamp = timestamp;
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
        rocm_check_status(rocprofiler_group_get_data(&group));
        rocm_check_status(rocprofiler_get_metrics(group.context));
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

    return true;
}

// Kernel disoatch callback
hsa_status_t
rocm_dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* arg,
                       rocprofiler_group_t* group)
{
    // Passed tool data
    hsa_agent_t agent = callback_data->agent;

    // Open profiling context
    const unsigned   gpu_id = HsaRsrcFactory::Instance().GetAgentInfo(agent)->dev_index;
    callbacks_arg_t* callbacks_arg = reinterpret_cast<callbacks_arg_t*>(arg);
    rocprofiler_pool_t*      pool  = callbacks_arg->pools[gpu_id];
    rocprofiler_pool_entry_t pool_entry{};
    rocm_check_status(rocprofiler_pool_fetch(pool, &pool_entry));
    // Profiling context entry
    rocprofiler_t*   context = pool_entry.context;
    context_entry_t* entry   = reinterpret_cast<context_entry_t*>(pool_entry.payload);

    // Get group[0]
    rocm_check_status(rocprofiler_get_group(context, 0, group));

    // Fill profiling context entry
    entry->agent            = agent;
    entry->group            = *group;
    entry->data             = *callback_data;
    entry->data.kernel_name = strdup(callback_data->kernel_name);
    reinterpret_cast<std::atomic<bool>*>(&entry->valid)->store(true);

    return HSA_STATUS_SUCCESS;
}

unsigned
metrics_input(unsigned _device, rocprofiler_feature_t** ret)
{
    // OMNITRACE_THROW("%s\n", __FUNCTION__);
    // Profiling feature objects
    auto                     _events = tim::delimit(config::get_rocm_events(), ", ;\t\n");
    std::vector<std::string> _features    = {};
    auto                     _this_device = JOIN("", ":device=", _device);
    for(auto itr : _events)
    {
        auto _pos = itr.find(":device=");
        if(_pos != std::string::npos)
        {
            if(itr.find(_this_device) != std::string::npos)
            {
                _features.emplace_back(itr.substr(0, _pos));
            }
        }
        else
        {
            _features.emplace_back(itr);
        }
    }
    const unsigned         feature_count = _features.size();
    rocprofiler_feature_t* features      = new rocprofiler_feature_t[feature_count];
    memset(features, 0, feature_count * sizeof(rocprofiler_feature_t));

    // PMC events
    for(unsigned i = 0; i < feature_count; ++i)
    {
        features[i].kind            = ROCPROFILER_FEATURE_KIND_METRIC;
        features[i].name            = strdup(_features.at(i).c_str());
        features[i].parameters      = nullptr;
        features[i].parameter_count = 0;
    }

    *ret = features;
    return feature_count;
}

struct info_data
{
    const AgentInfo*           agent = nullptr;
    std::vector<info_entry_t>* data  = nullptr;
};

hsa_status_t
info_data_callback(const rocprofiler_info_data_t info, void* arg)
{
    using qualifier_t     = tim::hardware_counters::qualifier;
    using qualifier_vec_t = std::vector<qualifier_t>;
    auto*       _arg      = static_cast<info_data*>(arg);
    const auto* _agent    = _arg->agent;
    auto*       _data     = _arg->data;

    switch(info.kind)
    {
        case ROCPROFILER_INFO_KIND_METRIC:
        {
            auto _device_qualifier_sym = JOIN("", ":device=", _agent->dev_index);
            auto _device_qualifier     = tim::hardware_counters::qualifier{
                true, static_cast<int>(_agent->dev_index), _device_qualifier_sym,
                JOIN(" ", "Device", _agent->dev_index)
            };
            auto _long_desc = std::string{ info.metric.description };
            auto _units     = std::string{};
            auto _pysym     = std::string{};
            if(info.metric.expr != nullptr)
            {
                auto _sym        = JOIN("", info.metric.name, _device_qualifier_sym);
                auto _short_desc = JOIN("", "Derived counter: ", info.metric.expr);
                _data->emplace_back(info_entry_t(true, tim::hardware_counters::api::rocm,
                                                 _data->size(), 0, _sym, _pysym,
                                                 _short_desc, _long_desc, _units,
                                                 qualifier_vec_t{ _device_qualifier }));
            }
            else
            {
                if(info.metric.instances == 1)
                {
                    auto _sym = JOIN("", info.metric.name, _device_qualifier_sym);
                    auto _short_desc =
                        JOIN("", info.metric.name, " on device ", _agent->dev_index);
                    _data->emplace_back(info_entry_t(
                        true, tim::hardware_counters::api::rocm, _data->size(), 0, _sym,
                        _pysym, _short_desc, _long_desc, _units,
                        qualifier_vec_t{ _device_qualifier }));
                }
                else
                {
                    for(uint32_t i = 0; i < info.metric.instances; ++i)
                    {
                        auto _instance_qualifier_sym = JOIN("", '[', i, ']');
                        auto _instance_qualifier =
                            tim::hardware_counters::qualifier{ true, static_cast<int>(i),
                                                               _instance_qualifier_sym,
                                                               JOIN(" ", "Instance", i) };
                        auto _sym = JOIN("", info.metric.name, _instance_qualifier_sym,
                                         _device_qualifier_sym);
                        auto _short_desc = JOIN("", info.metric.name, " instance ", i,
                                                " on device ", _agent->dev_index);
                        _data->emplace_back(info_entry_t(
                            true, tim::hardware_counters::api::rocm, _data->size(), 0,
                            _sym, _pysym, _short_desc, _long_desc, _units,
                            qualifier_vec_t{ _device_qualifier, _instance_qualifier }));
                    }
                }
            }
            // fflush(stdout);
            break;
        }
        default: printf("wrong info kind %u\n", info.kind); return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
}

std::vector<info_entry_t>
rocm_metrics()
{
    std::vector<info_entry_t> _data = {};
    // Available GPU agents
    const unsigned gpu_count = HsaRsrcFactory::Instance().GetCountOfGpuAgents();

    std::vector<AgentInfo*> _gpu_agents(gpu_count, nullptr);
    for(unsigned i = 0; i < gpu_count; ++i)
    {
        const AgentInfo*  _agent   = _gpu_agents[i];
        const AgentInfo** _agent_p = &_agent;
        HsaRsrcFactory::Instance().GetGpuAgentInfo(i, _agent_p);

        auto _v = info_data{ _agent, &_data };
        rocm_check_status(
            rocprofiler_iterate_info(&_agent->dev_id, ROCPROFILER_INFO_KIND_METRIC,
                                     info_data_callback, reinterpret_cast<void*>(&_v)));
    }

    auto _settings = tim::settings::shared_instance();
    if(_settings)
    {
        auto ritr = _settings->find("OMNITRACE_ROCM_EVENTS");
        if(ritr != _settings->end())
        {
            auto _rocm_events = ritr->second;
            if(_rocm_events->get_choices().empty())
            {
                std::vector<std::string> _choices = {};
                _choices.reserve(_data.size());
                for(auto itr : _data)
                {
                    if(!itr.symbol().empty()) _choices.emplace_back(itr.symbol());
                }
                _rocm_events->set_choices(_choices);
            }
        }
    }

    return _data;
}

void
rocm_initialize()
{
    // Available GPU agents
    const unsigned gpu_count = HsaRsrcFactory::Instance().GetCountOfGpuAgents();

    (void) rocm_metrics();

    // Adding dispatch observer
    callbacks_arg_t* callbacks_arg = new callbacks_arg_t{};
    callbacks_arg->pools           = new rocprofiler_pool_t*[gpu_count];
    for(unsigned gpu_id = 0; gpu_id < gpu_count; gpu_id++)
    {
        // Getting profiling features
        rocprofiler_feature_t* features = nullptr;
        // TAU doesn't support features or metrics yet! SSS
        unsigned feature_count = metrics_input(gpu_id, &features);
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

        // Getting GPU device info
        const AgentInfo* agent_info = nullptr;
        if(HsaRsrcFactory::Instance().GetGpuAgentInfo(gpu_id, &agent_info) == false)
        {
            fprintf(stderr, "GetGpuAgentInfo failed\n");
            abort();
        }

        // Open profiling pool
        rocprofiler_pool_t* pool = nullptr;
        uint32_t            mode = 0;  // ROCPROFILER_MODE_SINGLEGROUP
        rocm_check_status(rocprofiler_pool_open(agent_info->dev_id, features,
                                                feature_count, &pool, mode, &properties));
        callbacks_arg->pools[gpu_id] = pool;
    }

    rocprofiler_queue_callbacks_t callbacks_ptrs{};
    callbacks_ptrs.dispatch = rocm_dispatch_callback;
    int err = rocprofiler_set_queue_callbacks(callbacks_ptrs, callbacks_arg);
    OMNITRACE_VERBOSE_F(3, "err=%d, rocprofiler_set_queue_callbacks\n", err);
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
    OMNITRACE_VERBOSE_F(4, "metric_set_gpu_timestamp = %llu + %f = %f\n",
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
    OMNITRACE_VERBOSE_F(4, "Adding Metadata: %s, %d, for task %d\n", key, value, taskid);
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
    OMNITRACE_VERBOSE_F(4,
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
    OMNITRACE_VERBOSE_F(4, "Started event %s on task %d timestamp = %llu \n",
                        event.name.c_str(), event.taskid, timestamp);

    // then the exit
    timestamp                = event.exit.counters[0];  // Using first element of array
    last_timestamp_published = timestamp;
    metric_set_synchronized_gpu_timestamp(
        event.taskid, ((double) timestamp / 1e3));  // convert to microseconds
    // OMNITRACE_STOP_TASK(event.name.c_str(), event.taskid);
    OMNITRACE_VERBOSE_F(4, "Stopped event %s on task %d timestamp = %llu \n",
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
                OMNITRACE_VERBOSE_F(4, "Closing thread id: %d last timestamp = %llu\n",
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
