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

#include "common/defines.h"
#include "library/common.hpp"
#include "library/concepts.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/perfetto.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/timemory.hpp"
#include "library/tracing/annotation.hpp"
#include "library/utility.hpp"

#include <timemory/components/timing/backends.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/type_traits.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <ratio>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace omnitrace
{
namespace tracing
{
using interval_data_instances = thread_data<std::vector<bool>>;
using hash_value_t            = tim::hash_value_t;

extern OMNITRACE_HIDDEN_API bool debug_push;
extern OMNITRACE_HIDDEN_API bool debug_pop;
extern OMNITRACE_HIDDEN_API bool debug_user;

perfetto::TraceConfig&
get_perfetto_config();

std::unique_ptr<perfetto::TracingSession>&
get_perfetto_session();

template <typename CategoryT, typename... Args>
auto
get_perfetto_category_uuid(Args&&... _args)
{
    return tim::hash::get_combined_hash_id(
        tim::hash::get_hash_id(JOIN('_', "omnitrace", trait::name<CategoryT>::value)),
        std::forward<Args>(_args)...);
}

std::unordered_map<hash_value_t, std::string>&
get_perfetto_track_uuids();

template <typename CategoryT, typename TrackT = ::perfetto::Track, typename FuncT,
          typename... Args>
auto
get_perfetto_track(CategoryT, FuncT&& _desc_generator, Args&&... _args)
{
    auto  _uuid = get_perfetto_category_uuid<CategoryT>(std::forward<Args>(_args)...);
    auto& _track_uuids = get_perfetto_track_uuids();
    if(_track_uuids.find(_uuid) == _track_uuids.end())
    {
        const auto _track = TrackT(_uuid);
        auto       _desc  = _track.Serialize();

        auto _name = std::forward<FuncT>(_desc_generator)(std::forward<Args>(_args)...);
        _desc.set_name(_name);
        ::perfetto::TrackEvent::SetTrackDescriptor(_track, _desc);

        OMNITRACE_VERBOSE_F(4, "[%s] Created %s(%zu) with description: \"%s\"\n",
                            trait::name<CategoryT>::value, demangle<TrackT>().c_str(),
                            _uuid, _name.c_str());

        _track_uuids.emplace(_uuid, _name);
    }

    // guard this with ppdefs in addition to runtime check to avoid
    // overhead of generating string during releases
#if defined(OMNITRACE_CI) && OMNITRACE_CI > 0
    auto _name = std::forward<FuncT>(_desc_generator)(std::forward<Args>(_args)...);
    OMNITRACE_CI_THROW(_track_uuids.at(_uuid) != _name,
                       "Error! Multiple invocations of UUID %zu produced different "
                       "descriptions: \"%s\" and \"%s\"\n",
                       _uuid, _track_uuids.at(_uuid).c_str(), _name.c_str());
#endif

    return TrackT(_uuid);
}

std::vector<std::function<void()>>&
get_finalization_functions();

tim::hash_map_ptr_t&
get_timemory_hash_ids(int64_t _tid = threading::get_id());

tim::hash_alias_ptr_t&
get_timemory_hash_aliases(int64_t _tid = threading::get_id());

template <typename Tp = uint64_t>
OMNITRACE_INLINE auto
now()
{
    return ::tim::get_clock_real_now<Tp, std::nano>();
}

void
record_thread_start_time();

inline auto&
get_interval_data(int64_t _tid = threading::get_id())
{
    static auto& _v = interval_data_instances::instances(construct_on_init{});
    return _v.at(_tid);
}

inline auto&
get_instrumentation_bundles(int64_t _tid = threading::get_id())
{
    return instrumentation_bundles::instance(_tid);
}

inline auto&
push_count()
{
    static std::atomic<size_t> _v{ 0 };
    return _v;
}

inline auto&
pop_count()
{
    static std::atomic<size_t> _v{ 0 };
    return _v;
}

inline void
thread_init()
{
    static thread_local auto _dtor = scope::destructor{ []() {
        if(get_state() != State::Finalized)
        {
            if(get_use_sampling()) sampling::shutdown();
            auto& _thr_bundle = thread_data<thread_bundle_t>::instance();
            if(_thr_bundle && _thr_bundle->get<comp::wall_clock>() &&
               _thr_bundle->get<comp::wall_clock>()->get_is_running())
                _thr_bundle->stop();
        }
    } };
    static thread_local auto _thread_setup = []() {
        if(threading::get_id() > 0)
            threading::set_thread_name(JOIN(" ", "Thread", threading::get_id()).c_str());
        thread_data<thread_bundle_t>::construct(JOIN('/', "omnitrace/process",
                                                     process::get_id(), "thread",
                                                     threading::get_id()),
                                                quirk::config<quirk::auto_start>{});
        get_interval_data()->reserve(512);
        // save the hash maps
        get_timemory_hash_ids()     = tim::get_hash_ids();
        get_timemory_hash_aliases() = tim::get_hash_aliases();
        record_thread_start_time();
        return true;
    }();
    (void) _thread_setup;
    (void) _dtor;
}

inline void
thread_init_sampling()
{
    static thread_local auto _v = []() {
        auto _idx = utility::get_thread_index();
        // the main thread will initialize sampling when it initializes the tooling
        if(_idx > 0)
        {
            auto _use_sampling = get_use_sampling();
            if(_use_sampling) sampling::setup();
            return _use_sampling;
        }
        return false;
    }();
    (void) _v;
}

template <typename CategoryT, typename... Args>
inline void
push_timemory(CategoryT, const char* name, Args&&... args)
{
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    auto& _data = tracing::get_instrumentation_bundles();
    // this generates a hash for the raw string array
    auto _hash = tim::add_hash_id(tim::string_view_t{ name });
    _data.construct(_hash)->start(std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
pop_timemory(CategoryT, const char* name, Args&&... args)
{
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    auto  _hash = tim::hash::get_hash_id(tim::string_view_t{ name });
    auto& _data = tracing::get_instrumentation_bundles();
    if(_data.bundles.empty())
    {
        OMNITRACE_DEBUG("[%s] skipped %s :: empty bundle stack\n", "omnitrace_pop_trace",
                        name);
        return;
    }
    for(size_t i = _data.bundles.size(); i > 0; --i)
    {
        auto*& _v = _data.bundles.at(i - 1);
        if(_v->get_hash() == _hash)
        {
            _v->stop(std::forward<Args>(args)...);
            _data.allocator.destroy(_v);
            _data.allocator.deallocate(_v, 1);
            _data.bundles.erase(_data.bundles.begin() + (i - 1));
            break;
        }
    }
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto(CategoryT, const char* name, Args&&... args)
{
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    uint64_t _ts = comp::wall_clock::record();
    if constexpr(sizeof...(Args) == 1 &&
                 std::is_invocable<Args..., perfetto::EventContext>::value)
    {
        if(config::get_perfetto_annotations())
        {
            TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, perfetto::StaticString(name),
                              _ts, "begin_ns", _ts, std::forward<Args>(args)...);
        }
        else
        {
            TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, perfetto::StaticString(name),
                              _ts, std::forward<Args>(args)...);
        }
    }
    else
    {
        TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, perfetto::StaticString(name),
                          _ts, std::forward<Args>(args)...,
                          [&](perfetto::EventContext ctx) {
                              if(config::get_perfetto_annotations())
                              {
                                  tracing::add_perfetto_annotation(ctx, "begin_ns", _ts);
                              }
                          });
    }
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto(CategoryT, const char*, Args&&... args)
{
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    uint64_t _ts = comp::wall_clock::record();
    if constexpr(sizeof...(Args) == 1 &&
                 std::is_invocable<Args..., perfetto::EventContext>::value)
    {
        if(config::get_perfetto_annotations())
        {
            TRACE_EVENT_END(trait::name<CategoryT>::value, _ts, "end_ns", _ts,
                            std::forward<Args>(args)...);
        }
        else
        {
            TRACE_EVENT_END(trait::name<CategoryT>::value, _ts,
                            std::forward<Args>(args)...);
        }
    }
    else
    {
        TRACE_EVENT_END(trait::name<CategoryT>::value, _ts, std::forward<Args>(args)...,
                        [&](perfetto::EventContext ctx) {
                            if(config::get_perfetto_annotations())
                            {
                                tracing::add_perfetto_annotation(ctx, "end_ns", _ts);
                            }
                        });
    }
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto_ts(CategoryT, const char* name, uint64_t _ts, Args&&... args)
{
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, perfetto::StaticString(name), _ts,
                      std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_ts(CategoryT, const char*, uint64_t _ts, Args&&... args)
{
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    TRACE_EVENT_END(trait::name<CategoryT>::value, _ts, std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
push_perfetto_track(CategoryT, const char* name, perfetto::Track _track, uint64_t _ts,
                    Args&&... args)
{
    TRACE_EVENT_BEGIN(trait::name<CategoryT>::value, perfetto::StaticString(name), _track,
                      _ts, std::forward<Args>(args)...);
}

template <typename CategoryT, typename... Args>
inline void
pop_perfetto_track(CategoryT, const char*, perfetto::Track _track, uint64_t _ts,
                   Args&&... args)
{
    TRACE_EVENT_END(trait::name<CategoryT>::value, _track, _ts,
                    std::forward<Args>(args)...);
}

}  // namespace tracing
}  // namespace omnitrace
