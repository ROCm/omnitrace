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

#include "library/causal/progress_point.hpp"
#include "library/common.hpp"
#include "library/debug.hpp"
#include "library/thread_data.hpp"
#include "library/timemory.hpp"
#include "timemory/mpl/type_traits.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/units.hpp>

namespace omnitrace
{
namespace causal
{
namespace
{
struct progress_point_data
{
    comp::wall_clock throughput = {};
    comp::wall_clock latency    = {};
};

using data_t        = std::unordered_map<tim::hash_value_t, progress_point_data>;
using thread_data_t = thread_data<data_t, category::causal>;

auto&
get_progress_point_data()
{
    return thread_data_t::instance(construct_on_init{});
}
}  // namespace

std::string
progress_point::label()
{
    return "progress_point";
}

std::string
progress_point::description()
{
    return "Tracks progress point latency and throughput for casual profiling";
}

void
progress_point::start() const
{
    if(m_hash == 0) return;
    auto& itr = (*get_progress_point_data())[m_hash];
    if(itr.latency.get_is_running())
    {
        tim::invoke::invoke<operation::stop>(std::tie(itr.latency));
    }
    else
    {
        tim::invoke::invoke<operation::start>(std::tie(itr.latency));
    }
    tim::invoke::invoke<operation::start>(std::tie(itr.throughput));
}

void
progress_point::stop() const
{
    if(m_hash == 0) return;
    auto& itr = (*get_progress_point_data())[m_hash];
    tim::invoke::invoke<operation::stop>(std::tie(itr.throughput));
    // tim::invoke::invoke<operation::plus>(std::tie(itr.throughput), m_throughput);
    // itr.throughput.set_laps(itr.throughput.get_laps() + 1);
}

void
progress_point::set_prefix(hash_type _v)
{
    m_hash = _v;
}

void
progress_point::setup()
{
    if(!trait::runtime_enabled<progress_point>::get()) return;

    auto_lock_t _lk{ type_mutex<progress_point>() };
    trait::runtime_enabled<progress_point>::set(false);

    auto& _data = thread_data_t::instances();
    for(auto& itr : _data)
    {
        if(itr) *itr = data_t{};
    }

    trait::runtime_enabled<progress_point>::set(true);
}

progress_point::result_data_t
progress_point::get()
{
    auto_lock_t _lk{ type_mutex<progress_point>() };

    auto& _data   = thread_data_t::instances();
    auto  _result = result_data_t{};

    trait::runtime_enabled<progress_point>::set(false);
    for(auto& itr : _data)
    {
        if(!itr) continue;
        for(const auto& eitr : *itr)
        {
            auto _latency = eitr.second.latency.get_laps() / eitr.second.latency.get();
            auto _throughput =
                eitr.second.throughput.get() / eitr.second.throughput.get_laps();

            _result[eitr.first].latency +=
                (eitr.second.latency.get_laps() > 0) ? (1.0 / _latency) : 0.0;
            _result[eitr.first].throughput += 1.0 / _throughput;
        }
    }

    trait::runtime_enabled<progress_point>::set(true);
    return _result;
}
}  // namespace causal
}  // namespace omnitrace
