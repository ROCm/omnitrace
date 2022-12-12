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

#include "library/components/backtrace_causal.hpp"
#include "library/causal/data.hpp"
#include "library/causal/delay.hpp"
#include "library/causal/experiment.hpp"
#include "library/debug.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"
#include "library/utility.hpp"

#include <timemory/components/timing/backends.hpp>
#include <timemory/components/timing/wall_clock.hpp>

#include <atomic>
#include <type_traits>

namespace omnitrace
{
namespace component
{
namespace
{
auto sampling_count = std::atomic<uint64_t>{ 0 };

auto&
get_sampling_start()
{
    static auto _v = tracing::now();
    return _v;
}

auto samples = std::map<uint32_t, std::set<backtrace_causal::sample_data>>{};
}  // namespace

void
backtrace_causal::start()
{
    get_sampling_start() = tracing::now();
}

void
backtrace_causal::sample(int)
{
    m_valid    = (get_state() == State::Active);
    m_internal = (get_thread_state() == ThreadState::Internal);
    m_index    = causal::experiment::get_index();

    ++sampling_count;

    causal::delay::process();

    using namespace tim::backtrace;
    constexpr bool   with_signal_frame = false;
    constexpr size_t ignore_depth      = 3;
    m_stack    = get_unw_stack<causal::unwind_depth, ignore_depth, with_signal_frame>();
    m_selected = causal::experiment::is_selected(m_stack);

    if(m_selected)
    {
        causal::experiment::add_selected();
        causal::delay::get_local() += causal::experiment::get_delay();
    }

    causal::set_current_selection(m_stack);
}

template <typename Tp>
Tp
backtrace_causal::get_period(uint64_t _units)
{
    using cast_type = std::conditional_t<std::is_floating_point<Tp>::value, Tp, double>;

    auto _ticks = (tracing::now() - get_sampling_start());
    auto _count = sampling_count.load();
    if(_count < 20) return Tp{ 0 };
    auto _mean = static_cast<Tp>(_ticks) / static_cast<cast_type>(_count);
    return static_cast<Tp>(_mean) / static_cast<cast_type>(_units);
}

std::set<backtrace_causal::sample_data>
backtrace_causal::get_samples(uint32_t _index)
{
    return samples[_index];
}

std::map<uint32_t, std::set<backtrace_causal::sample_data>>
backtrace_causal::get_samples()
{
    return samples;
}

void
backtrace_causal::add_sample(uint32_t _index, const sample_data::base_type& _v)
{
    // auto_lock_t _lk{ type_mutex<backtrace_causal::sample_data>() };
    auto& _samples = samples[_index];
    auto  _value   = sample_data{ _v };
    _value.count   = 1;
    auto itr       = _samples.find(_value);
    if(itr == _samples.end())
        _samples.emplace(_value);
    else
        itr->count += 1;
}

void
backtrace_causal::add_samples(uint32_t                                   _index,
                              const std::vector<sample_data::base_type>& _v)
{
    for(const auto& itr : _v)
        add_sample(_index, itr);
}

}  // namespace component
}  // namespace omnitrace

#define INSTANTIATE_BT_CAUSAL_PERIOD(TYPE)                                               \
    template TYPE omnitrace::component::backtrace_causal::get_period<TYPE>(uint64_t);

INSTANTIATE_BT_CAUSAL_PERIOD(float)
INSTANTIATE_BT_CAUSAL_PERIOD(double)
INSTANTIATE_BT_CAUSAL_PERIOD(int64_t)
INSTANTIATE_BT_CAUSAL_PERIOD(uint64_t)
