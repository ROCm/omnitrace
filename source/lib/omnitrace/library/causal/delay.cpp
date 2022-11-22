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

#include "library/causal/delay.hpp"
#include "library/causal/experiment.hpp"
#include "library/runtime.hpp"
#include "library/state.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/mpl/concepts.hpp"

#include <timemory/mpl/types.hpp>

#include <random>

namespace omnitrace
{
namespace causal
{
namespace
{
using delay_thread_data    = thread_data<identity<int64_t>, delay>;
int64_t sleep_for_overhead = 0;
}  // namespace

std::string
delay::label()
{
    return "causal_delay";
}

std::string
delay::description()
{
    return "Tracks delays and handles implementing delays for casual profiling";
}

void
delay::preinit()
{
    using random_engine_t = std::mt19937_64;

    random_engine_t                        _engine{ std::random_device{}() };
    std::uniform_int_distribution<int64_t> _dist{ 1, 10 };
    size_t                                 _ntot  = 250;
    size_t                                 _nwarm = 50;
    tim::statistics<double>                _stats{};
    for(size_t i = 0; i < _ntot; ++i)
    {
        auto    _val = _dist(_engine);
        int64_t _beg = tracing::now();
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _val });
        int64_t _end = tracing::now();
        if(i < _nwarm) continue;
        auto _diff = (_end - _beg);
        OMNITRACE_CONDITIONAL_THROW(
            _diff < _val, "Error! sleep_for(%zu) [nanoseconds] >= %zu", _val, _diff);
        _stats += (_diff - _val);
    }
    sleep_for_overhead = _stats.get_mean();
    OMNITRACE_BASIC_PRINT("overhead of this_thread::sleep_for(chrono::nanoseconds{ ... "
                          "}) = %6.3f usec +/- %e\n",
                          _stats.get_mean() / units::usec,
                          _stats.get_stddev() / units::usec);
    tim::manager::instance()->add_metadata("this_thread::sleep_for overhead mean [nsec]",
                                           _stats.get_mean());
    tim::manager::instance()->add_metadata(
        "this_thread::sleep_for overhead stddev [nsec]", _stats.get_stddev());
}

void
delay::start()
{
    if(!trait::runtime_enabled<delay>::get()) return;

    sample();
}

void
delay::stop()
{
    if(!trait::runtime_enabled<delay>::get()) return;

    process();
}

void
delay::sample(int)
{
    if(!trait::runtime_enabled<delay>::get()) return;
    if(get_state() == ::omnitrace::State::Finalized) return;

    auto _causal_state = get_causal_state();
    // TIMEMORY_PRINTF(stderr, "[causal_state][%li] %s\n", utility::get_thread_index(),
    //                std::to_string(_causal_state.state).c_str());
    if(_causal_state.state == CausalState::Selected)
    {
        auto _delay = experiment::get_delay();
        get_local() += _delay;
        get_global() += _delay;
    }

    process();
}

void
delay::process()
{
    if(!trait::runtime_enabled<delay>::get()) return;
    if(get_state() == ::omnitrace::State::Finalized) return;

    static thread_local const auto& _info = thread_info::get();

    if(!_info) return;

    int64_t _diff = get_local() - get_global();
    if(_diff < 0)
    {
        int64_t _val = -1 * _diff;
        if(_val >= sleep_for_overhead)
        {
            std::this_thread::sleep_for(
                std::chrono::nanoseconds{ _val - sleep_for_overhead });
        }
        get_local() += _val;
    }
    else if(_diff > 0)
    {
        get_global() += _diff;
    }
}

void
delay::credit()
{
    if(!trait::runtime_enabled<delay>::get()) return;
    if(get_state() == ::omnitrace::State::Finalized) return;

    auto _diff = get_global() - get_local();
    if(_diff > 0)
    {
        get_local() += _diff;
    }
}

int64_t
delay::sync()
{
    auto _v = get_global().load();
    delay_thread_data::instances().fill(_v);
    return _v;
}

std::atomic<int64_t>&
delay::get_global()
{
    static auto _v = std::atomic<int64_t>{ 0 };
    return _v;
}

int64_t&
delay::get_local(int64_t _tid)
{
    static auto& _v = delay_thread_data::instances(construct_on_init{}, 0);
    return _v.at(_tid);
}

uint64_t
delay::compute_total_delay(uint64_t _baseline)
{
    // this function assumes that baseline is <= all entries due to call to sync
    uint64_t _sum = 0;
    for(auto itr : delay_thread_data::instances())
        _sum += (itr - _baseline);
    return _sum;
}
}  // namespace causal
}  // namespace omnitrace

TIMEMORY_INVOKE_PREINIT(omnitrace::causal::delay)
