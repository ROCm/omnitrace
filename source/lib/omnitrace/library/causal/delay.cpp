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
#include "library/utility.hpp"

#include <timemory/components/macros.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/types.hpp>

#include <atomic>
#include <random>

namespace omnitrace
{
namespace causal
{
namespace
{
auto&
get_delay_data()
{
    using thread_data_t = thread_data<identity<int64_t>, delay>;
    static auto& _v     = thread_data_t::construct(
        construct_on_init{}, []() { return delay::get_global().load(); });
    return _v;
}

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
    return "Tracks delays and handles implementing delays for causal profiling";
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
    OMNITRACE_BASIC_VERBOSE(1,
                            "[causal] overhead of std::this_thread::sleep_for(...) "
                            "invocation = %6.3f usec +/- %e\n",
                            _stats.get_mean() / units::usec,
                            _stats.get_stddev() / units::usec);
    tim::manager::instance()->add_metadata([_stats](auto& ar) {
        ar(tim::cereal::make_nvp("causal thread sleep overhead [nsec]", _stats));
    });
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

static auto _delay_sample_interval =
    get_env<int64_t>("OMNITRACE_CAUSAL_DELAY_INTERVAL", 8);

void
delay::sample(int)
{
    if(!trait::runtime_enabled<delay>::get()) return;
    if(get_state() >= ::omnitrace::State::Finalized) return;

    static thread_local int64_t _n = 0;
    if(++_n % _delay_sample_interval == 0) process();
}

void
delay::process()
{
    if(!trait::runtime_enabled<delay>::get()) return;
    if(get_state() >= ::omnitrace::State::Finalized) return;

    // if(causal::experiment::is_active())
    {
        if(get_global() < get_local())
        {
            auto _diff = (get_local() - get_global());
            if(_diff > sleep_for_overhead) get_global() += _diff;
        }
        else if(get_global() > get_local())
        {
            // using clock_type    = std::chrono::steady_clock;
            // using duration_type = std::chrono::duration<clock_type::rep, std::nano>;
            // auto _tp            = clock_type::now() + duration_type{ _global - _local
            // };
            auto _beg = tracing::now();
            // std::this_thread::sleep_until(_tp);
            std::this_thread::sleep_for(
                std::chrono::nanoseconds{ get_global() - get_local() });
            get_local() += (tracing::now() - _beg);
        }
    }
    // else
    {
        // get_local() = get_global();
    }
}

void
delay::credit()
{
    if(!trait::runtime_enabled<delay>::get()) return;
    if(get_state() >= ::omnitrace::State::Finalized) return;

    auto _diff = get_global() - get_local();
    if(_diff > 0)
    {
        get_local() += _diff;
    }
}

void
delay::preblock()
{
    if(!trait::runtime_enabled<delay>::get()) return;
    if(get_state() >= ::omnitrace::State::Finalized) return;

    auto _diff = get_global() - get_local();
    if(_diff > 0)
    {
        get_local() += _diff;
    }
}

void
delay::postblock(int64_t _preblock_global_delay_value)
{
    if(!trait::runtime_enabled<delay>::get()) return;
    if(get_state() >= ::omnitrace::State::Finalized) return;

    get_local() += (get_global() - _preblock_global_delay_value);
}

int64_t
delay::sync()
{
    auto _v = get_global().load(std::memory_order_seq_cst);
    if(get_delay_data()) get_delay_data()->fill(_v);
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
    auto&                    _data     = get_delay_data();
    static thread_local auto _thr_init = []() {
        using thread_data_t = thread_data<identity<int64_t>, delay>;
        thread_data_t::construct(construct_on_thread{ utility::get_thread_index() },
                                 get_global().load());
        return true;
    }();
    return _data->at(_tid);
    (void) _thr_init;
}

uint64_t
delay::compute_total_delay(uint64_t _baseline)
{
    // this function assumes that baseline is <= all entries due to call to sync
    return get_global().load() - _baseline;
    // uint64_t _sum = 0;
    // for(auto itr : delay_thread_data::instances())
    //    _sum += (itr - _baseline);
    // return _sum;
}
}  // namespace causal
}  // namespace omnitrace

TIMEMORY_INVOKE_PREINIT(omnitrace::causal::delay)
