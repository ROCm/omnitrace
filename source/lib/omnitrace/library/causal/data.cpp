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

#include "library/causal/data.hpp"
#include "library/causal/delay.hpp"
#include "library/causal/experiment.hpp"
#include "library/debug.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
#include "library/state.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/utility.hpp"

#include <timemory/hash/types.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <random>
#include <thread>

namespace omnitrace
{
namespace causal
{
namespace
{
using random_engine_t    = std::mt19937_64;
using progress_bundles_t = component_bundle_cache<progress_point>;

uint16_t              increment            = 5;
std::vector<uint16_t> virtual_speedup_dist = []() {
    size_t                _n = std::max<size_t>(1, 100 / increment);
    std::vector<uint16_t> _v(_n, uint16_t{ 0 });
    std::generate(_v.begin(), _v.end(),
                  [_value = 0]() mutable { return (_value += increment); });
    OMNITRACE_CI_THROW(_v.back() > 100, "Error! last value is too large: %i\n",
                       (int) _v.back());
    return _v;
}();
std::vector<size_t> seeds = {};

auto
get_seed()
{
    static auto       _v = std::random_device{};
    static std::mutex _mutex;
    std::unique_lock  _lk{ _mutex };
    seeds.emplace_back(_v());
    return seeds.back();
}

auto&
get_engine(int64_t _tid = threading::get_id())
{
    using thread_data_t = thread_data<identity<random_engine_t>, experiment>;
    static auto& _v =
        thread_data_t::instances(construct_on_init{}, []() { return get_seed(); });
    return _v.at(_tid);
}

std::atomic<int64_t>&
get_max_thread_index()
{
    static auto _v = std::atomic<int64_t>{ 1 };
    return _v;
}

void
update_max_thread_index(int64_t _tid)
{
    bool _success = false;
    while(!_success)
    {
        auto _v = get_max_thread_index().load();
        if(_tid <= _v) break;
        _success = get_max_thread_index().compare_exchange_strong(
            _v, _tid, std::memory_order_relaxed);
    }
}

int64_t
sample_thread_index(int64_t _tid)
{
    return std::uniform_int_distribution<int64_t>{ 0, get_max_thread_index().load() }(
        get_engine(_tid));
}

void
perform_experiment_impl()
{
    thread_info::init(true);

    auto _experim = experiment{};
    // loop until started or finalized
    while(!_experim.start())
    {
        if(get_state() == State::Finalized) return;
    }

    // wait for the experiment to complete
    _experim.wait();

    // loop until stopped or finalized
    while(!_experim.stop())
    {
        if(get_state() == State::Finalized) return;
    }

    if(get_state() == State::Finalized) return;

    // cool-down / inactive period
    std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });

    // start a new experiment
    tasking::general::get_task_group().exec(perform_experiment_impl);
}

auto latest_progress_point = unwind_stack_t{};
}  // namespace

//--------------------------------------------------------------------------------------//

void
set_current_selection(unwind_stack_t _stack)
{
    // data-race but that is ok
    latest_progress_point = _stack;
}

unwind_stack_t
sample_selection(size_t _nitr)
{
    size_t         _n   = 0;
    unwind_stack_t _val = {};
    while((_val = latest_progress_point).empty())
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::microseconds{ 1 });
        if(_n++ >= _nitr) break;
    }
    return _val;
}

void
push_progress_point(std::string_view _name)
{
    auto  _hash   = tim::add_hash_id(_name);
    auto& _data   = progress_bundles_t::instance(utility::get_thread_index());
    auto* _bundle = _data.construct(_hash);
    _bundle->push();
    _bundle->start();
}

void
pop_progress_point(std::string_view _name)
{
    auto  _hash = tim::add_hash_id(_name);
    auto& _data = progress_bundles_t::instance(utility::get_thread_index());
    for(auto itr = _data.rbegin(); itr != _data.rend(); ++itr)
    {
        if((*itr)->get_hash() == _hash)
        {
            (*itr)->stop();
            (*itr)->pop();
            _data.destroy(itr);
            return;
        }
    }
}

bool
sample_enabled(int64_t _tid)
{
    using thread_data_t =
        thread_data<identity<std::uniform_int_distribution<int>>, experiment>;
    static auto& _v = thread_data_t::instances(construct_on_init{}, 0, 1);
    return (_v.at(_tid)(get_engine(_tid)) == 1);
}

uint16_t
sample_virtual_speedup(int64_t _tid)
{
    using thread_data_t =
        thread_data<identity<std::uniform_int_distribution<size_t>>, experiment>;
    static auto& _v = thread_data_t::instances(construct_on_init{}, size_t{ 0 },
                                               virtual_speedup_dist.size() - 1);
    return virtual_speedup_dist.at(_v.at(_tid)(get_engine(_tid)));
}

void
start_experimenting()
{
    thread_info::init(true);

    // wait until fully activated before proceeding
    while(get_state() < State::Active)
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
    }

    if(get_state() != State::Finalized)
    {
        // start a new experiment
        tasking::general::get_task_group().exec(perform_experiment_impl);
    }
}
/*
void test()
{
    size_t                   _n     = 0;
    std::map<bool, size_t>   _on    = {};
    std::map<uint16_t, size_t> _speed = {};
    for(size_t j = 0; j < 1000; ++j)
    {
        for(int64_t i = 0; i < 64; ++i)
        {
            auto _enabled = sample_enabled(i);
            auto _speedup = (_enabled) ? sample_virtual_speedup(i) : 0;
            ++_n;
            _on[_enabled] += 1;
            _speed[_speedup] += 1;
        }
    }
    OMNITRACE_PRINT("\n");
    for(auto& itr : _on)
    {
        OMNITRACE_PRINT("[%5s] usage: %4.1f%s\n", std::to_string(itr.first).c_str(),
                        static_cast<double>(itr.second) / _n * 100., "%");
    }
    OMNITRACE_PRINT("\n");
    for(auto& itr : _speed)
    {
        OMNITRACE_PRINT("[%5i] usage: %4.1f%s\n", (int) itr.first,
                        static_cast<double>(itr.second) / _n * 100., "%");
    }
    std::exit(EXIT_SUCCESS);
}*/
}  // namespace causal
}  // namespace omnitrace
