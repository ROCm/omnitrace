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
#include "library/causal/experiment.hpp"
#include "library/debug.hpp"
#include "library/thread_data.hpp"
#include "library/utility.hpp"

#include <atomic>
#include <cstdlib>
#include <random>

namespace omnitrace
{
namespace causal
{
namespace
{
using random_engine_t = std::mt19937_64;

int8_t              increment            = 5;
std::vector<int8_t> virtual_speedup_dist = []() {
    size_t              _n = std::max<size_t>(1, 100 / increment);
    std::vector<int8_t> _v(_n, int8_t{ 0 });
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

using mutex_t       = std::mutex;
using lock_t        = std::unique_lock<mutex_t>;
auto progress_mutex = std::array<mutex_t, max_supported_threads>{};

unique_ptr_t<progress_stack_vector_t>&
get_progress_stack(int64_t _tid = utility::get_thread_index())
{
    using thread_data_t = thread_data<progress_stack_vector_t, category::causal>;
    static auto& _v     = thread_data_t::instances(thread_data_t::construct_on_init{},
                                               progress_stack_vector_t{});
    update_max_thread_index(_tid);
    return _v.at(_tid);
}

const progress_stack*
get_current_progress(int64_t _tid = utility::get_thread_index())
{
    if(!get_progress_stack(_tid)) return nullptr;
    if(get_progress_stack(_tid)->empty()) return nullptr;
    return &get_progress_stack(_tid)->back();
}
}  // namespace

void
push_progress_stack(progress_stack&& _v)
{
    lock_t _lk{ progress_mutex.at(utility::get_thread_index()) };
    get_progress_stack()->emplace_back(_v);
}

progress_stack
pop_progress_stack()
{
    lock_t _lk{ progress_mutex.at(utility::get_thread_index()) };
    auto&& _v = get_progress_stack()->back();
    get_progress_stack()->pop_back();
    return _v;
}

bool
sample_enabled(int64_t _tid)
{
    using thread_data_t =
        thread_data<identity<std::uniform_int_distribution<int>>, experiment>;
    static auto& _v = thread_data_t::instances(construct_on_init{}, 0, 1);
    return (_v.at(_tid)(get_engine(_tid)) == 1);
}

int8_t
sample_virtual_speedup(int64_t _tid)
{
    using thread_data_t =
        thread_data<identity<std::uniform_int_distribution<size_t>>, experiment>;
    static auto& _v = thread_data_t::instances(construct_on_init{}, size_t{ 0 },
                                               virtual_speedup_dist.size() - 1);
    return virtual_speedup_dist.at(_v.at(_tid)(get_engine(_tid)));
}

progress_stack
sample_progress_stack()
{
    auto                  _tid = utility::get_thread_index();
    const progress_stack* _v   = nullptr;
    while(_v == nullptr)
    {
        auto   _idx = sample_thread_index(_tid);
        lock_t _lk{ progress_mutex.at(_idx) };
        _v = get_current_progress(_idx);
        if(_v) return *_v;
    }
    return progress_stack{};
}

void
test()
{
    size_t                   _n     = 0;
    std::map<bool, size_t>   _on    = {};
    std::map<int8_t, size_t> _speed = {};
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
}
}  // namespace causal
}  // namespace omnitrace
