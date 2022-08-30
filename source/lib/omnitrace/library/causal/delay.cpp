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
#include "library/runtime.hpp"
#include "library/state.hpp"
#include "library/thread_data.hpp"

#include <timemory/mpl/types.hpp>

namespace omnitrace
{
namespace causal
{
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
delay::start()
{
    if(!trait::runtime_enabled<delay>::get()) return;

    // if thread state is internal, this thread is in an instrumentation region
    if(get_thread_state() == ThreadState::Internal)
    {
        ++get_local();
        ++get_global();
    }

    process();
}

void
delay::stop()
{
    if(!trait::runtime_enabled<delay>::get()) return;
}

void
delay::sample()
{
    if(!trait::runtime_enabled<delay>::get()) return;
}

void
delay::process()
{
    if(!trait::runtime_enabled<delay>::get()) return;

    auto _diff = get_local() - get_global();
    if(_diff < 0)
    {
        std::this_thread::sleep_for(std::chrono::microseconds{ -1000 * _diff });
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

    auto _diff = get_global() - get_local();
    if(_diff > 0)
    {
        get_local() += _diff;
    }
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
    using thread_data_t = thread_data<identity<int64_t>, delay>;
    static auto& _v     = thread_data_t::instances(construct_on_init{}, 0);
    return _v.at(_tid);
}

}  // namespace causal
}  // namespace omnitrace
