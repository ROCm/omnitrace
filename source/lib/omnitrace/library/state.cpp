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

#include "library/state.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/utility.hpp"

#include <string>

namespace omnitrace
{
namespace
{
auto&
get_state_value()
{
    static State _v{ State::PreInit };
    return _v;
}

ThreadState&
get_thread_state_value()
{
    static thread_local ThreadState _v{ ThreadState::Enabled };
    return _v;
}

auto&
get_thread_state_history(int64_t _idx = utility::get_thread_index())
{
    static auto _v = utility::get_filled_array<OMNITRACE_MAX_THREADS>(
        []() { return utility::get_reserved_vector<ThreadState>(32); });

    if(_idx >= OMNITRACE_MAX_THREADS)
    {
        static thread_local auto _tl_v = utility::get_reserved_vector<ThreadState>(32);
        return _tl_v;
    }

    return _v.at(_idx);
}

CausalState&
get_causal_state_impl()
{
    static thread_local auto _v = CausalState::Enabled;
    return _v;
}
}  // namespace

State
get_state()
{
    return get_state_value();
}

ThreadState
get_thread_state()
{
    return get_thread_state_value();
}

State
set_state(State _n)
{
    OMNITRACE_CONDITIONAL_PRINT_F(get_debug_init(), "Setting state :: %s -> %s\n",
                                  std::to_string(get_state()).c_str(),
                                  std::to_string(_n).c_str());
    // state should always be increased, not decreased
    OMNITRACE_CI_BASIC_THROW(
        _n < get_state(), "State is being assigned to a lesser value :: %s -> %s",
        std::to_string(get_state()).c_str(), std::to_string(_n).c_str());
    std::swap(get_state_value(), _n);
    return _n;
}

ThreadState
set_thread_state(ThreadState _n)
{
    std::swap(get_thread_state_value(), _n);
    return _n;
}

ThreadState
push_thread_state(ThreadState _v)
{
    if(get_thread_state() >= ThreadState::Completed) return get_thread_state();

    return get_thread_state_history().emplace_back(set_thread_state(_v));
}

ThreadState
pop_thread_state()
{
    if(get_thread_state() >= ThreadState::Completed) return get_thread_state();

    auto& _hist = get_thread_state_history();
    if(!_hist.empty())
    {
        set_thread_state(_hist.back());
        _hist.pop_back();
    }
    return get_thread_state();
}

//--------------------------------------------------------------------------------------//
//
//      Causal state
//
//--------------------------------------------------------------------------------------//

CausalState
get_causal_state()
{
    return get_causal_state_impl();
}

CausalState
set_causal_state(CausalState _v)
{
    // if causal state is disabled, immediately return
    // if causal state is selected, do not change the state. Instead
    // increment the selected state counter so that the thread
    // knows it is still in the selected function and does not delay
    auto& _state = get_causal_state_impl();
    std::swap(_state, _v);
    return _v;
}
}  // namespace omnitrace

namespace std
{
std::string
to_string(omnitrace::State _v)
{
    switch(_v)
    {
        case omnitrace::State::PreInit: return "PreInit";
        case omnitrace::State::Init: return "Init";
        case omnitrace::State::Active: return "Active";
        case omnitrace::State::Disabled: return "Disabled";
        case omnitrace::State::Finalized: return "Finalized";
    }
    return {};
}

std::string
to_string(omnitrace::ThreadState _v)
{
    switch(_v)
    {
        case omnitrace::ThreadState::Enabled: return "Enabled";
        case omnitrace::ThreadState::Internal: return "Internal";
        case omnitrace::ThreadState::Completed: return "Completed";
        case omnitrace::ThreadState::Disabled: return "Disabled";
    }
    return {};
}

std::string
to_string(omnitrace::CausalState _v)
{
    switch(_v)
    {
        case omnitrace::CausalState::Enabled: return "Enabled";
        case omnitrace::CausalState::Internal: return "Internal";
        case omnitrace::CausalState::Selected: return "Selected";
        case omnitrace::CausalState::Disabled: return "Disabled";
    }
    return {};
}

std::string
to_string(omnitrace::Mode _v)
{
    switch(_v)
    {
        case omnitrace::Mode::Trace: return "Trace";
        case omnitrace::Mode::Sampling: return "Sampling";
        case omnitrace::Mode::Causal: return "Causal";
        case omnitrace::Mode::Coverage: return "Coverage";
    }
    return {};
}

std::string
to_string(omnitrace::CausalMode _v)
{
    switch(_v)
    {
        case omnitrace::CausalMode::Line: return "Line";
        case omnitrace::CausalMode::Function: return "Function";
    }
    return {};
}
}  // namespace std
