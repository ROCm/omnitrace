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

#include "library/causal/progress_point.hpp"
#include "library/defines.hpp"
#include "library/thread_data.hpp"
#include "library/utility.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/utility/unwind.hpp>

#include <vector>

namespace omnitrace
{
namespace unwind = ::tim::unwind;

namespace causal
{
static constexpr size_t unwind_depth  = 4;
static constexpr size_t unwind_offset = 2;
using unwind_stack_t                  = unwind::stack<unwind_depth>;

using hash_value_t = tim::hash_value_t;

void set_current_selection(unwind_stack_t);

unwind_stack_t
sample_selection(size_t _nitr = 1000);

void push_progress_point(std::string_view);

void pop_progress_point(std::string_view);

bool
sample_enabled(int64_t _tid = utility::get_thread_index());

uint16_t
sample_virtual_speedup(int64_t _tid = utility::get_thread_index());

void
start_experimenting();

// void
// test();
}  // namespace causal
}  // namespace omnitrace
