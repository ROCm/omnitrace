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
#include "library/code_object.hpp"
#include "library/defines.hpp"
#include "library/thread_data.hpp"
#include "library/utility.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/utility/unwind.hpp>

#include <map>
#include <vector>

namespace omnitrace
{
namespace unwind = ::tim::unwind;

namespace causal
{
static constexpr size_t unwind_depth  = 8;
static constexpr size_t unwind_offset = 3;
using unwind_stack_t                  = unwind::stack<unwind_depth>;
using unwind_cache_t                  = typename unwind_stack_t::cache_type;
using hash_value_t                    = tim::hash_value_t;

struct selected_entry
{
    size_t         index = unwind_depth;
    unwind_stack_t stack = {};

    bool operator==(const selected_entry&) const;
    bool operator!=(const selected_entry&) const;

    bool empty() const;
    auto get_entry() const;

    template <typename... Args>
    decltype(auto) get(Args&&... _args);

    template <typename... Args>
    decltype(auto) get(Args&&... _args) const;
};

void set_current_selection(unwind_stack_t);

selected_entry
sample_selection(size_t _nitr = 1000, size_t _wait_ns = 10000);

void push_progress_point(std::string_view);

void pop_progress_point(std::string_view);

bool
sample_enabled(int64_t _tid = utility::get_thread_index());

uint16_t
sample_virtual_speedup(int64_t _tid = utility::get_thread_index());

void
start_experimenting();

inline bool
selected_entry::empty() const
{
    return (index == unwind_depth || stack.empty());
}

inline bool
selected_entry::operator==(const selected_entry& _v) const
{
    return std::tie(index, stack) == std::tie(_v.index, _v.stack);
}

inline bool
selected_entry::operator!=(const selected_entry& _v) const
{
    return !(*this == _v);
}

inline auto
selected_entry::get_entry() const
{
    if(index < unwind_depth)
    {
        OMNITRACE_LIKELY;
        return stack.at(index);
    }
    return std::optional<unwind::entry>{};
}

template <typename... Args>
inline decltype(auto)
selected_entry::get(Args&&... _args)
{
    return stack.get(std::forward<Args>(_args)...);
}

template <typename... Args>
inline decltype(auto)
selected_entry::get(Args&&... _args) const
{
    return stack.get(std::forward<Args>(_args)...);
}

const std::map<code_object::procfs::maps, code_object::basic_line_info>&
get_causal_line_info();

const std::map<code_object::procfs::maps, code_object::basic_line_info>&
get_causal_ignored_line_info();

code_object::basic::line_info
find_line_info(uintptr_t, bool _include_ignored = false);
}  // namespace causal
}  // namespace omnitrace
