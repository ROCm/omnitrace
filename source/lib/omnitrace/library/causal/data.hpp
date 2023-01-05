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
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/utility/unwind.hpp>

#include <dlfcn.h>
#include <map>
#include <vector>

namespace omnitrace
{
namespace unwind = ::tim::unwind;

namespace causal
{
static constexpr size_t unwind_depth  = 6;
static constexpr size_t unwind_offset = 0;
using unwind_stack_t                  = unwind::stack<unwind_depth>;
using unwind_cache_t                  = typename unwind_stack_t::cache_type;
using hash_value_t                    = tim::hash_value_t;

template <typename Tp>
std::string
as_hex(Tp _v, size_t _width = 16);

struct selected_entry
{
    using line_info = code_object::basic::line_info;

    uintptr_t address        = 0x0;
    uintptr_t symbol_address = 0x0;
    line_info info           = {};

    hash_value_t hash() const;

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned int);

    bool contains(uintptr_t) const;
    bool operator==(const selected_entry&) const;
    bool operator!=(const selected_entry&) const;
         operator bool() const { return (address > 0 && info.address > 0); }
};

inline hash_value_t
selected_entry::hash() const
{
    return tim::get_combined_hash_id(tim::hash_value_t{ address }, info.hash());
}

template <typename ArchiveT>
void
selected_entry::serialize(ArchiveT& ar, const unsigned int)
{
    using ::tim::cereal::make_nvp;
    ar(make_nvp("address", address), make_nvp("symbol_address", symbol_address),
       make_nvp("info", info));
}

void
save_line_info(const settings::compose_filename_config&);

std::vector<code_object::basic::line_info>
get_line_info(uintptr_t _addr, bool include_discarded = true);

bool is_eligible_address(uintptr_t);

void set_current_selection(utility::c_array<void*>);

void set_current_selection(unwind_stack_t);

selected_entry
sample_selection(size_t _nitr = 1000, size_t _wait_ns = 10000);

void push_progress_point(std::string_view);

void pop_progress_point(std::string_view);

uint16_t
sample_virtual_speedup(int64_t _tid = utility::get_thread_index());

void
start_experimenting();

inline bool
selected_entry::contains(uintptr_t _v) const
{
    if(symbol_address > 0)
    {
        Dl_info _dl_info{};
        if(dladdr(reinterpret_cast<void*>(_v), &_dl_info) != 0 && _dl_info.dli_saddr)
            return (symbol_address == reinterpret_cast<uintptr_t>(_dl_info.dli_saddr));
    }
    return (address == _v);
}

inline bool
selected_entry::operator==(const selected_entry& _v) const
{
    return (address == _v.address && symbol_address == _v.symbol_address &&
            info == _v.info);
}

inline bool
selected_entry::operator!=(const selected_entry& _v) const
{
    return !(*this == _v);
}
}  // namespace causal
}  // namespace omnitrace
