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

#include "library/code_object.hpp"
#include "library/containers/c_array.hpp"
#include "library/containers/static_vector.hpp"
#include "library/defines.hpp"
#include "library/thread_data.hpp"
#include "library/utility.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/utility/procfs/maps.hpp>
#include <timemory/utility/unwind.hpp>

#include <deque>
#include <dlfcn.h>
#include <map>

namespace omnitrace
{
namespace unwind = ::tim::unwind;

namespace causal
{
static constexpr size_t unwind_depth  = 8;
static constexpr size_t unwind_offset = 0;
using unwind_stack_t                  = unwind::stack<unwind_depth>;
using unwind_addr_t                   = container::static_vector<uintptr_t, unwind_depth>;
using hash_value_t                    = tim::hash_value_t;

struct selected_entry
{
    using line_info = code_object::basic::line_info;

    uintptr_t address        = 0x0;
    uintptr_t symbol_address = 0x0;
    line_info info           = {};

    hash_value_t hash() const;

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned int);

    bool     contains(uintptr_t) const;
    bool     operator==(const selected_entry&) const;
    bool     operator!=(const selected_entry&) const;
    explicit operator bool() const { return (address > 0 && info.address); }
};

void
save_line_info(const settings::compose_filename_config&);

using line_mapping_info_t =
    std::pair<code_object::procfs::maps, code_object::basic::line_info>;

std::deque<line_mapping_info_t>
get_line_info(uintptr_t _addr, bool include_discarded = true);

bool is_eligible_address(uintptr_t);

void set_current_selection(unwind_stack_t);

void set_current_selection(unwind_addr_t);

selected_entry
sample_selection(size_t _nitr = 1000, size_t _wait_ns = 10000);

void push_progress_point(std::string_view);

void pop_progress_point(std::string_view);

void mark_progress_point(std::string_view);

uint16_t
sample_virtual_speedup();

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
