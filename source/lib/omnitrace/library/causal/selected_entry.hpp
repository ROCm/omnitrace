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

#include "library/binary/basic_line_info.hpp"
#include "library/binary/fwd.hpp"
#include "library/causal/fwd.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/unwind/dlinfo.hpp>
#include <timemory/unwind/stack.hpp>
#include <timemory/utility/macros.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <map>
#include <utility>

namespace omnitrace
{
namespace causal
{
struct selected_entry
{
    using line_info = binary::basic_line_info;

    TIMEMORY_DEFAULT_OBJECT(selected_entry)

    selected_entry(uintptr_t, uintptr_t, uintptr_t, const line_info&);

    uintptr_t       address        = 0x0;
    uintptr_t       symbol_address = 0x0;
    uintptr_t       binary_address = {};
    address_range_t range          = {};
    line_info       info           = {};

    hash_value_t hash() const;

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned int);

    bool     contains(uintptr_t) const;
    bool     operator==(const selected_entry&) const;
    bool     operator!=(const selected_entry&) const;
    explicit operator bool() const { return (address > 0 && info.address); }
};

inline bool
selected_entry::contains(uintptr_t _v) const
{
    auto _addr = (symbol_address > 0) ? symbol_address : address;
    return (_addr == _v || range.contains(_v));
}

inline bool
selected_entry::operator==(const selected_entry& _v) const
{
    return (address == _v.address && symbol_address == _v.symbol_address &&
            binary_address == _v.binary_address && info == _v.info);
}

inline bool
selected_entry::operator!=(const selected_entry& _v) const
{
    return !(*this == _v);
}
}  // namespace causal
}  // namespace omnitrace
