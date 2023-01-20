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

#include "library/binary/address_range.hpp"
#include "library/binary/fwd.hpp"
#include "library/utility.hpp"

#include <deque>
#include <memory>
#include <vector>

namespace omnitrace
{
namespace binary
{
template <typename DataT>
struct line_info
{
    using value_type = DataT;

    std::shared_ptr<bfd_file>                file     = {};
    std::deque<value_type>                   data     = {};
    std::vector<address_range>               ranges   = {};
    std::unordered_map<address_range, void*> sections = {};

    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto size() const { return data.size(); }
    auto empty() const { return data.empty(); }

    auto range_begin() const { return ranges.begin(); }
    auto range_end() const { return ranges.end(); }
    auto range_begin() { return ranges.begin(); }
    auto range_end() { return ranges.end(); }
    auto range_size() const { return ranges.size(); }
    auto range_empty() const { return ranges.empty(); }

    void sort();

    template <typename RetT = void>
    RetT* find_section(uintptr_t);
};

template <typename DataT>
void
line_info<DataT>::sort()
{
    utility::filter_sort_unique(data);
    utility::filter_sort_unique(ranges);
}

template <typename DataT>
template <typename RetT>
RetT*
line_info<DataT>::find_section(uintptr_t _addr)
{
    for(const auto& sitr : sections)
    {
        if(sitr.first.contains(_addr)) return static_cast<RetT*>(sitr.second);
    }
    return nullptr;
}
}  // namespace binary
}  // namespace omnitrace
