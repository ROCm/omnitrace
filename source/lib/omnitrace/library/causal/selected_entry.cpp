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

#include "library/causal/selected_entry.hpp"
#include "library/common.hpp"
#include "library/timemory.hpp"

namespace omnitrace
{
namespace causal
{
selected_entry::selected_entry(uintptr_t _addr, uintptr_t _sym, uintptr_t _bin,
                               const line_info& _info)
: address{ _addr }
, symbol_address{ _sym }
, binary_address{ _bin }
, range{ _info.address + _bin }
, info{ _info }
{}

hash_value_t
selected_entry::hash() const
{
    return tim::get_combined_hash_id(tim::hash_value_t{ address }, symbol_address,
                                     binary_address, info.hash());
}

template <typename ArchiveT>
void
selected_entry::serialize(ArchiveT& ar, const unsigned int)
{
    using ::tim::cereal::make_nvp;
    ar(make_nvp("address", address), make_nvp("symbol_address", symbol_address),
       make_nvp("binary_address", binary_address), make_nvp("address_range", range),
       make_nvp("info", info));
}

template void
selected_entry::serialize<cereal::JSONInputArchive>(cereal::JSONInputArchive&,
                                                    const unsigned int);

template void
selected_entry::serialize<cereal::MinimalJSONOutputArchive>(
    cereal::MinimalJSONOutputArchive&, const unsigned int);

template void
selected_entry::serialize<cereal::PrettyJSONOutputArchive>(
    cereal::PrettyJSONOutputArchive&, const unsigned int);
}  // namespace causal
}  // namespace omnitrace
