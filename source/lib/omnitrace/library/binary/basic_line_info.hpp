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

namespace omnitrace
{
namespace binary
{
struct basic_line_info
{
    bool          inlined = false;  // inlined symbol
    bool          weak    = false;  // weak symbol
    unsigned int  line    = 0;
    address_range address = {};
    std::string   file    = {};
    std::string   func    = {};

    bool         is_valid() const { return !file.empty() && address > address_range{}; }
    hash_value_t hash() const;
    std::string  name() const;
    explicit     operator bool() const { return is_valid(); }

    friend bool operator<(const basic_line_info& _lhs, const basic_line_info& _rhs)
    {
        return _lhs.address < _rhs.address;
    }

    friend bool operator==(const basic_line_info& _lhs, const basic_line_info& _rhs)
    {
        return std::tie(_lhs.address, _lhs.line, _lhs.file, _lhs.func) ==
               std::tie(_rhs.address, _rhs.line, _rhs.file, _rhs.func);
    }

    friend bool operator!=(const basic_line_info& _lhs, const basic_line_info& _rhs)
    {
        return !(_lhs == _rhs);
    }

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned int);
};

inline hash_value_t
basic_line_info::hash() const
{
    return tim::get_combined_hash_id(address.hash(), line, file, func);
}

inline std::string
basic_line_info::name() const
{
    if(!file.empty()) return (line > 0) ? JOIN(':', file, line) : file;
    return func;
}

template <typename ArchiveT>
void
basic_line_info::serialize(ArchiveT& ar, const unsigned int)
{
    ar(cereal::make_nvp("address", address), cereal::make_nvp("file", file),
       cereal::make_nvp("line", line), cereal::make_nvp("func", func));
    if constexpr(concepts::is_output_archive<ArchiveT>::value)
        ar(cereal::make_nvp("dfunc", demangle(func)));
    ar(cereal::make_nvp("weak", weak));
    ar(cereal::make_nvp("inlined", inlined));
}
}  // namespace binary
}  // namespace omnitrace
