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
struct bfd_line_info
{
    bool          weak         = false;
    unsigned int  priority     = 0;
    unsigned int  line         = 0;
    uintptr_t     load_address = 0;
    address_range address      = {};
    std::string   file         = {};
    std::string   func         = {};

    bool     is_inlined() const;
    bool     is_weak() const;
    bool     is_valid() const;
    explicit operator bool() const { return is_valid(); }

    basic_line_info get_basic() const;
    address_range   ipaddr() const { return address + load_address; }
    auto            low() const { return address.low; }
    auto            high() const { return address.high; }

    friend bool operator<(const bfd_line_info& _lhs, const bfd_line_info& _rhs)
    {
        return (_lhs.load_address == _rhs.load_address)
                   ? ((_lhs.address == _rhs.address) ? (_lhs.priority < _rhs.priority)
                                                     : (_lhs.address < _rhs.address))
                   : (_lhs.load_address < _rhs.load_address);
    }

    friend bool operator==(const bfd_line_info& _lhs, const bfd_line_info& _rhs)
    {
        return std::tie(_lhs.load_address, _lhs.address, _lhs.line, _lhs.file,
                        _lhs.func) ==
               std::tie(_rhs.load_address, _rhs.address, _rhs.line, _rhs.file, _rhs.func);
    }

    friend bool operator!=(const bfd_line_info& _lhs, const bfd_line_info& _rhs)
    {
        return !(_lhs == _rhs);
    }

    static std::vector<bfd_line_info> process_bfd(bfd_file& _bfd, void* _section,
                                                  uintptr_t _pc, uintptr_t _pc_len = 0);
};
}  // namespace binary
}  // namespace omnitrace
