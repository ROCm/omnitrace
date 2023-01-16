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
struct dwarf_line_info
{
    TIMEMORY_DEFAULT_OBJECT(dwarf_line_info)

    bool         valid           = false;
    bool         begin_statement = false;
    bool         end_sequence    = false;
    bool         line_block      = false;
    bool         prologue_end    = false;
    bool         epilogue_begin  = false;
    unsigned int line            = 0;
    int          col             = 0;
    unsigned int vliw_op_index   = 0;
    unsigned int isa             = 0;
    unsigned int discriminator   = 0;
    uintptr_t    address         = 0;
    std::string  file            = {};

    bool     is_valid() const;
    explicit operator bool() const { return is_valid(); }

    basic_line_info get_basic() const;

    friend bool operator<(const dwarf_line_info& _lhs, const dwarf_line_info& _rhs)
    {
        return std::tie(_lhs.address, _lhs.line, _lhs.col, _lhs.discriminator) <
               std::tie(_rhs.address, _rhs.line, _rhs.col, _rhs.discriminator);
    }

    friend bool operator==(const dwarf_line_info& _lhs, const dwarf_line_info& _rhs)
    {
        return std::tie(_lhs.address, _lhs.line, _lhs.col, _lhs.discriminator,
                        _lhs.vliw_op_index, _lhs.isa, _lhs.file) ==
               std::tie(_rhs.address, _rhs.line, _rhs.col, _rhs.discriminator,
                        _rhs.vliw_op_index, _rhs.isa, _rhs.file);
    }

    friend bool operator!=(const dwarf_line_info& _lhs, const dwarf_line_info& _rhs)
    {
        return !(_lhs == _rhs);
    }

    static std::deque<dwarf_line_info> process_dwarf(int _fd,
                                                     std::vector<address_range>&);
};
}  // namespace binary
}  // namespace omnitrace
