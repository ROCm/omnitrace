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

#include "library/common.hpp"
#include "library/defines.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/unwind/bfd.hpp>
#include <timemory/unwind/types.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <variant>

namespace omnitrace
{
namespace code_object
{
namespace procfs = ::tim::procfs;

using bfd_file     = ::tim::unwind::bfd_file;
using hash_value_t = ::tim::hash_value_t;

struct bfd_line_info;
struct dwarf_line_info;

struct address_range
{
    TIMEMORY_DEFAULT_OBJECT(address_range)

    address_range(uintptr_t _v);
    address_range(uintptr_t _low, uintptr_t _high);

    bool contains(uintptr_t) const;
    bool contains(address_range) const;
    bool overlaps(address_range) const;
    bool contiguous_with(address_range) const;

    bool operator==(address_range _v) const;
    bool operator!=(address_range _v) const { return !(*this == _v); }
    bool operator<(address_range _v) const;

    bool        is_range() const;
    std::string as_string(int _depth = 0) const;

    uintptr_t low  = 0x0;
    uintptr_t high = 0x0;
};

namespace basic
{
struct line_info
{
    bool         inlined = false;
    unsigned int line    = 0;
    uintptr_t    address = 0x0;
    std::string  file    = {};
    std::string  func    = {};

    bool         is_valid() const { return !file.empty() && address > 0; }
    hash_value_t hash() const;
    std::string  name() const;

    friend bool operator<(const line_info& _lhs, const line_info& _rhs)
    {
        return _lhs.address < _rhs.address;
    }

    friend bool operator==(const line_info& _lhs, const line_info& _rhs)
    {
        return std::tie(_lhs.address, _lhs.line, _lhs.file, _lhs.func) ==
               std::tie(_rhs.address, _rhs.line, _rhs.file, _rhs.func);
    }

    friend bool operator!=(const line_info& _lhs, const line_info& _rhs)
    {
        return !(_lhs == _rhs);
    }

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned int);
};

inline hash_value_t
line_info::hash() const
{
    return tim::get_combined_hash_id(hash_value_t{ address }, line, file, func);
}

inline std::string
line_info::name() const
{
    if(!file.empty()) return (line > 0) ? JOIN(':', file, line) : file;
    return func;
}

template <typename ArchiveT>
void
line_info::serialize(ArchiveT& ar, const unsigned int)
{
    ar(tim::cereal::make_nvp("address", address), tim::cereal::make_nvp("file", file),
       tim::cereal::make_nvp("line", line), tim::cereal::make_nvp("func", func));
}
}  // namespace basic

struct bfd_line_info
{
    unsigned int priority = 0;
    unsigned int line     = 0;
    union
    {
        uintptr_t     address;
        address_range range = {};
    };
    std::string file = {};
    std::string func = {};

    bool             is_inlined() const;
    bool             is_valid() const;
    basic::line_info get_basic() const;

    friend bool operator<(const bfd_line_info& _lhs, const bfd_line_info& _rhs)
    {
        return (_lhs.address == _rhs.address) ? (_lhs.priority < _rhs.priority)
                                              : (_lhs.address < _rhs.address);
    }

    friend bool operator==(const bfd_line_info& _lhs, const bfd_line_info& _rhs)
    {
        return std::tie(_lhs.address, _lhs.priority, _lhs.line, _lhs.file, _lhs.func) ==
               std::tie(_rhs.address, _rhs.priority, _rhs.line, _rhs.file, _rhs.func);
    }

    friend bool operator!=(const bfd_line_info& _lhs, const bfd_line_info& _rhs)
    {
        return !(_lhs == _rhs);
    }
};

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

    bool             is_valid() const;
    basic::line_info get_basic() const;

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
};

template <typename DataT>
struct line_info
{
    using value_type = DataT;

    std::shared_ptr<bfd_file> file = {};
    std::deque<value_type>    data = {};

    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    void sort();
};

namespace ext
{
using line_info = std::variant<bfd_line_info, dwarf_line_info>;
}

using ext_line_info   = line_info<ext::line_info>;
using basic_line_info = line_info<basic::line_info>;
template <typename Tp = code_object::ext_line_info>
using line_info_map = std::map<code_object::procfs::maps, Tp>;

line_info_map<ext_line_info>
get_ext_line_info(line_info_map<ext_line_info>* _discarded = nullptr);

line_info_map<basic_line_info>
get_basic_line_info(line_info_map<basic_line_info>* _discarded = nullptr);
}  // namespace code_object
}  // namespace omnitrace
