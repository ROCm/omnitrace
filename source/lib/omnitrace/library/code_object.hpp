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

#include "common/defines.h"
#include "library/common.hpp"
#include "library/defines.hpp"
#include "library/exception.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/unwind/bfd.hpp>
#include <timemory/unwind/types.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <regex>
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
    // set to low to max and high to min to support std::min(...)
    // and std::max(...) assignment
    uintptr_t low  = std::numeric_limits<uintptr_t>::max();
    uintptr_t high = std::numeric_limits<uintptr_t>::min();

    TIMEMORY_DEFAULT_OBJECT(address_range)

    explicit address_range(uintptr_t _v);
    address_range(uintptr_t _low, uintptr_t _high);

    bool contains(uintptr_t) const;
    bool contains(address_range) const;
    bool overlaps(address_range) const;
    bool contiguous_with(address_range) const;

    bool operator==(address_range _v) const;
    bool operator!=(address_range _v) const { return !(*this == _v); }
    bool operator<(address_range _v) const;
    bool operator>(address_range _v) const;

    address_range& operator+=(uintptr_t);
    address_range& operator-=(uintptr_t);

    bool         is_range() const;
    hash_value_t hash() const;
    std::string  as_string(int _depth = 0) const;
    bool         is_valid() const;
    uintptr_t    size() const;
    explicit     operator bool() const { return is_valid(); }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned)
    {
        ar(cereal::make_nvp("low", low));
        ar(cereal::make_nvp("high", high));
    }
};

namespace basic
{
struct line_info
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
    return tim::get_combined_hash_id(address.hash(), line, file, func);
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
    ar(cereal::make_nvp("address", address), cereal::make_nvp("file", file),
       cereal::make_nvp("line", line), cereal::make_nvp("func", func));
    if constexpr(concepts::is_output_archive<ArchiveT>::value)
        ar(cereal::make_nvp("dfunc", demangle(func)));
    ar(cereal::make_nvp("inlined", inlined));
}
}  // namespace basic

struct bfd_line_info
{
    bool          weak     = false;
    unsigned int  priority = 0;
    unsigned int  line     = 0;
    address_range address  = {};
    std::string   file     = {};
    std::string   func     = {};

    bool     is_inlined() const;
    bool     is_weak() const;
    bool     is_valid() const;
    explicit operator bool() const { return is_valid(); }

    basic::line_info get_basic() const;
    auto             low() const { return address.low; }
    auto             high() const { return address.high; }

    friend bool operator<(const bfd_line_info& _lhs, const bfd_line_info& _rhs)
    {
        return (_lhs.address == _rhs.address) ? (_lhs.priority < _rhs.priority)
                                              : (_lhs.address < _rhs.address);
    }

    friend bool operator==(const bfd_line_info& _lhs, const bfd_line_info& _rhs)
    {
        return std::tie(_lhs.address, _lhs.line, _lhs.file, _lhs.func) ==
               std::tie(_rhs.address, _rhs.line, _rhs.file, _rhs.func);
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

    bool     is_valid() const;
    explicit operator bool() const { return is_valid(); }

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

    std::shared_ptr<bfd_file>  file   = {};
    std::deque<value_type>     data   = {};
    std::vector<address_range> ranges = {};

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
};

struct scope_filter
{
    enum filter_mode : uint8_t
    {
        FILTER_INCLUDE = 0,
        FILTER_EXCLUDE
    };

    enum filter_scope : uint8_t
    {
        UNIVERSAL_FILTER = (1 << 0),
        BINARY_FILTER    = (1 << 1),
        SOURCE_FILTER    = (1 << 2),
        FUNCTION_FILTER  = (1 << 3),
        FILELINE_FILTER  = (1 << 4)
    };

    filter_mode  mode       = FILTER_INCLUDE;
    filter_scope scope      = UNIVERSAL_FILTER;
    std::string  expression = {};

    bool operator()(const std::string& _value) const
    {
        if(mode == FILTER_INCLUDE)
            return (expression.empty())
                       ? true
                       : std::regex_search(_value, std::regex{ expression });
        else if(mode == FILTER_EXCLUDE)
            return (expression.empty())
                       ? false
                       : !std::regex_search(_value, std::regex{ expression });
        throw exception<std::runtime_error>{ "invalid scope filter mode" };
    }
};

namespace ext
{
using line_info = bfd_line_info;
}

using ext_line_info   = line_info<ext::line_info>;
using basic_line_info = line_info<basic::line_info>;
template <typename Tp = code_object::ext_line_info>
using line_info_map = std::map<code_object::procfs::maps, Tp>;

line_info_map<ext_line_info>
get_ext_line_info(const std::vector<scope_filter>&,
                  line_info_map<ext_line_info>* _discarded = nullptr);

line_info_map<basic_line_info>
get_basic_line_info(const std::vector<scope_filter>&,
                    line_info_map<basic_line_info>* _discarded = nullptr);
}  // namespace code_object

inline code_object::address_range
operator+(code_object::address_range _lhs, uintptr_t _v)
{
    return (_lhs += _v);
}

inline code_object::address_range
operator+(uintptr_t _v, code_object::address_range _lhs)
{
    return (_lhs += _v);
}
}  // namespace omnitrace

namespace std
{
template <>
struct hash<::omnitrace::code_object::address_range>
{
    using address_range_t = ::omnitrace::code_object::address_range;

    auto operator()(const address_range_t& _v) const { return _v.hash(); }
    auto operator()(address_range_t&& _v) const { return _v.hash(); }
};
}  // namespace std
