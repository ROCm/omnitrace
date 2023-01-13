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

#include "library/config.hpp"
#include "library/state.hpp"

#if !defined(TIMEMORY_USE_BFD)
#    error "BFD support not enabled"
#endif

#define PACKAGE "omnitrace-bfd"
#define DO_NOT_DEFINE_AOUTHDR
#define DO_NOT_DEFINE_FILHDR
#define DO_NOT_DEFINE_LINENO
#define DO_NOT_DEFINE_SCNHDR

#include <bfd.h>
#include <coff/external.h>
#include <coff/internal.h>
#include <cstdio>
#include <dwarf.h>
#include <elf-bfd.h>
#include <elfutils/libdw.h>
#include <libcoff.h>

#include "library/code_object.hpp"
#include "library/common.hpp"
#include "library/debug.hpp"
#include "library/utility.hpp"

#include <timemory/log/macros.hpp>
#include <timemory/unwind/bfd.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <regex>
#include <set>
#include <stdexcept>

namespace omnitrace
{
namespace code_object
{
namespace
{
using dwarf_line_info_vec_t = std::deque<dwarf_line_info>;

template <typename Tp>
struct is_variant : std::false_type
{};

template <typename... Tp>
struct is_variant<std::variant<Tp...>> : std::true_type
{};

using utility::combine;

auto
get_dwarf_address_ranges(Dwarf_Die* _die)
{
    auto _ranges = std::vector<address_range>{};

    if(dwarf_tag(_die) != DW_TAG_compile_unit) return _ranges;

    Dwarf_Addr _low_pc;
    Dwarf_Addr _high_pc;
    dwarf_lowpc(_die, &_low_pc);
    dwarf_highpc(_die, &_high_pc);

    _ranges.emplace_back(address_range{ _low_pc, _high_pc });

    Dwarf_Addr _base_addr;
    ptrdiff_t  _offset = 0;
    do
    {
        _ranges.emplace_back(address_range{ 0, 0 });
    } while((_offset = dwarf_ranges(_die, _offset, &_base_addr, &_ranges.back().low,
                                    &_ranges.back().high)) > 0);
    // will always have one extra
    _ranges.pop_back();

    return _ranges;
}

auto
get_dwarf_line_info(Dwarf_Die* _die)
{
    auto _line_info = dwarf_line_info_vec_t{};

    if(dwarf_tag(_die) != DW_TAG_compile_unit) return _line_info;

    Dwarf_Lines* _lines     = nullptr;
    size_t       _num_lines = 0;
    if(dwarf_getsrclines(_die, &_lines, &_num_lines) == 0)
    {
        _line_info.resize(_num_lines);
        for(size_t j = 0; j < _num_lines; ++j)
        {
            auto& itr   = _line_info.at(j);
            auto* _line = dwarf_onesrcline(_lines, j);
            if(_line)
            {
                int _lineno = 0;
                itr.valid   = true;
                dwarf_lineno(_line, &_lineno);
                dwarf_linecol(_line, &itr.col);
                dwarf_linebeginstatement(_line, &itr.begin_statement);
                dwarf_lineendsequence(_line, &itr.end_sequence);
                dwarf_lineblock(_line, &itr.line_block);
                dwarf_lineepiloguebegin(_line, &itr.epilogue_begin);
                dwarf_lineprologueend(_line, &itr.prologue_end);
                dwarf_lineisa(_line, &itr.isa);
                dwarf_linediscriminator(_line, &itr.discriminator);
                dwarf_lineaddr(_line, &itr.address);
                if(_lineno > 0) itr.line = _lineno;
                const auto* _file = dwarf_linesrc(_line, nullptr, nullptr);
                if(!_file) _file = dwarf_diename(_die);
                itr.file = filepath::realpath(_file, nullptr, false);
            }
        }
    }

    return _line_info;
}

auto
process_dwarf(int _fd, std::vector<address_range>& _ranges)
{
    auto* _dwarf_v   = dwarf_begin(_fd, DWARF_C_READ);
    auto  _line_info = dwarf_line_info_vec_t{};

    size_t    cu_header_size = 0;
    Dwarf_Off cu_off         = 0;
    Dwarf_Off next_cu_off    = 0;
    for(; dwarf_nextcu(_dwarf_v, cu_off, &next_cu_off, &cu_header_size, nullptr, nullptr,
                       nullptr) == 0;
        cu_off = next_cu_off)
    {
        Dwarf_Off cu_die_off = cu_off + cu_header_size;
        Dwarf_Die cu_die;
        if(dwarf_offdie(_dwarf_v, cu_die_off, &cu_die) != nullptr)
        {
            Dwarf_Die* _die = &cu_die;
            if(dwarf_tag(_die) == DW_TAG_compile_unit)
            {
                combine(_line_info, get_dwarf_line_info(_die));
                combine(_ranges, get_dwarf_address_ranges(_die));
            }
        }
    }

    dwarf_end(_dwarf_v);
    utility::filter_sort_unique(_line_info);
    utility::filter_sort_unique(_ranges);

    return _line_info;
}

void
read_inliner_info(bfd* _inp, std::vector<bfd_line_info>& _data, bfd_vma _pc,
                  unsigned int& _prio)
{
    bool        _found = false;
    const char* _file  = nullptr;
    const char* _func  = nullptr;
    while(_found)
    {
        auto _info     = bfd_line_info{};
        _info.address  = address_range_t{ _pc };  // inliner info is not over range
        _info.priority = _prio++;
        _found         = (bfd_find_inliner_info(_inp, &_file, &_func, &_info.line) != 0);

        if(_found)
        {
            if(_file) _info.file = _file;
            if(_func) _info.func = _func;
            if(!_file || (_file && strnlen(_file, 1) == 0))
                _info.file = bfd_get_filename(_inp);
            _info.file = filepath::realpath(_info.file, nullptr, false);
            _data.emplace_back(_info);
        }
        else
            break;
    }
}

auto
read_pc(bfd_file& _bfd, asection* _section, bfd_vma _pc, bfd_vma _pc_len = 0)
{
    auto          _data = std::vector<bfd_line_info>{};
    bfd_vma       _vma  = bfd_section_vma(_section);
    bfd_size_type _size = bfd_section_size(_section);

    if(_pc < _vma || _pc >= _vma + _size) return _data;
    if(_pc + _pc_len > _vma + _size) _pc_len = (_vma + _size) - _pc;

    auto* _inp  = static_cast<bfd*>(_bfd.data);
    auto* _syms = reinterpret_cast<asymbol**>(_bfd.syms);

    {
        auto         _info          = bfd_line_info{};
        unsigned int _prio          = 0;
        _info.address               = address_range{ _pc, _pc + _pc_len };
        _info.priority              = _prio++;
        const char*  _file          = nullptr;
        const char*  _func          = nullptr;
        unsigned int _discriminator = 0;

        if(bfd_find_nearest_line_discriminator(_inp, _section, _syms, _info.low() - _vma,
                                               &_file, &_func, &_info.line,
                                               &_discriminator) != 0)
        {
            if(_file) _info.file = _file;
            if(_func) _info.func = _func;
            if(!_file || (_file && strnlen(_file, 1) == 0))
                _info.file = bfd_get_filename(_inp);
            _info.file = filepath::realpath(_info.file, nullptr, false);
            if(_info)
            {
                _data.emplace_back(_info);
                if(config::get_sampling_include_inlines())
                    read_inliner_info(_inp, _data, _pc, _prio);
            }
        }
    }

    {
        auto         _info = bfd_line_info{};
        unsigned int _prio = 0;
        _info.address      = address_range{ _pc, _pc + _pc_len };
        _info.priority     = _prio++;
        const char* _file  = nullptr;
        const char* _func  = nullptr;

        if(bfd_find_nearest_line(_inp, _section, _syms, _info.low() - _vma, &_file,
                                 &_func, &_info.line) != 0)
        {
            if(_file) _info.file = _file;
            if(_func) _info.func = _func;
            if(!_file || (_file && strnlen(_file, 1) == 0))
                _info.file = bfd_get_filename(_inp);
            _info.file = filepath::realpath(_info.file, nullptr, false);
            if(_info)
            {
                _data.emplace_back(_info);
                if(config::get_sampling_include_inlines())
                    read_inliner_info(_inp, _data, _pc, _prio);
            }
        }
    }

    utility::filter_sort_unique(_data);

    return _data;
}

auto
parse_line_info(const std::string& _fname)
{
    auto _bfd = std::make_shared<bfd_file>(_fname);

    OMNITRACE_VERBOSE(0, "Reading line info for '%s'...\n", _fname.c_str());

    auto _info = ext_line_info{};
    _info.file = _bfd;
    if(_bfd && _bfd->is_good())
    {
        auto _section_map = std::map<address_range, asection*>{};
        auto _section_set = std::set<asection*>{};
        auto _processed   = std::set<uintptr_t>{};
        for(auto&& itr : _bfd->get_symbols())
        {
            if(itr.symsize == 0) continue;
            auto* _section = static_cast<asection*>(itr.section);
            _section_set.emplace(_section);
            _processed.emplace(itr.address);
            _info.ranges.emplace_back(
                address_range{ itr.address, itr.address + itr.symsize });
            for(auto&& ditr : read_pc(*_bfd, _section, itr.address, itr.symsize))
            {
                ditr.weak = (itr.binding == tim::unwind::bfd_binding::Weak);
                _info.data.emplace_back(ditr);
            }
        }

        for(auto* itr : _section_set)
        {
            auto*         _section     = const_cast<asection*>(itr);
            bfd_vma       _section_vma = bfd_section_vma(_section);
            bfd_size_type _section_len = bfd_section_size(_section);
            auto          _section_range =
                address_range{ _section_vma, _section_vma + _section_len };
            _section_map[_section_range] = _section;
        }

        TIMEMORY_REQUIRE(_section_set.size() == _section_map.size())
            << "section set size (" << _section_set.size() << ") != section map size ("
            << _section_map.size() << ")\n";

        _info.sort();

        for(auto&& itr : process_dwarf(_bfd->fd, _info.ranges))
        {
            if(itr.address == 0) continue;
            if(_processed.count(itr.address) > 0) continue;
            auto      _filename = filepath::realpath(itr.file);
            asection* _section  = nullptr;
            for(const auto& sitr : _section_map)
            {
                if(sitr.first.contains(itr.address))
                {
                    _section = sitr.second;
                    break;
                }
            }

            if(!_section) continue;

            TIMEMORY_REQUIRE(_section != nullptr)
                << "bfd section not found for address " << as_hex(itr.address) << "\n";

            for(auto&& ditr : read_pc(*_bfd, _section, itr.address, 0))
            {
                // dwarf line info is more accurate than bfd
                /*
                if(_filename == itr.file && ditr.line != itr.line)
                {
                    auto _ditr = ditr;
                    _ditr.line = itr.line;
                    _info.data.emplace_back(_ditr);
                    ++ditr.priority;
                }*/
                _info.data.emplace_back(ditr);
            }

            _processed.emplace(itr.address);
        }
        _info.sort();
    }

    OMNITRACE_VERBOSE(1, "Reading line info for '%s'... %zu entries\n", _fname.c_str(),
                      _info.data.size());

    return _info;
}
}  // namespace

address_range::address_range(uintptr_t _v)
: low{ _v }
, high{ _v }
{}

address_range::address_range(uintptr_t _low, uintptr_t _high)
: low{ _low }
, high{ _high }
{
    TIMEMORY_REQUIRE(high >= low) << "Error! address_range high must be >= low\n";
}

bool
address_range::is_range() const
{
    return (low < high);
}

std::string
address_range::as_string(int _depth) const
{
    std::stringstream _ss{};
    _ss << std::hex;
    _ss << std::setw(2 * _depth) << "";
    _ss.fill('0');
    _ss << "0x" << std::setw(16) << low << "-"
        << "0x" << std::setw(16) << high;
    return _ss.str();
}

uintptr_t
address_range::size() const
{
    return (low == high) ? 1 : (high > low) ? (high - low + 1) : (low - high + 1);
}

bool
address_range::is_valid() const
{
    return (low <= high && (low + 1) > 1);
}

bool
address_range::contains(uintptr_t _v) const
{
    return (is_range()) ? (low <= _v && high > _v) : (_v == low);
}

bool
address_range::contains(address_range _v) const
{
    return (*this == _v) || (contains(_v.low) && contains(_v.high));
}

bool
address_range::overlaps(address_range _v) const
{
    if(contains(_v)) return false;
    int64_t _lhs_diff = (high - low);
    int64_t _rhs_diff = (_v.high - _v.low);
    int64_t _diff     = (std::max(high, _v.high) - std::min(low, _v.low));
    return (_diff < (_lhs_diff + _rhs_diff));
}

bool
address_range::contiguous_with(address_range _v) const
{
    return (_v.low == high || low == _v.high);
}

bool
address_range::operator==(address_range _v) const
{
    // if arg is range and this is not range, call this function with arg
    // if(_v.is_range() && !is_range()) return false;
    // check if arg is in range
    // if(is_range() && !_v.is_range()) return false;
    // both are ranges or both are just address
    return std::tie(low, high) == std::tie(_v.low, _v.high);
}

bool
address_range::operator<(address_range _v) const
{
    if(is_range() && !_v.is_range())
    {
        return (low == _v.low) ? true : (low < _v.low);
    }
    else if(!is_range() && _v.is_range())
    {
        return (low == _v.low) ? false : (low < _v.low);
    }
    else if(!is_range() && !_v.is_range())
    {
        return (low < _v.low);
    }
    return std::tie(low, high) < std::tie(_v.low, _v.high);
    // if(_v.low == _v.high && _v.low >= low && _v.low < high) return false;
    // return (low == _v.low) ? (high > _v.high) : (low < _v.low);
}

bool
address_range::operator>(address_range _v) const
{
    return !(*this < _v) && !(*this == _v);
}

address_range&
address_range::operator+=(uintptr_t _v)
{
    if(is_valid())
    {
        low += _v;
        high += _v;
    }
    else
    {
        low  = _v;
        high = _v;
    }
    return *this;
}

address_range&
address_range::operator-=(uintptr_t _v)
{
    if(is_valid())
    {
        low -= _v;
        high -= _v;
    }
    else
    {
        low  = _v;
        high = _v;
    }
    return *this;
}

hash_value_t
address_range::hash() const
{
    return (is_range()) ? tim::get_combined_hash_id(hash_value_t{ low }, high)
                        : hash_value_t{ low };
}

bool
bfd_line_info::is_inlined() const
{
    return priority > 0;
}

bool
bfd_line_info::is_weak() const
{
    return weak;
}

bool
bfd_line_info::is_valid() const
{
    return !file.empty() || !func.empty();
}

basic::line_info
bfd_line_info::get_basic() const
{
    return basic::line_info{ is_weak(), is_inlined(), line, address, file, func };
}

bool
dwarf_line_info::is_valid() const
{
    return valid && !file.empty();
}

basic::line_info
dwarf_line_info::get_basic() const
{
    return basic::line_info{ false, false,        line, address_range_t{ address },
                             file,  std::string{} };
}

template <typename DataT>
void
line_info<DataT>::sort()
{
    utility::filter_sort_unique(data);
    utility::filter_sort_unique(ranges);
}

std::map<procfs::maps, ext_line_info>
get_ext_line_info(const std::vector<scope_filter>&       _filters,
                  std::map<procfs::maps, ext_line_info>* _discarded)
{
    auto _get_chain = [](const char* _name) {
        void* _handle = dlopen(_name, RTLD_LAZY | RTLD_NOLOAD);
        auto  _chain  = std::set<std::string>{};
        if(_handle)
        {
            struct link_map* _link_map = nullptr;
            dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
            struct link_map* _next = _link_map;
            while(_next)
            {
                if(_name == nullptr && _next == _link_map &&
                   std::string_view{ _next->l_name }.empty())
                {
                    // only insert exe name if dlopened the exe and
                    // empty name is first entry
                    _chain.emplace(config::get_exe_realpath());
                }
                else if(!std::string_view{ _next->l_name }.empty())
                {
                    _chain.emplace(_next->l_name);
                }
                _next = _next->l_next;
            }
        }
        return _chain;
    };

    auto _add_basenames = [](auto& _data) {
        auto _tmp = _data;
        for(auto&& itr : _tmp)
        {
            _data.emplace(filepath::basename(itr));
            _data.emplace(filepath::realpath(itr, nullptr, false));
        }
        return _data;
    };

    auto _full_chain = _get_chain(nullptr);            // all libraries loaded
    auto _omni_chain = _get_chain("libomnitrace.so");  // libraries loaded by omnitrace
    auto _main_chain = std::set<std::string>{};        // libraries loaded by main program

    for(const auto& itr : _full_chain)
    {
        if(_omni_chain.find(itr) == _omni_chain.end()) _main_chain.emplace(itr);
    }

    for(const auto& itr : _main_chain)
    {
        OMNITRACE_VERBOSE(2, "[library][%s]: %s\n", config::get_exe_realpath().c_str(),
                          itr.c_str());
    }

    for(const auto& itr : _omni_chain)
    {
        OMNITRACE_VERBOSE(3, "[library][libomnitrace]: %s\n", itr.c_str());
    }

    _add_basenames(_main_chain);
    _add_basenames(_omni_chain);

    auto _satisfies_filter = [&_filters](auto _scope, const std::string& _value) {
        for(const auto& itr : _filters)  // NOLINT
        {
            // if the filter is for the specified scope and itr does not satisfy the
            // include/exclude mode, return false
            if((itr.scope & _scope) == _scope && !itr(_value)) return false;
        }
        return true;
    };

    auto _satisfies_binary_filter = [&_satisfies_filter](const std::string& _value) {
        return _satisfies_filter(scope_filter::BINARY_FILTER, _value);
    };

    auto _satisfies_source_filter = [&_satisfies_filter](const std::string& _value) {
        return _satisfies_filter(scope_filter::SOURCE_FILTER, _value);
    };

    auto _satisfies_function_filter = [&_satisfies_filter](const std::string& _value) {
        return _satisfies_filter(scope_filter::FUNCTION_FILTER, _value);
    };

    // filter function used by procfs::get_contiguous_maps
    // ensures that we do not process omnitrace/gotcha/libunwind libraries
    // and do not process the libraries outside of the binary scope
    auto _filter = [&_satisfies_binary_filter](const procfs::maps& _v) {
        if(_v.pathname.empty()) return false;
        auto _path = filepath::realpath(_v.pathname, nullptr, false);
        return (filepath::exists(_path) &&
                //_main_chain.find(filepath::basename(_path)) != _main_chain.end() &&
                _satisfies_binary_filter(_path));
    };

    // get the memory maps and create a unique set of filenames
    auto _maps  = procfs::get_contiguous_maps(process::get_id(), _filter, true);
    auto _files = std::set<std::string>{};
    for(const auto& itr : _maps)
        _files.emplace(filepath::realpath(itr.pathname, nullptr, false));

    // get the line info for the set of files
    auto _fdata = std::map<std::string, ext_line_info>{};
    for(auto itr : _files)
    {
        if(_main_chain.count(itr) > 0 || _main_chain.count(basename(itr.c_str())) > 0)
            _fdata.emplace(itr, parse_line_info(itr));
    }

    // sort the line info into one map which contains only the line info
    // in the source scope and another map which contains the line info
    // outside the source scope
    auto _data = std::map<procfs::maps, ext_line_info>{};
    for(const auto& mitr : _maps)
    {
        auto mrange = address_range_t{ mitr.load_address, mitr.last_address };
        auto itr    = _fdata.find(mitr.pathname);
        if(itr != _fdata.end())
        {
            // create an empty entry in data and discarded
            _data.emplace(mitr, ext_line_info{ itr->second.file });
            if(_discarded) _discarded->emplace(mitr, ext_line_info{ itr->second.file });
            for(const auto& vitr : itr->second.data)
            {
                const auto& _file = vitr.file;
                const auto& _func = vitr.func;
                if(!_satisfies_source_filter(_file) ||
                   (!_satisfies_function_filter(_func) &&
                    !_satisfies_function_filter(demangle(_func))))
                {
                    // insert into discarded if does not match source scope regex
                    if(_discarded) _discarded->at(mitr).data.emplace_back(vitr);
                }
                else
                {
                    auto _addr = mitr.load_address + vitr.address;
                    if(OMNITRACE_LIKELY(mrange.contains(_addr)))
                        _data.at(mitr).data.emplace_back(vitr);
                    else if(_discarded)
                    {
                        // this is potentially an error. may need to emit a warning
                        _discarded->at(mitr).data.emplace_back(vitr);
                    }
                }
            }
            _data.at(mitr).sort();
            if(_discarded) _discarded->at(mitr).sort();
        }
    }

    auto _patch_line_info = [&](auto& _info) {
        /*
        auto _starting_ranges = std::vector<const bfd_line_info*>{};
        for(auto& litr : _info)
        {
            for(auto& ditr : litr.second)
            {
                if(ditr.address.is_range())
                    _starting_ranges.emplace_back(&ditr);
            }
        }*/
        for(auto& litr : _info)
        {
            const code_object::bfd_line_info* _curr_range = nullptr;
            for(auto ditr = litr.second.begin(); ditr != litr.second.end(); ++ditr)
            {
                // ranges are ordered first so store this for future exact entries
                if(ditr->address.is_range())
                    _curr_range = &*ditr;
                else
                {
                    auto nitr = ditr + 1;
                    // if at end of entire range, set to the end of last range
                    if(nitr == litr.second.end())
                    {
                        if(_curr_range) ditr->address.high = _curr_range->address.high;
                    }
                    else
                    {
                        ditr->address.high =
                            (_curr_range)
                                ? std::min(_curr_range->address.high, nitr->address.low)
                                : nitr->address.low;
                    }
                }
            }
        }
    };

    _patch_line_info(_data);
    if(_discarded) _patch_line_info(*_discarded);

    return _data;
}

std::map<procfs::maps, basic_line_info>
get_basic_line_info(const std::vector<scope_filter>&         _filters,
                    std::map<procfs::maps, basic_line_info>* _discarded)
{
    // convert the extended line info format into the basic line info format
    using ext_line_info_t   = std::map<procfs::maps, ext_line_info>;
    using basic_line_info_t = std::map<procfs::maps, basic_line_info>;
    auto _ext_discarded     = ext_line_info_t{};
    auto _ext_data          = get_ext_line_info(_filters, &_ext_discarded);

    auto _data = basic_line_info_t{};
    for(auto& itr : _ext_data)
    {
        _data.emplace(itr.first, basic_line_info{ itr.second.file });
        auto& _data_v = _data.at(itr.first).data;
        for(const auto& ditr : itr.second.data)
            _data_v.emplace_back(ditr.get_basic());
    }

    if(_discarded)
    {
        for(auto& itr : _ext_discarded)
        {
            _discarded->emplace(itr.first, basic_line_info{ itr.second.file });
            auto& _data_v = _discarded->at(itr.first).data;
            for(const auto& ditr : itr.second.data)
                _data_v.emplace_back(ditr.get_basic());
        }
    }

    return _data;
}
}  // namespace code_object
}  // namespace omnitrace
