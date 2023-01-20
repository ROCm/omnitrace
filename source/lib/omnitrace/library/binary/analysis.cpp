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

#if !defined(TIMEMORY_USE_BFD)
#    error "BFD support not enabled"
#endif

#define PACKAGE "omnitrace-binary-analysis"
#define DO_NOT_DEFINE_AOUTHDR
#define DO_NOT_DEFINE_FILHDR
#define DO_NOT_DEFINE_LINENO
#define DO_NOT_DEFINE_SCNHDR

#include <bfd.h>

#include "library/binary/address_range.hpp"
#include "library/binary/analysis.hpp"
#include "library/binary/basic_line_info.hpp"
#include "library/binary/bfd_line_info.hpp"
#include "library/binary/dwarf_line_info.hpp"
#include "library/binary/line_info.hpp"
#include "library/binary/scope_filter.hpp"
#include "library/common.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/state.hpp"
#include "library/utility.hpp"

#include <timemory/log/macros.hpp>
#include <timemory/unwind/bfd.hpp>
#include <timemory/unwind/dlinfo.hpp>
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
namespace binary
{
namespace
{
namespace unwind = ::tim::unwind;

auto
parse_line_info(const std::string& _fname)
{
    auto _bfd = std::make_shared<bfd_file>(_fname);

    OMNITRACE_VERBOSE(0, "[binary] Reading line info for '%s'...\n", _fname.c_str());

    auto _info = line_info<bfd_line_info>{};
    _info.file = _bfd;
    if(_bfd && _bfd->is_good())
    {
        auto& _section_map = _info.sections;
        auto  _section_set = std::set<asection*>{};
        auto  _processed   = std::set<uintptr_t>{};
        for(auto&& itr : _bfd->get_symbols())
        {
            if(itr.symsize == 0) continue;
            auto* _section = static_cast<asection*>(itr.section);
            _section_set.emplace(_section);
            _processed.emplace(itr.address);
            _info.ranges.emplace_back(
                address_range{ itr.address, itr.address + itr.symsize });
            for(auto&& ditr :
                bfd_line_info::process_bfd(*_bfd, _section, itr.address, itr.symsize))
            {
                ditr.weak = (itr.binding == unwind::bfd_binding::Weak);
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

        for(auto&& itr : dwarf_line_info::process_dwarf(_bfd->fd, _info.ranges))
        {
            if(itr.address == 0) continue;
            if(_processed.count(itr.address) > 0) continue;
            auto      _filename = filepath::realpath(itr.file);
            asection* _section  = _info.find_section<asection>(itr.address);

            if(!_section) continue;

            for(auto&& ditr : bfd_line_info::process_bfd(*_bfd, _section, itr.address, 0))
                _info.data.emplace_back(ditr);

            _processed.emplace(itr.address);
        }

        _info.sort();
    }

    OMNITRACE_VERBOSE(1, "[binary] Reading line info for '%s'... %zu entries\n",
                      _fname.c_str(), _info.data.size());

    return _info;
}
}  // namespace

line_info_t
get_line_info(const std::vector<std::string>&  _files,
              const std::vector<scope_filter>& _filters, line_info_t* _discarded)
{
    using value_type = typename line_info_t::mapped_type;

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

    auto _satisfies_source_filter = [&_satisfies_filter](const std::string& _value,
                                                         int32_t            _line) {
        return _satisfies_filter(scope_filter::SOURCE_FILTER, _value) ||
               _satisfies_filter(scope_filter::SOURCE_FILTER, join(":", _value, _line));
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
        return (filepath::exists(_path) && _satisfies_binary_filter(_path));
    };

    // get the line info for the set of files
    auto _fdata = std::map<std::string, value_type>{};
    for(auto itr : _files)
    {
        if(filepath::exists(itr) && _satisfies_binary_filter(itr))
            _fdata.emplace(itr, parse_line_info(itr));
    }

    // get the memory maps
    auto _maps = procfs::get_contiguous_maps(process::get_id(), _filter, true);

    for(const auto& mitr : _maps)
    {
        auto itr = _fdata.find(mitr.pathname);
        if(itr != _fdata.end())
        {
            uintptr_t _start     = mitr.load_address;
            auto      _gap_addrs = std::vector<uintptr_t>{};
            for(auto& ditr : itr->second)
            {
                if(_start < ditr.address.low)
                {
                    auto _len = ditr.address.low - _start;
                    _gap_addrs.reserve(_gap_addrs.size() + _len);
                    for(uintptr_t i = _start; i < ditr.address.low; ++i)
                        _gap_addrs.emplace_back(i);
                    _start = ditr.address.high;
                }
            }

            OMNITRACE_WARNING(1, "[analysis] found %zu gaps in line info for %s\n",
                              _gap_addrs.size(), itr->second.file->name.c_str());

            line_info<bfd_line_info>& _bfd_info = itr->second;
            for(auto gitr : _gap_addrs)
            {
                auto* _section = _bfd_info.find_section<asection>(gitr);
                if(_section)
                {
                    for(auto&& ditr :
                        bfd_line_info::process_bfd(*_bfd_info.file, _section, gitr, 0))
                        _bfd_info.data.emplace_back(ditr);
                }
            }
            _bfd_info.sort();
        }
    }

    // environment variable to force not using the load offset
    auto _force_no_offset =
        get_env<bool>("OMNITRACE_BINARY_ANALYSIS_DISABLE_LOAD_ADDRESS_OFFSET", false);

    // sort the line info into one map which contains only the line info
    // in the source scope and another map which contains the line info
    // outside the source scope
    auto _data = std::map<procfs::maps, value_type>{};
    for(const auto& mitr : _maps)
    {
        auto mrange = address_range{ mitr.load_address, mitr.last_address };
        auto itr    = _fdata.find(mitr.pathname);
        if(itr != _fdata.end())
        {
            // create an empty entry in data and discarded
            _data.emplace(mitr, value_type{ itr->second.file });
            if(_discarded) _discarded->emplace(mitr, value_type{ itr->second.file });

            size_t _local_no_offset = 0;
            auto   _local_included  = decltype(std::declval<value_type>().data){};
            auto   _local_excluded  = decltype(std::declval<value_type>().data){};
            for(auto& vitr : itr->second.data)
            {
                const auto& _file = vitr.file;
                const auto& _func = vitr.func;
                const auto& _line = vitr.line;

                auto _addr = mitr.load_address + vitr.address;
                if(mrange.contains(_addr) && !_force_no_offset)
                    vitr.load_address = mitr.load_address;
                else if(vitr.address.low >= mitr.load_address &&
                        vitr.address.high <= mitr.last_address)
                    ++_local_no_offset;
                else
                    continue;

                if(!_satisfies_function_filter(demangle(_func)) ||
                   !_satisfies_source_filter(_file, _line))
                {
                    // store thos which do not satisfy function or source constraints
                    _local_excluded.emplace_back(vitr);
                }
                else
                {
                    _local_included.emplace_back(vitr);
                }
            }

            utility::combine(_data.at(mitr).data, _local_included);
            _data.at(mitr).sort();

            if(_discarded)
            {
                utility::combine(_discarded->at(mitr).data, _local_excluded);
                _discarded->at(mitr).sort();
            }

            auto _max_offsets = (_local_included.size() + _local_excluded.size());
            OMNITRACE_PREFER(_local_no_offset == 0 || _local_no_offset == _max_offsets)
                << "Warning! When mapping binary addresses to instruction pointer "
                   "addresses for "
                << mitr.pathname << ", " << _local_no_offset
                << " addresses were not contained in the address range of the binary ("
                << mrange.as_string() << "). Expected either zero (no ASLR) or "
                << _max_offsets << ".\n"
                << "Please report this issue on GitHub "
                   "(https://github.com/AMDResearch/omnitrace/issues/) with the contents "
                   "of "
                   "/etc/os-release and the two files in the causal/line-info "
                   "subdirectory.\n"
                << "If this value is close to zero, try setting the environment variable "
                   "OMNITRACE_BINARY_ANALYSIS_DISABLE_LOAD_ADDRESS_OFFSET=ON to "
                   "forcefully "
                   "disable "
                   "offsetting the binary address by the load address and inspect the "
                   "line "
                   "info output files afterwards.\n";
        }
        else
        {
            OMNITRACE_WARNING(1, "[analysis] no line info data for '%s'...\n",
                              mitr.pathname.c_str());
        }
    }

    auto _combine_line_info = [&](auto& _info) {
        for(auto& litr : _info)
        {
            auto _tmp = litr.second.data;
            _tmp.clear();
            for(auto ditr = litr.second.begin(); ditr != litr.second.end(); ++ditr)
            {
                if(_tmp.empty())
                {
                    _tmp.emplace_back(*ditr);
                    continue;
                }
                else
                {
                    auto& titr = _tmp.back();
                    if(titr.file == ditr->file && titr.func == ditr->func &&
                       titr.line == ditr->line &&
                       titr.address.contiguous_with(ditr->address))
                    {
                        titr.address += ditr->address;
                    }
                    else
                    {
                        _tmp.emplace_back(*ditr);
                    }
                }
            }
            litr.second.data = std::move(_tmp);
            litr.second.sort();
        }
    };

    auto _patch_line_info = [&](auto& _info) {
        for(auto& litr : _info)
        {
            const bfd_line_info* _curr_range = nullptr;
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
        _combine_line_info(_info);
    };

    _patch_line_info(_data);
    if(_discarded) _patch_line_info(*_discarded);

    return _data;
}

bool
using_load_address_offset()
{
    return false;
}
}  // namespace binary
}  // namespace omnitrace
