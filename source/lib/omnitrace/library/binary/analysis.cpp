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
auto
parse_line_info(const std::string& _fname)
{
    auto _bfd = std::make_shared<bfd_file>(_fname);

    OMNITRACE_VERBOSE(0, "Reading line info for '%s'...\n", _fname.c_str());

    auto _info = line_info<bfd_line_info>{};
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
            for(auto&& ditr :
                bfd_line_info::process_bfd(*_bfd, _section, itr.address, itr.symsize))
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

        for(auto&& itr : dwarf_line_info::process_dwarf(_bfd->fd, _info.ranges))
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

            for(auto&& ditr : bfd_line_info::process_bfd(*_bfd, _section, itr.address, 0))
            {
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

line_info_t
get_line_info(const std::vector<scope_filter>& _filters, line_info_t* _discarded)
{
    using value_type = typename line_info_t::mapped_type;

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

    auto _satisfies_fileline_filter = [&_satisfies_filter](const std::string& _value) {
        return _satisfies_filter(scope_filter::FILELINE_FILTER, _value);
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
    auto _fdata = std::map<std::string, value_type>{};
    for(auto itr : _files)
    {
        if(_main_chain.count(itr) > 0 || _main_chain.count(basename(itr.c_str())) > 0)
            _fdata.emplace(itr, parse_line_info(itr));
    }

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
            for(const auto& vitr : itr->second.data)
            {
                const auto& _file = vitr.file;
                const auto& _func = vitr.func;
                const auto& _line = vitr.line;
                if(!_satisfies_source_filter(_file) ||
                   !_satisfies_function_filter(demangle(_func)) ||
                   !_satisfies_fileline_filter(join(":", _file, _line)))
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
    };

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

    _patch_line_info(_data);
    _combine_line_info(_data);

    if(_discarded) _patch_line_info(*_discarded);

    return _data;
}
}  // namespace binary
}  // namespace omnitrace
