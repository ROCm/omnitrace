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

#include <timemory/unwind/bfd.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>

#include <regex>
#include <set>

namespace omnitrace
{
namespace code_object
{
namespace
{
using address_range_t       = std::pair<uintptr_t, uintptr_t>;
using dwarf_line_info_vec_t = std::deque<dwarf_line_info>;

template <typename Tp>
struct is_variant : std::false_type
{};

template <typename... Tp>
struct is_variant<std::variant<Tp...>> : std::true_type
{};

template <typename LhsT, typename RhsT>
inline LhsT&
combine(LhsT& _lhs, RhsT&& _rhs)
{
    for(auto&& itr : _rhs)
        _lhs.emplace_back(itr);
    return _lhs;
}

template <template <typename, typename...> class ContainerT, typename DataT,
          typename... TailT>
inline ContainerT<DataT, TailT...>&
filter_sort_unique(ContainerT<DataT, TailT...>& _v)
{
    if constexpr(is_variant<DataT>::value)
    {
        _v.erase(std::remove_if(_v.begin(), _v.end(),
                                [](const auto& itr) {
                                    return std::visit(
                                        [](const auto& _val) { return !_val.is_valid(); },
                                        itr);
                                }),
                 _v.end());
        std::sort(_v.begin(), _v.end(), [](const auto& _lhs, const auto& _rhs) {
            auto _lhs_addr =
                std::visit([](const auto& _val) { return _val.address; }, _lhs);
            auto _rhs_addr =
                std::visit([](const auto& _val) { return _val.address; }, _rhs);
            return _lhs_addr < _rhs_addr;
        });
    }
    else
    {
        _v.erase(std::remove_if(_v.begin(), _v.end(),
                                [](const auto& itr) { return !itr.is_valid(); }),
                 _v.end());
        std::sort(_v.begin(), _v.end());
    }

    auto _last = std::unique(_v.begin(), _v.end());
    if(std::distance(_v.begin(), _last) > 0) _v.erase(_last, _v.end());
    return _v;
}

auto
read_pc(ext_line_info* _info_vec, asection* _section, bfd_vma _pc)
{
    bfd_vma       _vma  = bfd_section_vma(_section);
    bfd_size_type _size = bfd_section_size(_section);

    if(_pc < _vma || _pc >= _vma + _size) return;

    auto* _inp  = static_cast<bfd*>(_info_vec->file->data);
    auto* _syms = reinterpret_cast<asymbol**>(_info_vec->file->syms);
    auto  _info = bfd_line_info{};
    // auto         _slim  = bfd_get_section_limit(_inp, _section);
    unsigned int _prio  = 0;
    bool         _found = false;
    _info.address       = _pc;
    _info.priority      = _prio++;
    {
        const char*  _file          = nullptr;
        const char*  _func          = nullptr;
        unsigned int _discriminator = 0;
        _found                      = (bfd_find_nearest_line_discriminator(
                      _inp, _section, _syms, _info.address - _vma, &_file, &_func,
                      &_info.line, &_discriminator) != 0);
        if(_found)
        {
            if(_file) _info.file = _file;
            if(_func) _info.function = _func;
            if(!_file || (_file && strnlen(_file, 1) == 0))
                _info.file = bfd_get_filename(_inp);
            _info.file = filepath::realpath(_info.file, nullptr, false);
            _info_vec->data.emplace_back(_info);

            while(_found)
            {
                _info          = bfd_line_info{};
                _info.address  = _pc;
                _info.priority = _prio++;
                _found = (bfd_find_inliner_info(_inp, &_file, &_func, &_info.line) != 0);

                if(_found)
                {
                    if(_file) _info.file = _file;
                    if(_func) _info.function = _func;
                    if(!_file || (_file && strnlen(_file, 1) == 0))
                        _info.file = bfd_get_filename(_inp);
                    _info.file = filepath::realpath(_info.file, nullptr, false);
                    _info_vec->data.emplace_back(_info);
                }
                else
                    break;
            }
        }
    }
}

auto
process_dwarf_cu_die(Dwarf_Die* _die)
{
    auto _line_info = dwarf_line_info_vec_t{};

    if(dwarf_tag(_die) != DW_TAG_compile_unit) return _line_info;
    /*
    Dwarf_Addr _low_pc;
    Dwarf_Addr _high_pc;
    Dwarf_Addr _entry_pc;

    dwarf_lowpc(_die, &_low_pc);
    dwarf_highpc(_die, &_high_pc);
    dwarf_entrypc(_die, &_entry_pc);

    Dwarf_Addr _base_addr;
    ptrdiff_t  _offset = 0;
    auto       _ranges = std::vector<address_range_t>{};
    do
    {
        _ranges.emplace_back(address_range_t{ 0, 0 });
    } while((_offset = dwarf_ranges(_die, _offset, &_base_addr, &_ranges.back().first,
                                    &_ranges.back().second)) > 0);
    // will always have one extra
    _ranges.pop_back();
    */

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

dwarf_line_info_vec_t
iterate_dwarf_die(Dwarf_Die* _die, int _nsib = 0, int _nchild = 0)
{
    auto _line_info = process_dwarf_cu_die(_die);

    Dwarf_Die _next_child;
    if(dwarf_child(_die, &_next_child) == 0)
        combine(_line_info, iterate_dwarf_die(&_next_child, _nsib, _nchild + 1));

    Dwarf_Die _next_sibling;
    if(dwarf_siblingof(_die, &_next_sibling) == 0)
        combine(_line_info, iterate_dwarf_die(&_next_sibling, _nsib + 1, _nchild));

    if(!_line_info.empty()) filter_sort_unique(_line_info);

    return _line_info;
}

auto
process_dwarf(const std::string& _file)
{
    FILE* _fd        = filepath::fopen(_file, "r");
    auto* _dwarf_v   = dwarf_begin(fileno(_fd), DWARF_C_READ);
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
            combine(_line_info, iterate_dwarf_die(&cu_die));
    }

    dwarf_end(_dwarf_v);
    ::fclose(_fd);

    filter_sort_unique(_line_info);

    return _line_info;
}

auto
parse_line_info(const std::string& _fname)
{
    auto _bfd = std::make_shared<bfd_file>(_fname);

    TIMEMORY_PRINTF(stderr, "Reading line info for '%s'...\n", _fname.c_str());

    auto _info = ext_line_info{};
    _info.file = _bfd;
    if(_bfd && _bfd->is_good())
    {
        for(auto&& itr : _bfd->get_symbols())
            read_pc(&_info, static_cast<asection*>(itr.section), itr.address);
        for(auto&& itr : process_dwarf(_fname))
            _info.data.emplace_back(itr);
        _info.sort();
    }

    return _info;
}
}  // namespace

bool
bfd_line_info::is_inlined() const
{
    return priority > 0;
}

bool
bfd_line_info::is_valid() const
{
    return !file.empty() || !function.empty();
}

basic::line_info
bfd_line_info::get_basic() const
{
    return basic::line_info{ line, address, file, file };
}

bool
dwarf_line_info::is_valid() const
{
    return valid && !file.empty();
}

basic::line_info
dwarf_line_info::get_basic() const
{
    return basic::line_info{ line, address, file, std::string{} };
}

template <typename DataT>
void
line_info<DataT>::sort()
{
    filter_sort_unique(data);
}

std::map<procfs::maps, ext_line_info>
get_line_info(std::map<procfs::maps, ext_line_info>* _discarded)
{
    TIMEMORY_PRINTF(stderr, "Line info binary scope: %s\n",
                    config::get_causal_binary_scope().c_str());
    TIMEMORY_PRINTF(stderr, "Line info source scope: %s\n",
                    config::get_causal_source_scope().c_str());

    auto _filter = [](const procfs::maps& _v) {
        // exclude internal libraries used by omnitrace
        auto&& _ourlib_scope =
            std::regex{ "lib(omnitrace[-\\.]|gotcha\\.|unwind\\.so\\.99)" };
        auto&& _binary_scope = std::regex{ config::get_causal_binary_scope() };
        auto&& _path         = _v.pathname;
        return (!_path.empty() && filepath::exists(_path) &&
                std::regex_search(_path, _ourlib_scope) == false &&
                std::regex_search(_path, _binary_scope));
    };

    auto _maps  = procfs::get_contiguous_maps(process::get_id(), _filter, true);
    auto _files = std::set<std::string>{};
    for(auto itr : _maps)
        _files.emplace(itr.pathname);

    auto _fdata = std::map<std::string, ext_line_info>{};
    for(auto itr : _files)
        _fdata.emplace(itr, parse_line_info(itr));

    auto _source_scope = std::regex{ config::get_causal_source_scope() };
    auto _data         = std::map<procfs::maps, ext_line_info>{};
    for(const auto& mitr : _maps)
    {
        auto itr = _fdata.find(mitr.pathname);
        if(itr != _fdata.end())
        {
            _data.emplace(mitr, ext_line_info{ itr->second.file });
            if(_discarded) _discarded->emplace(mitr, ext_line_info{ itr->second.file });
            for(const auto& vitr : itr->second.data)
            {
                auto _file = std::visit([](auto&& _v) { return _v.file; }, vitr);
                if(!std::regex_search(_file, _source_scope))
                {
                    if(_discarded) _discarded->at(mitr).data.emplace_back(vitr);
                }
                else
                {
                    auto _addr = std::visit([](auto&& _v) { return _v.address; }, vitr);
                    if(_addr < mitr.length())
                        _data.at(mitr).data.emplace_back(vitr);
                    else if(_discarded)
                        _discarded->at(mitr).data.emplace_back(vitr);
                }
            }
            _data.at(mitr).sort();
            if(_discarded) _discarded->at(mitr).sort();
        }
    }

    return _data;
}

std::map<procfs::maps, basic_line_info>
get_basic_line_info(std::map<procfs::maps, basic_line_info>* _discarded)
{
    using ext_line_info_t   = std::map<procfs::maps, ext_line_info>;
    using basic_line_info_t = std::map<procfs::maps, basic_line_info>;
    auto _ext_discarded     = ext_line_info_t{};
    auto _ext_data          = get_line_info(&_ext_discarded);

    auto _data = basic_line_info_t{};
    for(auto& itr : _ext_data)
    {
        _data.emplace(itr.first, basic_line_info{ itr.second.file });
        auto& _data_v = _data.at(itr.first).data;
        for(const auto& ditr : itr.second.data)
            _data_v.emplace_back(
                std::visit([](const auto& _v) { return _v.get_basic(); }, ditr));
    }

    if(_discarded)
    {
        for(auto& itr : _ext_discarded)
        {
            _discarded->emplace(itr.first, basic_line_info{ itr.second.file });
            auto& _data_v = _discarded->at(itr.first).data;
            for(const auto& ditr : itr.second.data)
                _data_v.emplace_back(
                    std::visit([](const auto& _v) { return _v.get_basic(); }, ditr));
        }
    }

    return _data;
}
}  // namespace code_object
}  // namespace omnitrace
