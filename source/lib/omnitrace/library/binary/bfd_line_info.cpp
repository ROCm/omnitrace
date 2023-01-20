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

#define PACKAGE "omnitrace-binary-bfd-line-info"
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

#include "library/binary/basic_line_info.hpp"
#include "library/binary/bfd_line_info.hpp"
#include "library/utility.hpp"

#include <deque>
#include <memory>
#include <vector>

namespace omnitrace
{
namespace binary
{
namespace
{
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
        _info.address  = address_range{ _pc };  // inliner info is not over range
        _info.priority = _prio++;
        _found         = (bfd_find_inliner_info(_inp, &_file, &_func, &_info.line) != 0);

        if(_found)
        {
            if(_file) _info.file = _file;
            if(_func) _info.func = _func;
            if(!_file || strnlen(_file, 1) == 0) _info.file = bfd_get_filename(_inp);
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

    // for(bfd_vma i = _pc; i < _pc + _pc_len + 1; ++i)
    {
        auto         _info          = bfd_line_info{};
        unsigned int _prio          = 0;
        _info.address               = address_range{ _pc, _pc + _pc_len + 1 };
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
            if(!_file || strnlen(_file, 1) == 0) _info.file = bfd_get_filename(_inp);
            _info.file = filepath::realpath(_info.file, nullptr, false);
            if(_info)
            {
                _data.emplace_back(_info);
                // if(config::get_sampling_include_inlines())
                read_inliner_info(_inp, _data, _pc, _prio);
            }
        }
    }

    // for(bfd_vma i = _pc; i < _pc + _pc_len + 1; ++i)
    {
        auto         _info = bfd_line_info{};
        unsigned int _prio = 0;
        _info.address      = address_range{ _pc, _pc + _pc_len + 1 };
        _info.priority     = _prio++;
        const char* _file  = nullptr;
        const char* _func  = nullptr;

        if(bfd_find_nearest_line(_inp, _section, _syms, _info.low() - _vma, &_file,
                                 &_func, &_info.line) != 0)
        {
            if(_file) _info.file = _file;
            if(_func) _info.func = _func;
            if(!_file || strnlen(_file, 1) == 0) _info.file = bfd_get_filename(_inp);
            _info.file = filepath::realpath(_info.file, nullptr, false);
            if(_info)
            {
                _data.emplace_back(_info);
                // if(config::get_sampling_include_inlines())
                read_inliner_info(_inp, _data, _pc, _prio);
            }
        }
    }

    utility::filter_sort_unique(_data);

    return _data;
}
}  // namespace

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

basic_line_info
bfd_line_info::get_basic() const
{
    return basic_line_info{ is_weak(), is_inlined(), line, load_address,
                            address,   file,         func };
}

std::vector<bfd_line_info>
bfd_line_info::process_bfd(bfd_file& _bfd, void* _section, uintptr_t _pc,
                           uintptr_t _pc_len)
{
    return read_pc(_bfd, static_cast<asection*>(_section), static_cast<bfd_vma>(_pc),
                   static_cast<bfd_vma>(_pc_len));
}
}  // namespace binary
}  // namespace omnitrace
