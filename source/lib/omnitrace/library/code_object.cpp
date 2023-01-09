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
#include "library/debug.hpp"

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

#define PAGE_SIZE ::tim::units::get_page_size()
#define PAGE_MASK (PAGE_SIZE - 1)
/* Convenience macros to make page address/offset computations more explicit */
/* Returns the address of the page starting at address 'x' */
#define PAGE_START(x) ((x) & ~PAGE_MASK)
/* Returns the offset of address 'x' in its memory page, i.e. this is the
 * same than 'x' - PAGE_START(x) */
#define PAGE_OFFSET(x) ((x) &PAGE_MASK)
/* Returns the address of the next page after address 'x', unless 'x' is
 * itself at the start of a page. Equivalent to:
 *
 *  (x == PAGE_START(x)) ? x : PAGE_START(x)+PAGE_SIZE
 */
#define PAGE_END(x) PAGE_START((x) + (PAGE_SIZE - 1))

namespace omnitrace
{
namespace causal
{
template <typename Tp>
std::string
as_hex(Tp _v, size_t _width = 16);
}

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
read_pcs(bfd_file& _bfd, asection* _section, const std::vector<bfd_vma>& _pcs)
{
    auto          _data = std::vector<bfd_line_info>{};
    bfd_vma       _vma  = bfd_section_vma(_section);
    bfd_size_type _size = bfd_section_size(_section);

    auto* _inp  = static_cast<bfd*>(_bfd.data);
    auto* _syms = reinterpret_cast<asymbol**>(_bfd.syms);

    auto _read_pc = [&](bfd_vma _pc) {
        auto         _info  = bfd_line_info{};
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
                if(_func) _info.func = _func;
                if(!_file || (_file && strnlen(_file, 1) == 0))
                    _info.file = bfd_get_filename(_inp);
                _info.file = filepath::realpath(_info.file, nullptr, false);
                _data.emplace_back(_info);

                while(_found)
                {
                    _info          = bfd_line_info{};
                    _info.address  = _pc;
                    _info.priority = _prio++;
                    _found =
                        (bfd_find_inliner_info(_inp, &_file, &_func, &_info.line) != 0);

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
        }
    };

    for(auto itr : _pcs)
    {
        if(itr >= _vma && itr <= _vma + _size) _read_pc(itr);
    }

    return _data;
}

auto
read_pc(bfd_file& _bfd, asection* _section, bfd_vma _pc)
{
    auto          _data = std::vector<bfd_line_info>{};
    bfd_vma       _vma  = bfd_section_vma(_section);
    bfd_size_type _size = bfd_section_size(_section);

    if(_pc < _vma || _pc >= _vma + _size) return _data;

    auto* _inp  = static_cast<bfd*>(_bfd.data);
    auto* _syms = reinterpret_cast<asymbol**>(_bfd.syms);
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
            if(_func) _info.func = _func;
            if(!_file || (_file && strnlen(_file, 1) == 0))
                _info.file = bfd_get_filename(_inp);
            _info.file = filepath::realpath(_info.file, nullptr, false);
            _data.emplace_back(_info);

            while(_found)
            {
                _info          = bfd_line_info{};
                _info.address  = _pc;
                _info.priority = _prio++;
                _found = (bfd_find_inliner_info(_inp, &_file, &_func, &_info.line) != 0);

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
    }

    return _data;
}

auto
process_dwarf_cu_die(Dwarf_Die* _die)
{
    auto _line_info = dwarf_line_info_vec_t{};

    auto _tag = dwarf_tag(_die);
    if(_tag != DW_TAG_compile_unit) return _line_info;
    /*
    Dwarf_Addr _low_pc;
    Dwarf_Addr _high_pc;
    Dwarf_Addr _entry_pc;

    dwarf_lowpc(_die, &_low_pc);
    dwarf_highpc(_die, &_high_pc);
    dwarf_entrypc(_die, &_entry_pc);
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

    Dwarf_Addr _base_addr;
    ptrdiff_t  _offset = 0;
    auto       _ranges = std::vector<address_range_t>{};
    do
    {
        _ranges.emplace_back(address_range_t{ 0, 0 });
    } while((_offset = dwarf_ranges(_die, _offset, &_base_addr, &_ranges.back().first,
                                    &_ranges.back().second)) > 0);
    // will always have one extra
    //_ranges.pop_back();

    for(auto ritr : _ranges)
    {
        for(uintptr_t addr = ritr.first; addr <= ritr.second; ++addr)
        {
            auto* _line = dwarf_getsrc_die(_die, addr);
            if(_line)
            {
                _line_info.emplace_back();
                auto& itr     = _line_info.back();
                int   _lineno = 0;
                itr.valid     = true;
                dwarf_lineno(_line, &_lineno);
                dwarf_linecol(_line, &itr.col);
                dwarf_linebeginstatement(_line, &itr.begin_statement);
                dwarf_lineendsequence(_line, &itr.end_sequence);
                dwarf_lineblock(_line, &itr.line_block);
                dwarf_lineepiloguebegin(_line, &itr.epilogue_begin);
                dwarf_lineprologueend(_line, &itr.prologue_end);
                dwarf_lineisa(_line, &itr.isa);
                dwarf_linediscriminator(_line, &itr.discriminator);
                // dwarf_lineaddr(_line, &itr.address);
                itr.address = addr;
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

    /*
    Dwarf_Die _next_child;
    if(dwarf_child(_die, &_next_child) == 0)
        combine(_line_info, iterate_dwarf_die(&_next_child, _nsib, _nchild + 1));

    Dwarf_Die _next_sibling;
    if(dwarf_siblingof(_die, &_next_sibling) == 0)
        combine(_line_info, iterate_dwarf_die(&_next_sibling, _nsib + 1, _nchild));
    */

    (void) _nsib;
    (void) _nchild;

    if(!_line_info.empty()) filter_sort_unique(_line_info);

    return _line_info;
}

auto
process_dwarf(int _fd)
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
            combine(_line_info, iterate_dwarf_die(&cu_die));
    }

    dwarf_end(_dwarf_v);
    filter_sort_unique(_line_info);

    return _line_info;
}

auto
process_elf(ext_line_info& _info)
{
    auto _to_string = [](auto _v) {
        switch(_v)
        {
            case ELF_T_BYTE: return "BYTE";
            case ELF_T_ADDR: return "ADDR";
            case ELF_T_DYN: return "DYN";
            case ELF_T_EHDR: return "EHDR";
            case ELF_T_HALF: return "HALF";
            case ELF_T_OFF: return "OFF";
            case ELF_T_PHDR: return "PHDR";
            case ELF_T_RELA: return "RELA";
            case ELF_T_REL: return "REL";
            case ELF_T_SHDR: return "SHDR";
            case ELF_T_SWORD: return "SWORD";
            case ELF_T_SYM: return "SYM";
            case ELF_T_WORD: return "WORD";
            case ELF_T_XWORD: return "XWORD";
            case ELF_T_SXWORD: return "SXWORD";
            case ELF_T_VDEF: return "VDEF";
            case ELF_T_VDAUX: return "VDAUX";
            case ELF_T_VNEED: return "VNEED";
            case ELF_T_VNAUX: return "VNAUX";
            case ELF_T_NHDR: return "NHDR";
            case ELF_T_SYMINFO: return "SYMINFO";
            case ELF_T_MOVE: return "MOVE";
            case ELF_T_LIB: return "LIB";
            case ELF_T_GNUHASH: return "GNUHASH";
            case ELF_T_AUXV: return "AUXV";
            case ELF_T_CHDR: return "CHDR";
            case ELF_T_NHDR8: return "NHDR8";
            case ELF_T_NUM: return "NUM";
        }
        return "UNK";
    };

    TIMEMORY_PRINTF(stderr, "Opening elf file: %s...\n", _info.file->name);
    fflush(stderr);
    auto* _elf_v = elf_begin(_info.file->fd, ELF_C_READ, nullptr);
    if(_elf_v)
    {
        size_t _num_sections = 0;
        elf_getshdrnum(_elf_v, &_num_sections);
        TIMEMORY_PRINTF(stderr, "sections :: %zu \n", _num_sections);
        fflush(stderr);
        for(size_t i = 0; i < _num_sections; ++i)
        {
            auto*     _section = elf_getscn(_elf_v, i);
            Elf_Data* _tmp     = elf_getdata(_section, nullptr);
            TIMEMORY_PRINTF(stderr, "section %zu type :: %s\n", i,
                            _to_string((_tmp) ? _tmp->d_type : ELF_T_NUM));
            Elf_Data _data = {};
            while(elf_getdata(_section, &_data) != nullptr)
            {
                TIMEMORY_PRINTF(stderr, "section %zu type :: %i\n", i,
                                _to_string(_data.d_type));
            }
        }
        (void) _info;
    }
    elf_end(_elf_v);
}

auto
parse_line_info(const std::string& _fname)
{
    auto _bfd = std::make_shared<bfd_file>(_fname);

    OMNITRACE_VERBOSE(1, "Reading line info for '%s'...\n", _fname.c_str());

    auto _info = ext_line_info{};
    _info.file = _bfd;
    if(_bfd && _bfd->is_good())
    {
        auto _section_map = std::map<uintptr_t, std::set<asection*>>{};
        auto _processed   = std::set<uintptr_t>{};
        for(auto&& itr : _bfd->get_symbols())
        {
            auto* _section = static_cast<asection*>(itr.section);
            _section_map[itr.address].emplace(_section);
            _processed.emplace(itr.address);
            for(auto&& ditr : read_pc(*_bfd, _section, itr.address))
                _info.data.emplace_back(ditr);
        }
        _info.sort();
        // process_elf(_info);
        /*
        elf::process_elf(_info.file->name);
        for(auto& itr : _info.data)
        {
            auto& _v    = std::get<bfd_line_info>(itr);
            auto  _addr = _v.address;
            auto  _sz   = elf::get_size(_info.file->name, _addr);
            _v.range    = { _addr, _addr + _sz };
            std::cout << "    [" << causal::as_hex(_v.range.init) << "-"
                      << causal::as_hex(_v.range.last) << "][" << std::setw(8)
                      << (_v.range.last - _v.range.init) << "] " << _v.func << "\n";
        }
        */
        for(auto&& itr : process_dwarf(_bfd->fd))
        {
            if(_processed.count(itr.address) > 0) continue;
            for(size_t i = 0; i < _section_map.size() - 1; ++i)
            {
                auto bitr = _section_map.begin();
                std::advance(bitr, i);
                if(itr.address < bitr->first) continue;
                auto eitr = bitr;
                std::advance(eitr, 1);
                if(itr.address >= eitr->first) continue;
                for(auto&& _section : bitr->second)
                    for(auto&& ditr : read_pc(*_bfd, _section, itr.address))
                        _info.data.emplace_back(ditr);
            }
            _processed.emplace(itr.address);
        }
        _info.sort();
    }

    return _info;
}

struct program_segment_header
{
    uint64_t  offset        = 0x0;
    uint64_t  file_size     = 0;
    uint64_t  memory_size   = 0;
    uint64_t  alignment     = 0;
    uintptr_t virtual_addr  = 0x0;
    uintptr_t physical_addr = 0x0;

    bool operator==(program_segment_header _v) const
    {
        return std::tie(offset, file_size, memory_size, alignment, virtual_addr,
                        physical_addr) == std::tie(_v.offset, _v.file_size,
                                                   _v.memory_size, _v.alignment,
                                                   _v.virtual_addr, _v.physical_addr);
    }

    bool operator!=(program_segment_header _v) const { return !(*this == _v); }
    bool operator<(program_segment_header _v) const
    {
        return std::tie(virtual_addr, memory_size, offset) <
               std::tie(_v.virtual_addr, _v.memory_size, _v.offset);
    }
};

struct program_segment_info
{
    uintptr_t                           address = 0x0;
    std::string                         name    = {};
    std::vector<program_segment_header> headers = {};

    auto get_load_bias() const
    {
        uintptr_t _p_vaddr0 = PAGE_SIZE;
        for(auto itr : headers)
            _p_vaddr0 = std::min<uintptr_t>(_p_vaddr0, itr.virtual_addr);
        return (address - PAGE_START(_p_vaddr0));
    }

    void validate() const
    {
        TIMEMORY_REQUIRE(address == PAGE_START(address))
            << "program segment for " << name << " is not aligned to page";

        for(auto&& itr : headers)
        {
            TIMEMORY_REQUIRE(PAGE_OFFSET(itr.virtual_addr) == PAGE_OFFSET(itr.offset))
                << "program segment header for " << name
                << " has a virtual address that is not offset from load address "
                   "correctly.";
        }
    }
};

int
dl_iterate_callback(struct dl_phdr_info* _info, size_t, void* _pdata)
{
    auto* _data = static_cast<std::vector<program_segment_info>*>(_pdata);

    static auto _exe_name = []() {
        auto _cmd     = tim::read_command_line(process::get_id());
        auto _cmd_env = tim::get_env<std::string>("OMNITRACE_COMMAND_LINE", "");
        if(!_cmd_env.empty()) _cmd = tim::delimit(_cmd_env, " ");
        return (_cmd.empty()) ? config::get_exe_name() : _cmd.front();
    }();

    auto _name = std::string{ (_info->dlpi_name) ? _info->dlpi_name : "" };

    auto _v = program_segment_info{};
    _v.name = (_name.empty()) ? _exe_name : _name;

    if(!filepath::exists(_v.name)) return 0;
    _v.name    = filepath::realpath(_v.name);
    _v.address = _info->dlpi_addr;
    _v.headers.reserve(_info->dlpi_phnum);

    for(int j = 0; j < _info->dlpi_phnum; j++)
    {
        auto _hdr = _info->dlpi_phdr[j];
        if(_hdr.p_type == PT_LOAD)
        {
            _v.headers.emplace_back(program_segment_header{
                _hdr.p_offset, _hdr.p_filesz, _hdr.p_memsz, _hdr.p_align,
                _v.address + _hdr.p_vaddr, _hdr.p_paddr });
        }
    }

    std::sort(_v.headers.begin(), _v.headers.end());
    _v.headers.erase(std::unique(_v.headers.begin(), _v.headers.end()), _v.headers.end());

    _v.validate();
    _data->emplace_back(std::move(_v));

    /*
    constexpr size_t MAX_DEPTH   = 32;
    const size_t     num_discard = 0;
    // retrieve call-stack
    void* trace[MAX_DEPTH];
    int   stack_depth = backtrace(trace, MAX_DEPTH);

    for(int i = num_discard + 1; i < stack_depth; i++)
    {
        Dl_info _dl_info;
        ElfW(Sym)* _elf_info = nullptr;
        if(trace[i] == nullptr) break;
        if(dladdr1(trace[i], &_dl_info, reinterpret_cast<void**>(&_elf_info),
                   RTLD_DL_SYMENT) == 0)
            continue;
        if(_elf_info == nullptr) continue;
        if(_dl_info.dli_saddr == nullptr) continue;

        const char* symname = _dl_info.dli_sname;
        // printf("entry: %s, %s\n", _dl_info.dli_fname, symname);

        // store entry to stack
        if(_dl_info.dli_fname && symname)
        {
            printf("  [%s][%u][%i][%i][%s][%zu][%32s][%32s]\n",
                   causal::as_hex((uintptr_t) trace[i]).c_str(), _elf_info->st_name,
                   _elf_info->st_info, _elf_info->st_other,
                   causal::as_hex(_elf_info->st_value).c_str(), _elf_info->st_size,
                   _dl_info.dli_fname, symname);
        }
    }*/

    return 0;
}

auto
iterate_program_headers()
{
    auto _data = std::vector<program_segment_info>{};
    dl_iterate_phdr(dl_iterate_callback, static_cast<void*>(&_data));
    return _data;
}
}  // namespace

address_range::address_range(uintptr_t _v)
: low{ _v }
, high{ _v }
{}

address_range::address_range(uintptr_t _low, uintptr_t _high)
: low{ std::min(_low, _high) }
, high{ std::max(_low, _high) }
{}

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

bool
address_range::contains(uintptr_t _v) const
{
    return (_v >= low || _v < high);
}

bool
address_range::contains(address_range _v) const
{
    return contains(_v.low) && contains(_v.high);
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
    if(_v.is_range() && !is_range()) return (_v == *this);
    // check if arg is in range
    if(is_range() && !_v.is_range()) return (_v.low >= low && _v.low < high);
    // both are ranges or both are just address
    return std::tie(low, high) == std::tie(_v.low, _v.high);
}

bool
address_range::operator<(address_range _v) const
{
    if(_v.low == _v.high && _v.low >= low && _v.low < high) return false;
    return (low == _v.low) ? (high > _v.high) : (low < _v.low);
}

bool
bfd_line_info::is_inlined() const
{
    return priority > 0;
}

bool
bfd_line_info::is_valid() const
{
    return !file.empty() || !func.empty();
}

basic::line_info
bfd_line_info::get_basic() const
{
    return basic::line_info{ is_inlined(), line, address, file, func };
}

bool
dwarf_line_info::is_valid() const
{
    return valid && !file.empty();
}

basic::line_info
dwarf_line_info::get_basic() const
{
    return basic::line_info{ false, line, address, file, std::string{} };
}

template <typename DataT>
void
line_info<DataT>::sort()
{
    filter_sort_unique(data);
}

std::map<procfs::maps, ext_line_info>
get_ext_line_info(std::map<procfs::maps, ext_line_info>* _discarded)
{
    OMNITRACE_VERBOSE(1, "Line info binary scope: %s\n",
                      config::get_causal_binary_scope().c_str());
    OMNITRACE_VERBOSE(1, "Line info source scope: %s\n",
                      config::get_causal_source_scope().c_str());

    // iterate program headers and add them to the procfs maps
    {
        auto _pheaders = iterate_program_headers();
        auto _pmaps    = std::vector<procfs::maps>{};
        for(const auto& itr : _pheaders)
        {
            auto _local_pmaps = std::vector<procfs::maps>{};
            for(const auto& pitr : itr.headers)
            {
                if(pitr.memory_size == 0) continue;
                auto _v         = procfs::maps{};
                _v.pathname     = itr.name;
                _v.load_address = pitr.virtual_addr;
                _v.last_address =
                    pitr.virtual_addr + std::max(pitr.memory_size, pitr.file_size);
                _v.offset      = pitr.offset;
                bool _overlaps = false;
                for(auto& litr : _local_pmaps)
                {
                    if(litr.is_contiguous_with(_v))
                    {
                        litr += _v;
                        _overlaps = true;
                        break;
                    }
                }
                if(!_overlaps) _local_pmaps.emplace_back(_v);
            }

            for(auto& litr : _local_pmaps)
                _pmaps.emplace_back(std::move(litr));
        }
        std::sort(_pmaps.begin(), _pmaps.end());
        auto& _maps = procfs::get_maps(process::get_id(), true);
        for(auto&& itr : _pmaps)
            _maps.emplace_back(std::move(itr));
        std::sort(_maps.begin(), _maps.end());
    }

    // filter function used by procfs::get_contiguous_maps
    // ensures that we do not process omnitrace/gotcha/libunwind libraries
    // and do not process the libraries outside of the binary scope
    auto _filter = [](const procfs::maps& _v) {
        // exclude internal libraries used by omnitrace
        auto&& _ourlib_scope = std::regex{
            "lib(omnitrace[-\\.]|dyninst|tbbmalloc|gotcha\\.|unwind\\.so\\.99)"
        };
        auto&& _binary_scope = std::regex{ config::get_causal_binary_scope() };
        auto&& _path         = _v.pathname;
        return (!_path.empty() && filepath::exists(_path) &&
                std::regex_search(_path, _ourlib_scope) == false &&
                std::regex_search(_path, _binary_scope));
    };

    // get the memory maps and create a unique set of filenames
    auto _maps  = procfs::get_contiguous_maps(process::get_id(), _filter, true);
    auto _files = std::set<std::string>{};
    for(auto itr : _maps)
        _files.emplace(itr.pathname);

    // get the line info for the set of files
    auto _fdata = std::map<std::string, ext_line_info>{};
    for(auto itr : _files)
        _fdata.emplace(itr, parse_line_info(itr));

    // sort the line info into one map which contains only the line info
    // in the source scope and another map which contains the line info
    // outside the source scope
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
    // convert the extended line info format into the basic line info format
    using ext_line_info_t   = std::map<procfs::maps, ext_line_info>;
    using basic_line_info_t = std::map<procfs::maps, basic_line_info>;
    auto _ext_discarded     = ext_line_info_t{};
    auto _ext_data          = get_ext_line_info(&_ext_discarded);

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
