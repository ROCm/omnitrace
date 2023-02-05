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

#include "internal_libs.hpp"
#include "binary/analysis.hpp"
#include "binary/binary_info.hpp"
#include "binary/link_map.hpp"
#include "binary/scope_filter.hpp"
#include "binary/symbol.hpp"
#include "core/utility.hpp"
#include "fwd.hpp"
#include "log.hpp"
#include "timemory/environment/types.hpp"

#include <dlfcn.h>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/filepath.hpp>

#include <algorithm>
#include <initializer_list>
#include <set>
#include <string>
#include <vector>

namespace
{
namespace binary   = ::omnitrace::binary;
namespace filepath = ::tim::filepath;
using ::tim::delimit;
using ::timemory::join::join;
using strview_init_t = std::initializer_list<std::string_view>;
using strview_set_t  = std::set<std::string_view>;

template <template <typename, typename...> class ContainerT, typename... TailT>
bool
check_regex_restrictions(const ContainerT<std::string, TailT...>& _names,
                         const regexvec_t&                        _regexes)
{
    for(const auto& nitr : _names)
        for(const auto& ritr : _regexes)
            if(std::regex_search(nitr, ritr)) return true;
    return false;
}

std::vector<std::string>
get_library_search_paths_impl()
{
    auto _paths = std::vector<std::string>{};

    auto _path_exists = [](const std::string& _filename) {
        struct stat dummy;
        return (_filename.empty()) ? false : (stat(_filename.c_str(), &dummy) == 0);
    };

    auto _emplace_if_exists = [&_paths, _path_exists](const std::string& _directory) {
        if(_path_exists(_directory)) _paths.emplace_back(_directory);
    };

    // search paths from environment variables
    for(const auto& itr :
        delimit(tim::get_env("LD_LIBRARY_PATH", std::string{}, false), ":"))
        _emplace_if_exists(itr);

    // search ld.so.cache
    // apparently ubuntu doesn't like pclosing NULL, so a shared pointer custom
    // destructor is out. Ugh.
    FILE* ldconfig = popen("/sbin/ldconfig -p", "r");
    if(ldconfig)
    {
        constexpr size_t buffer_size = 512;
        char             buffer[buffer_size];
        // ignore first line
        if(fgets(buffer, buffer_size, ldconfig))
        {
            // each line constaining relevant info should be in form:
            //      <LIBRARY_BASENAME> (...) => <RESOLVED_ABSOLUTE_PATH>
            // example:
            //      libz.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libz.so
            auto _get_entry = [](const std::string& _inp) {
                auto _paren_pos = _inp.find('(');
                auto _arrow_pos = _inp.find("=>", _paren_pos);
                if(_arrow_pos == std::string::npos || _paren_pos == std::string::npos)
                    return std::string{};
                if(_arrow_pos + 2 < _inp.length())
                {
                    auto _pos = _inp.find_first_not_of(" \t", _arrow_pos + 2);
                    if(_pos < _inp.length()) return _inp.substr(_pos);
                }
                return std::string{};
            };

            auto _data = std::stringstream{};
            while(fgets(buffer, buffer_size, ldconfig) != nullptr)
            {
                _data << buffer;
                auto _len = strnlen(buffer, buffer_size);
                if(_len > 0 && buffer[_len - 1] == '\n')
                {
                    auto _v = _data.str();
                    if(!_v.empty())
                    {
                        _v = _v.substr(_v.find_first_not_of(" \t"));
                        if(_v.length() > 1)
                        {
                            auto _entry = _get_entry(_v.substr(0, _v.length() - 1));
                            if(!_entry.empty()) _emplace_if_exists(_entry);
                        }
                    }
                    _data = std::stringstream{};
                }
            }
        }
        pclose(ldconfig);
    }

    // search hard-coded system paths
    for(const char* itr :
        { "/usr/local/lib", "/usr/share/lib", "/usr/lib", "/usr/lib64",
          "/usr/lib/x86_64-linux-gnu", "/lib", "/lib64", "/lib/x86_64-linux-gnu",
          "/usr/lib/i386-linux-gnu", "/usr/lib32" })
    {
        _emplace_if_exists(itr);
    }

    return _paths;
}

std::set<std::string>
get_internal_libs_impl()
{
    auto       _libs    = std::set<std::string>{};
    const auto _exclude = strview_set_t{ LIBM_SO, LIBMVEC_SO };

    // GNU libraries likely to be used by instrumentation
    const auto _gnu_libs = strview_init_t{ LD_LINUX_X86_64_SO, LD_SO,
                                           LIBANL_SO,          LIBBROKENLOCALE_SO,
                                           LIBCRYPT_SO,        LIBC_SO,
                                           LIBDL_SO,           LIBGCC_S_SO,
                                           LIBMVEC_SO,         LIBM_SO,
                                           LIBNSL_SO,          LIBNSS_COMPAT_SO,
                                           LIBNSS_DB_SO,       LIBNSS_DNS_SO,
                                           LIBNSS_FILES_SO,    LIBNSS_HESIOD_SO,
                                           LIBNSS_LDAP_SO,     LIBNSS_NISPLUS_SO,
                                           LIBNSS_NIS_SO,      LIBNSS_TEST1_SO,
                                           LIBNSS_TEST2_SO,    LIBPTHREAD_SO,
                                           LIBRESOLV_SO,       LIBRT_SO,
                                           LIBTHREAD_DB_SO,    LIBUTIL_SO };

    // shared libraries used by or provided by dyninst
    const auto _dyn_libs = strview_init_t{ "libdyninstAPI_RT.so",
                                           "libcommon.so",
                                           "libstackwalk.so",
                                           "libbfd.so",
                                           "libelf.so",
                                           "libdwarf.so",
                                           "libdw.so",
                                           "libtbb.so",
                                           "libtbbmalloc.so",
                                           "libtbbmalloc_proxy.so",
                                           "libz.so",
                                           "libzstd.so",
                                           "libbz2.so",
                                           "liblzma.so" };

    // shared libraries used by omnitrace
    const auto _omni_libs = strview_init_t{
        "libstdc++.so.6",       "libgotcha.so",        "libunwind-coredump.so",
        "libunwind-generic.so", "libunwind-ptrace.so", "libunwind-setjmp.so",
        "libunwind.so",         "libunwind-x86_64.so", "librocm_smi64.so",
        "libroctx64.so",        "librocmtools.so",     "libroctracer64.so",
        "librocprofiler64.so",  "libpapi.so",          "libpfm.so"
    };

    // shared libraries potentially used by timemory
    const auto _3rdparty_libs = strview_init_t{ "libcaliper.so",
                                                "liblikwid.so",
                                                "libprofiler.so",
                                                "libtcmalloc.so",
                                                "libtcmalloc_and_profiler.so",
                                                "libtcmalloc_debug.so",
                                                "libtcmalloc_minimal.so",
                                                "libtcmalloc_minimal_debug.so" };

    for(const auto& gitr : { _gnu_libs, _dyn_libs, _omni_libs, _3rdparty_libs })
    {
        for(auto itr : gitr)
        {
            if(!itr.empty() && _exclude.count(itr) == 0)
            {
                if(parse_all_modules)
                {
                    auto _lib_v = find_libraries(itr);
                    if(!_lib_v.empty())
                    {
                        for(const auto& litr : _lib_v)
                        {
                            verbprintf(1, "Library '%s' found: %s\n", itr.data(),
                                       litr.c_str());
                            _libs.emplace(litr);
                        }
                    }
                    else
                    {
                        verbprintf(2, "Library '%s' not found\n", itr.data());
                    }
                }
                else
                {
                    auto _lib_v = find_library(itr);
                    if(_lib_v)
                    {
                        verbprintf(1, "Library '%s' found: %s\n", itr.data(),
                                   _lib_v->c_str());
                        _libs.emplace(*_lib_v);
                    }
                    else
                    {
                        verbprintf(2, "Library '%s' not found\n", itr.data());
                    }
                }
            }
        }
    }

    // auto _link_map = binary::get_link_map(nullptr, "", "", { (RTLD_LAZY | RTLD_NOLOAD)
    // }); for(const auto& itr : _link_map)
    //    _libs.emplace(itr.real());

    return _libs;
}

library_module_map_t
get_internal_libs_data_impl()
{
    auto _libs_v = get_internal_libs();
    auto _libs   = std::vector<std::string>{};
    _libs.assign(_libs_v.begin(), _libs_v.end());

    auto _omnitrace_base_path = filepath::dirname(
        filepath::dirname(filepath::realpath("/proc/self/exe", nullptr, false)));
    auto _omnitrace_lib_path = std::string{};

    for(const auto* itr : { "lib", "lib64" })
    {
        for(const auto* litr :
            { "libomnitrace-dl.so", "libomnitrace-user.so", "libomnitrace-rt.so" })
        {
            auto _libpath = join('/', _omnitrace_base_path, itr, litr);
            if(filepath::exists(_libpath))
            {
                _libs.emplace_back(filepath::realpath(_libpath, nullptr, false));
            }
        }
    }

    omnitrace::utility::filter_sort_unique(
        _libs, [](const auto& itr) { return itr.empty() || !filepath::exists(itr); });

    for(const auto& itr : _libs)
    {
        verbprintf(2, "Analyzing %s...\n", itr.c_str());
        OMNITRACE_ADD_LOG_ENTRY("Found system library:", itr);
    }

    // get the symbols in the libraries but avoid processing any DWARF info
    const auto _filters       = std::vector<binary::scope_filter>{};
    const auto _process_dwarf = false;  // read dwarf info
    const auto _process_lines = true;   // read BFD line info
    const auto _include_all   = false;  // include undefined
    auto _info = binary::get_binary_info(_libs, _filters, _process_dwarf, _process_lines,
                                         _include_all);

    auto _get_files = [](const std::string& _file_v) {
        if(_file_v.empty()) return std::vector<std::string>{};
        std::string _base = omnitrace::filepath::basename(_file_v);
        if(_file_v != _base) return std::vector<std::string>{ _file_v, _base };
        return std::vector<std::string>{ _file_v };
    };

    auto _get_funcs = [](std::string _func_v) {
        auto _v = func_set_t{};
        if(_func_v.empty()) return _v;

        _v.emplace(_func_v);
        auto _func_v_d = tim::demangle(_func_v);
        if(_func_v.find("_Z") != 0 || _func_v == _func_v_d)
        {
            for(std::string_view itr : { "__old", "__new", "__IO", "__GI", "_IO", "_GI" })
            {
                auto _pos = _func_v.find(itr);
                if(_pos == 0)
                {
                    _func_v = _func_v.substr(itr.length());
                    _v.emplace(_func_v);
                }
            }

            auto _pos = std::string::npos;
            while((_pos = _func_v.find("__")) == 0)
            {
                _func_v = _func_v.substr(2);
                _v.emplace(_func_v);
            }

            while((_pos = _func_v.find('_')) == 0)
            {
                _func_v = _func_v.substr(1);
                _v.emplace(_func_v);
            }
        }
        else
        {
            _v.emplace(_func_v_d);
        }

        return _v;
    };

    auto _data = library_module_map_t{};
    for(const auto& itr : _info)
    {
        for(const auto& iitr : itr.symbols)
        {
            auto _files = _get_files(iitr.file);
            if(check_regex_restrictions(_files, file_internal_include)) continue;

            auto _funcs = _get_funcs(iitr.func);
            if(check_regex_restrictions(_funcs, func_internal_include)) continue;

            if(_files.empty())
            {
                for(const auto& fitr : _funcs)
                    _data[itr.filename()][itr.filename()].emplace(fitr);
            }
            else if(_funcs.empty())
            {
                for(const auto& fitr : _files)
                    _data[itr.filename()].emplace(fitr, func_set_t{});
            }
            else
            {
                for(const auto& fiitr : _files)
                {
                    for(const auto& fuitr : _funcs)
                    {
                        _data[itr.filename()][fiitr].emplace(fuitr);
                    }
                }
            }
        }
    }

    const auto _verbose_lvl = 3;
    for(const auto& ditr : _data)
    {
        verbprintf(_verbose_lvl, "%s\n", ditr.first.c_str());
        OMNITRACE_ADD_LOG_ENTRY(ditr.first);
        for(const auto& fitr : ditr.second)
        {
            verbprintf(_verbose_lvl, "  - %s\n", fitr.first.c_str());
            OMNITRACE_ADD_LOG_ENTRY("  -", fitr.first);
            for(const auto& itr : fitr.second)
            {
                verbprintf(_verbose_lvl, "    + %s\n", itr.c_str());
                OMNITRACE_ADD_LOG_ENTRY("    +", itr);
            }
        }
    }

    return _data;
}
}  // namespace

std::optional<std::string>
find_library(std::string_view _lib_v)
{
    auto _lib = binary::get_linked_path(_lib_v.data(), { (RTLD_LAZY | RTLD_NOLOAD) });
    if(_lib) return _lib;

    for(const auto& itr : get_library_search_paths())
    {
        auto _path = join('/', itr, _lib_v);
        if(filepath::exists(_path)) return _path;
    }

    return _lib;
}

std::vector<std::string>
find_libraries(std::string_view _lib_v)
{
    auto _libs = std::vector<std::string>{};
    auto _lib  = binary::get_linked_path(_lib_v.data(), { (RTLD_LAZY | RTLD_NOLOAD) });
    if(_lib) _libs.emplace_back(*_lib);

    for(const auto& itr : get_library_search_paths())
    {
        auto _path = join('/', itr, _lib_v);
        if(filepath::exists(_path)) _libs.emplace_back(_path);
    }
    return _libs;
}

const std::vector<std::string>&
get_library_search_paths()
{
    static auto _v = get_library_search_paths_impl();
    return _v;
}

std::set<std::string>&
get_internal_libs(bool _all_modules)
{
    static auto _parse_v = false;
    static auto _v       = get_internal_libs_impl();

    // make sure up-to-date if parse_all_modules changed
    if(_parse_v != parse_all_modules)
    {
        _v       = get_internal_libs_impl();
        _parse_v = parse_all_modules;
    }

    // if all modules are requested
    if(_all_modules && !_parse_v)
    {
        std::swap(_all_modules, parse_all_modules);
        _v = get_internal_libs_impl();
        std::swap(_all_modules, parse_all_modules);
    }

    return _v;
}

const library_module_map_t&
get_internal_libs_data()
{
    static auto _v = get_internal_libs_data_impl();
    return _v;
}
