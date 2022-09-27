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

#include "common/delimit.hpp"
#include "common/environment.hpp"
#include "common/join.hpp"
#include "common/setup.hpp"

#include <timemory/log/color.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/join.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace color = tim::log::color;
using tim::log::stream;
using namespace timemory::join;

namespace
{
int verbose = 0;
}

std::string
get_command(const char* _argv0)
{
    return omnitrace::path::find_path(_argv0, 0, omnitrace::common::get_env("PATH", ""));
}

void
print_command(const std::vector<char*>& _argv)
{
    if(verbose >= 1)
        stream(std::cout, color::info())
            << "Executing '" << join(array_config{ " " }, _argv) << "'...\n";
}

std::vector<char*>
get_environment()
{
    std::vector<char*> _environ;
    if(environ != nullptr)
    {
        int idx = 0;
        while(environ[idx] != nullptr)
            _environ.emplace_back(strdup(environ[idx++]));
    }

    auto _exe = std::string_view{ realpath("/proc/self/exe", nullptr) };
    auto _pos = _exe.find_last_of('/');
    auto _dir = std::string{ "./" };
    if(_pos != std::string_view::npos) _dir = _exe.substr(0, _pos);
    auto _lib = omnitrace::common::join("/", _dir, "..", "lib", "libomnitrace-dl.so");
    _environ.emplace_back(
        strdup(omnitrace::common::join("=", "LD_PRELOAD", realpath(_lib.c_str(), nullptr))
                   .c_str()));

    return _environ;
}

template <typename Tp>
void
update_env(std::vector<char*>& _environ, std::string_view _env_var, Tp&& _env_val)
{
    for(auto& itr : _environ)
    {
        if(!itr) continue;
        if(std::string_view{ itr }.find(_env_var) == 0)
        {
            free(itr);
            itr = strdup(omnitrace::common::join('=', _env_var, _env_val).c_str());
            return;
        }
    }
    _environ.emplace_back(
        strdup(omnitrace::common::join('=', _env_var, _env_val).c_str()));
}

std::vector<char*>
parse_args(int argc, char** argv, std::vector<char*>& _env)
{
    using parser_t     = tim::argparse::argument_parser;
    using parser_err_t = typename parser_t::result_type;

    auto help_check = [](parser_t& p, int _argc, char** _argv) {
        std::set<std::string> help_args = { "-h", "--help", "-?" };
        return (p.exists("help") || _argc == 1 ||
                (_argc > 1 && help_args.find(_argv[1]) != help_args.end()));
    };

    auto _pec        = EXIT_SUCCESS;
    auto help_action = [&_pec, argc, argv](parser_t& p) {
        if(_pec != EXIT_SUCCESS)
        {
            std::stringstream msg;
            msg << "Error in command:";
            for(int i = 0; i < argc; ++i)
                msg << " " << argv[i];
            msg << "\n\n";
            stream(std::cerr, color::fatal()) << msg.str();
            std::cerr << std::flush;
        }

        p.print_help();
        exit(_pec);
    };

    auto parser = parser_t(argv[0]);

    parser.enable_help();
    parser.on_error([=, &_pec](parser_t& p, const parser_err_t& _err) {
        stream(std::cerr, color::fatal()) << _err << "\n";
        _pec = EXIT_FAILURE;
        help_action(p);
    });

    parser.add_argument()
        .names({ "--debug" })
        .description("Debug output")
        .max_count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_DEBUG", p.get<bool>("debug"));
        });
    parser.add_argument()
        .names({ "-v", "--verbose" })
        .description("Verbose output")
        .count(1)
        .action([&](parser_t& p) {
            auto _v = p.get<int>("verbose");
            verbose = _v;
            update_env(_env, "OMNITRACE_VERBOSE", _v);
        });
    parser.add_argument({ "-N", "--no-color" }, "Disable colorized output")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            auto _colorized = !p.get<bool>("no-color");
            update_env(_env, "OMNITRACE_COLORIZED_LOG", (_colorized) ? "1" : "0");
            update_env(_env, "COLORIZED_LOG", (_colorized) ? "1" : "0");
        });
    parser.add_argument()
        .names({ "-c", "--config" })
        .description("Configuration file")
        .min_count(1)
        .action([&](parser_t& p) {
            update_env(
                _env, "OMNITRACE_CONFIG_FILE",
                join(array_config{ ":" }, p.get<std::vector<std::string>>("config")));
        });
    parser
        .add_argument({ "-d", "--delay" },
                      "Set the delay before the sampler starts (seconds)")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_SAMPLING_DELAY", p.get<double>("delay"));
        });
    parser
        .add_argument({ "-f", "--freq" }, "Set the frequency of the sampler "
                                          "(number of interrupts per second)")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_SAMPLING_FREQ", p.get<double>("freq"));
        });
    parser
        .add_argument({ "-D", "--duration" },
                      "Set the duration of the sampling (seconds)")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_SAMPLING_DURATION", p.get<double>("duration"));
        });
    parser
        .add_argument(
            { "-C", "--cpu-events" },
            "Set the hardware counter events to record (ref: `omnitrace-avail -H -c CPU)")
        .action([&](parser_t& p) {
            auto _events =
                join(array_config{ "," }, p.get<std::vector<std::string>>("cpu-events"));
            update_env(_env, "OMNITRACE_PAPI_EVENTS", _events);
        });
    parser
        .add_argument({ "-G", "--gpu-events" },
                      "Set the GPU hardware counter events to record (ref: "
                      "`omnitrace-avail -H -c GPU)")
        .action([&](parser_t& p) {
            auto _events =
                join(array_config{ "," }, p.get<std::vector<std::string>>("gpu-events"));
            update_env(_env, "OMNITRACE_ROCM_EVENTS", _events);
        });

    auto  _args = parser.parse_known_args(argc, argv);
    auto  _cmdc = std::get<1>(_args);
    auto* _cmdv = std::get<2>(_args);

    if(help_check(parser, _cmdc, _cmdv)) help_action(parser);

    std::vector<char*> _argv = {};
    _argv.reserve(_cmdc);
    for(int i = 1; i < _cmdc; ++i)
        _argv.emplace_back(_cmdv[i]);

    return _argv;
}

/*
void
update_env(char*** envp)
{
    if(!envp) return;

    static constexpr size_t N = 3;

    using pair_t                = std::pair<std::string_view, int64_t>;
    std::array<pair_t, N> _locs = { pair_t{ "HSA_TOOLS_LIB", -1 },
                                    pair_t{ "ROCP_TOOL_LIB", -1 },
                                    pair_t{ "HSA_TOOLS_REPORT_LOAD_FAILURE", -1 } };

    char**& _envp = *envp;
    size_t  nenv  = 0;
    int64_t nadd  = _locs.size();
    if(_envp)
    {
        size_t i = 0;
        while(_envp[(i = nenv)])
        {
            ++nenv;
            for(auto& itr : _locs)
            {
                if(itr.second < 0 && std::string_view{ _envp[i] }.find(itr.first) == 0)
                {
                    itr.second = i;
                    --nadd;
                    fprintf(stderr, "found %s at index %zi\n", itr.first.data(), i);
                }
            }
        }
    }

    size_t nsize = nenv + 1;
    if(nadd > 0)
    {
        nsize += nadd;

        size_t _off = 0;
        for(auto& itr : _locs)
            if(itr.second < 0) itr.second = nenv + _off++;

        char** _envp_new = new char*[nsize];
        memset(_envp_new, 0, nsize * sizeof(char*));
        for(size_t i = 0; i < nenv; ++i)
            _envp_new[i] = _envp[i];

        _envp = _envp_new;
    }

    fprintf(stderr, "nsize=%zu, nenv=%zu, nadd=%zu\n", nsize, nenv, nadd);

    using loc_pair_t                = std::pair<std::string_view, std::string>;
    std::array<loc_pair_t, N> _libs = {
        loc_pair_t{ "HSA_TOOLS_LIB", omnitrace::dl::get_indirect().get_dl_library() },
        loc_pair_t{ "ROCP_TOOL_LIB", omnitrace::dl::get_indirect().get_omni_library() },
        loc_pair_t{ "HSA_TOOLS_REPORT_LOAD_FAILURE", "1" }
    };

    for(auto itr : _locs)
    {
        fprintf(stderr, "%s is at index %zu\n", itr.first.data(), itr.second);
        for(const auto& litr : _libs)
        {
            if(itr.first == litr.first)
                _envp[itr.second] =
                    strdup(omnitrace::common::join("=", itr.first, litr.second).c_str());
        }
    }
}
*/
