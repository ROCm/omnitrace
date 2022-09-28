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

    update_env(_env, "OMNITRACE_USE_SAMPLING", true);
    update_env(_env, "OMNITRACE_CRITICAL_TRACE", false);

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

    const auto* _cputime_desc =
        R"(Sample based on a CPU-clock timer. Accepts up to 2 arguments:
    %{INDENT}%1. Interrupts per second. E.g., 100 == sample every 10 milliseconds of CPU-time.
    %{INDENT}%2. Delay (in seconds of CPU-clock time). I.e., how long each thread should wait before taking first sample.)";

    const auto* _realtime_desc =
        R"(Sample based on a real-clock timer. Accepts up to 2 arguments:
    %{INDENT}%1. Interrupts per second. E.g., 100 == sample every 10 milliseconds of realtime.
    %{INDENT}%2. Delay (in seconds of real-clock time). I.e., how long each thread should wait before taking first sample.)";

    const auto* _trace_policy_desc =
        R"(Policy for new data when the buffer size limit is reached:
    %{INDENT}%- discard     : new data is ignored
    %{INDENT}%- ring_buffer : new data overwrites oldest data)";

    parser.add_argument({ "" }, "");
    parser.add_argument({ "--monochrome" }, "Disable colorized output")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            auto _colorized = !p.get<bool>("monochrome");
            update_env(_env, "OMNITRACE_COLORIZED_LOG", (_colorized) ? "1" : "0");
            update_env(_env, "COLORIZED_LOG", (_colorized) ? "1" : "0");
        });
    parser.add_argument({ "--debug" }, "Debug output")
        .max_count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_DEBUG", p.get<bool>("debug"));
        });
    parser.add_argument({ "-v", "--verbose" }, "Verbose output")
        .count(1)
        .action([&](parser_t& p) {
            auto _v = p.get<int>("verbose");
            verbose = _v;
            update_env(_env, "OMNITRACE_VERBOSE", _v);
        });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "-c", "--config" }, "Configuration file")
        .min_count(1)
        .action([&](parser_t& p) {
            update_env(
                _env, "OMNITRACE_CONFIG_FILE",
                join(array_config{ ":" }, p.get<std::vector<std::string>>("config")));
        });
    parser.add_argument({ "-o", "--output" }, "Output path")
        .min_count(1)
        .max_count(2)
        .action([&](parser_t& p) {
            auto _v = p.get<std::vector<std::string>>("output");
            update_env(_env, "OMNITRACE_OUTPUT_PATH", _v.at(0));
            if(_v.size() > 1) update_env(_env, "OMNITRACE_OUTPUT_PREFIX", _v.at(1));
        });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "--trace" }, "Generate a detailed trace")
        .max_count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_USE_PERFETTO", p.get<bool>("trace"));
        });
    parser
        .add_argument({ "--trace-buffer-size" },
                      "Size limit for the trace output (in KB)")
        .count(1)
        .dtype("KB")
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_PERFETTO_BUFFER_SIZE_KB",
                       p.get<int64_t>("trace-buffer-size"));
        });
    parser.add_argument({ "--trace-fill-policy" }, _trace_policy_desc)
        .count(1)
        .choices({ "discard", "ring_buffer" })
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_PERFETTO_FILL_POLICY",
                       p.get<std::string>("trace-fill-policy"));
        });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "--profile" }, "Generate a call-stack-based profile")
        .min_count(0)
        .max_count(3)
        .choices({ "text", "json", "console" })
        .action([&](parser_t& p) {
            auto _v = p.get<std::set<std::string>>("profile");
            update_env(_env, "OMNITRACE_USE_TIMEMORY", true);
            if(!_v.empty())
            {
                update_env(_env, "OMNITRACE_TEXT_OUTPUT", _v.count("text") != 0);
                update_env(_env, "OMNITRACE_JSON_OUTPUT", _v.count("json") != 0);
                update_env(_env, "OMNITRACE_COUT_OUTPUT", _v.count("console") != 0);
            }
        });

    parser.add_argument({ "--flat-profile" }, "Generate a flat profile")
        .min_count(0)
        .max_count(3)
        .choices({ "text", "json", "console" })
        .action([&](parser_t& p) {
            auto _v = p.get<std::set<std::string>>("flat-profile");
            update_env(_env, "OMNITRACE_USE_TIMEMORY", true);
            update_env(_env, "OMNITRACE_FLAT_PROFILE", true);
            if(!_v.empty())
            {
                update_env(_env, "OMNITRACE_TEXT_OUTPUT", _v.count("text") != 0);
                update_env(_env, "OMNITRACE_JSON_OUTPUT", _v.count("json") != 0);
                update_env(_env, "OMNITRACE_COUT_OUTPUT", _v.count("console") != 0);
            }
        });
    parser
        .add_argument({ "--diff-profile" },
                      "Generate a profile diff from the specified input directory")
        .min_count(1)
        .max_count(2)
        .action([&](parser_t& p) {
            auto _v = p.get<std::vector<std::string>>("diff-profile");
            update_env(_env, "OMNITRACE_DIFF_OUTPUT", true);
            update_env(_env, "OMNITRACE_INPUT_PATH", _v.at(0));
            if(_v.size() > 1) update_env(_env, "OMNITRACE_INPUT_PREFIX", _v.at(1));
        });

    parser.add_argument({ "" }, "");
    parser
        .add_argument({ "-f", "--freq" }, "Set the default sampling frequency "
                                          "(number of interrupts per second)")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_SAMPLING_FREQ", p.get<double>("freq"));
        });
    parser
        .add_argument(
            { "-w", "--wait" },
            "Set the default wait time (i.e. delay) before taking first sample "
            "(in seconds). This delay time is based on the clock of the sampler, i.e., a "
            "delay of 1 second for CPU-clock sampler may not equal 1 second of realtime")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_SAMPLING_DELAY", p.get<double>("delay"));
        });
    parser
        .add_argument(
            { "-d", "--duration" },
            "Set the duration of the sampling (in seconds of realtime). I.e., it is "
            "possible (currently) to set a CPU-clock time delay that exceeds the "
            "real-time duration... resulting in zero samples being taken")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_SAMPLING_DURATION", p.get<double>("duration"));
        });
    parser
        .add_argument({ "-t", "--tids" },
                      "Specify the default thread IDs for sampling, where 0 (zero) is "
                      "the main thread and each thread created by the target application "
                      "is assigned an atomically incrementing value.")
        .min_count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_SAMPLING_TIDS",
                       join(array_config{ ", " }, p.get<std::vector<int64_t>>("tids")));
        });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "--cputime" }, _cputime_desc)
        .min_count(0)
        .max_count(2)
        .action([&](parser_t& p) {
            auto _v = p.get<std::vector<double>>("cputime");
            update_env(_env, "OMNITRACE_SAMPLING_CPUTIME", true);
            if(!_v.empty()) update_env(_env, "OMNITRACE_SAMPLING_CPUTIME_FREQ", _v.at(0));
            if(_v.size() > 1)
                update_env(_env, "OMNITRACE_SAMPLING_CPUTIME_DELAY", _v.at(1));
        });
    parser.add_argument({ "" }, "");
    parser.add_argument({ "--realtime" }, _realtime_desc)
        .min_count(0)
        .max_count(2)
        .action([&](parser_t& p) {
            auto _v = p.get<std::vector<double>>("realtime");
            update_env(_env, "OMNITRACE_SAMPLING_REALTIME", true);
            if(!_v.empty())
                update_env(_env, "OMNITRACE_SAMPLING_REALTIME_FREQ", _v.at(0));
            if(_v.size() > 1)
                update_env(_env, "OMNITRACE_SAMPLING_REALTIME_DELAY", _v.at(1));
        });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "-E", "--enable" }, "Enable these backends")
        .choices({ "all", "kokkosp", "mpip", "ompt", "rcclp", "rocm-smi", "roctracer",
                   "rocprofiler", "roctx", "mutex-locks", "spin-locks", "rw-locks" })
        .action([&](parser_t& p) {
            auto _v      = p.get<std::set<std::string>>("enable");
            auto _update = [&](const auto& _opt, bool _cond) {
                if(_cond || _v.count("all") > 0) update_env(_env, _opt, true);
            };
            _update("OMNITRACE_USE_KOKKOSP", _v.count("kokkosp") > 0);
            _update("OMNITRACE_USE_MPIP", _v.count("mpip") > 0);
            _update("OMNITRACE_USE_OMPT", _v.count("ompt") > 0);
            _update("OMNITRACE_USE_RCCLP", _v.count("rcclp") > 0);
            _update("OMNITRACE_USE_ROCTX", _v.count("roctx") > 0);
            _update("OMNITRACE_USE_ROCM_SMI", _v.count("rocm-smi") > 0);
            _update("OMNITRACE_USE_ROCTRACER", _v.count("roctracer") > 0);
            _update("OMNITRACE_USE_ROCPROFILER", _v.count("rocprofiler") > 0);
            _update("OMNITRACE_TRACE_THREAD_LOCKS", _v.count("mutex-locks") > 0);
            _update("OMNITRACE_TRACE_THREAD_RW_LOCKS", _v.count("rw-locks") > 0);
            _update("OMNITRACE_TRACE_THREAD_SPIN_LOCKS", _v.count("spin-locks") > 0);
        });

    parser.add_argument({ "-D", "--disable" }, "Disable these backends")
        .choices({ "all", "kokkosp", "mpip", "ompt", "rcclp", "rocm-smi", "roctracer",
                   "rocprofiler", "roctx", "mutex-locks", "spin-locks", "rw-locks" })
        .action([&](parser_t& p) {
            auto _v      = p.get<std::set<std::string>>("disable");
            auto _update = [&](const auto& _opt, bool _cond) {
                if(_cond || _v.count("all") > 0) update_env(_env, _opt, false);
            };
            _update("OMNITRACE_USE_KOKKOSP", _v.count("kokkosp") > 0);
            _update("OMNITRACE_USE_MPIP", _v.count("mpip") > 0);
            _update("OMNITRACE_USE_OMPT", _v.count("ompt") > 0);
            _update("OMNITRACE_USE_RCCLP", _v.count("rcclp") > 0);
            _update("OMNITRACE_USE_ROCTX", _v.count("roctx") > 0);
            _update("OMNITRACE_USE_ROCM_SMI", _v.count("rocm-smi") > 0);
            _update("OMNITRACE_USE_ROCTRACER", _v.count("roctracer") > 0);
            _update("OMNITRACE_USE_ROCPROFILER", _v.count("rocprofiler") > 0);
            _update("OMNITRACE_TRACE_THREAD_LOCKS", _v.count("mutex-locks") > 0);
            _update("OMNITRACE_TRACE_THREAD_RW_LOCKS", _v.count("rw-locks") > 0);
            _update("OMNITRACE_TRACE_THREAD_SPIN_LOCKS", _v.count("spin-locks") > 0);
        });

    parser.add_argument({ "" }, "");
    parser
        .add_argument({ "--cpus" },
                      "CPU IDs for frequency sampling. Supports integers and/or ranges")
        .dtype("int or range")
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_USE_PROCESS_SAMPLING", true);
            update_env(
                _env, "OMNITRACE_PROCESS_SAMPLING_CPUS",
                join(array_config{ "," }, p.get<std::vector<std::string>>("cpus")));
        });
    parser
        .add_argument({ "--gpus" },
                      "GPU IDs for SMI queries. Supports integers and/or ranges")
        .dtype("int or range")
        .action([&](parser_t& p) {
            update_env(
                _env, "OMNITRACE_PROCESS_SAMPLING_GPUS",
                join(array_config{ "," }, p.get<std::vector<std::string>>("gpus")));
        });

    parser.add_argument({ "" }, "");
    parser
        .add_argument({ "-C", "--cpu-events" },
                      "Set the CPU hardware counter events to record (ref: "
                      "`omnitrace-avail -H -c CPU`)")
        .action([&](parser_t& p) {
            auto _events =
                join(array_config{ "," }, p.get<std::vector<std::string>>("cpu-events"));
            update_env(_env, "OMNITRACE_PAPI_EVENTS", _events);
        });
    parser
        .add_argument({ "-G", "--gpu-events" },
                      "Set the GPU hardware counter events to record (ref: "
                      "`omnitrace-avail -H -c GPU`)")
        .action([&](parser_t& p) {
            auto _events =
                join(array_config{ "," }, p.get<std::vector<std::string>>("gpu-events"));
            update_env(_env, "OMNITRACE_ROCM_EVENTS", _events);
        });

    auto  _args = parser.parse_known_args(argc, argv);
    auto  _cmdc = std::get<1>(_args);
    auto* _cmdv = std::get<2>(_args);

    if(parser.exists("realtime") && !parser.exists("cputime"))
        update_env(_env, "OMNITRACE_SAMPLING_CPUTIME", false);
    if(parser.exists("profile") && parser.exists("flat-profile"))
        throw std::runtime_error(
            "Error! '--profile' argument conflicts with '--flat-profile' argument");

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
