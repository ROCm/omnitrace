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

#include "omnitrace-causal.hpp"

#include "common/defines.h"
#include "common/delimit.hpp"
#include "common/environment.hpp"
#include "common/join.hpp"
#include "common/setup.hpp"

#include <timemory/environment.hpp>
#include <timemory/log/color.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/console.hpp>
#include <timemory/utility/join.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#if !defined(OMNITRACE_USE_ROCTRACER)
#    define OMNITRACE_USE_ROCTRACER 0
#endif

#if !defined(OMNITRACE_USE_ROCPROFILER)
#    define OMNITRACE_USE_ROCPROFILER 0
#endif

namespace color = tim::log::color;
using namespace timemory::join;
using tim::get_env;
using tim::log::colorized;
using tim::log::stream;

namespace
{
int  verbose       = 0;
auto updated_envs  = std::set<std::string_view>{};
auto original_envs = std::set<std::string>{};
auto child_pids    = std::set<pid_t>{};

inline signal_handler&
get_signal_handler(int _sig)
{
    static auto _v  = std::unordered_map<int, signal_handler>{};
    auto        itr = _v.emplace(_sig, signal_handler{});
    return itr.first->second;
}

void
create_signal_handler(int sig, signal_handler& sh, void (*func)(int))
{
    if(sig < 1) return;
    sh.m_custom_sigaction.sa_handler = func;
    sigemptyset(&sh.m_custom_sigaction.sa_mask);
    sh.m_custom_sigaction.sa_flags = SA_RESTART;
    if(sigaction(sig, &sh.m_custom_sigaction, &sh.m_original_sigaction) == -1)
    {
        std::cerr << "Failed to create signal handler for " << sig << std::endl;
    }
}

void
forward_signal(int sig)
{
    for(auto itr : child_pids)
    {
        TIMEMORY_PRINTF_WARNING(stderr, "[%i] Killing pid=%i with signal %i...\n",
                                getpid(), itr, sig);
        kill(itr, sig);
        diagnose_status(itr, wait_pid(itr));
    }
    signal(sig, SIG_DFL);
    kill(getpid(), sig);
}
}  // namespace

void
forward_signals(const std::set<int>& _signals)
{
    for(auto itr : _signals)
        create_signal_handler(itr, get_signal_handler(itr), &forward_signal);
}

void
add_child_pid(pid_t _v)
{
    child_pids.emplace(_v);
}

void
remove_child_pid(pid_t _v)
{
    child_pids.erase(_v);
}

int
wait_pid(pid_t _pid, int _opts)
{
    int   _status = 0;
    pid_t _pid_v  = -1;
    _opts |= WUNTRACED;
    do
    {
        if((_opts & WNOHANG) > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
        _pid_v = waitpid(_pid, &_status, _opts);
    } while(_pid <= 0);
    return _status;
}

int
diagnose_status(pid_t _pid, int _status)
{
    auto _verbose = get_env<int>("OMNITRACE_VERBOSE", 0);
    auto _debug   = get_env<bool>("OMNITRACE_DEBUG", false);

    if(_verbose >= 4)
        TIMEMORY_PRINTF(stderr, "[omnitrace-causal][%i] diagnosing status %i...\n", _pid,
                        _status);

    if(WIFEXITED(_status) && WEXITSTATUS(_status) == EXIT_SUCCESS)
    {
        if(_verbose >= 4 || (_debug && _verbose >= 2))
        {
            TIMEMORY_PRINTF(
                stderr,
                "[omnitrace-causal][%i] program terminated normally with exit code: %i\n",
                _pid, WEXITSTATUS(_status));
        }
        // normal terminatation
        return 0;
    }

    int ret = WEXITSTATUS(_status);
    if(WIFSTOPPED(_status))
    {
        int sig = WSTOPSIG(_status);
        // stopped with signal 'sig'
        if(_verbose >= 5)
        {
            TIMEMORY_PRINTF_WARNING(
                stderr,
                "[omnitrace-causal][%i] program stopped with signal %i. Exit code: %i\n",
                _pid, sig, ret);
        }
    }
    else if(WCOREDUMP(_status))
    {
        if(_verbose >= 5)
        {
            TIMEMORY_PRINTF_WARNING(stderr,
                                    "[omnitrace-causal][%i] program terminated and "
                                    "produced a core dump. Exit code: %i\n",
                                    _pid, ret);
        }
    }
    else if(WIFSIGNALED(_status))
    {
        ret = WTERMSIG(_status);
        if(_verbose >= 5)
        {
            TIMEMORY_PRINTF_WARNING(
                stderr,
                "[omnitrace-causal][%i] program terminated because it received a signal "
                "(%i) that was not handled. Exit code: %i\n",
                _pid, WTERMSIG(_status), ret);
        }
    }
    else if(WIFEXITED(_status) && WEXITSTATUS(_status))
    {
        if(ret == 127 && (_verbose >= 5))
        {
            TIMEMORY_PRINTF_WARNING(stderr, "[omnitrace-causal][%i] execv failed\n",
                                    _pid);
        }
        else if(_verbose >= 5)
        {
            TIMEMORY_PRINTF_WARNING(stderr,
                                    "[omnitrace-causal][%i] program terminated with a "
                                    "non-zero status. Exit code: %i\n",
                                    _pid, ret);
        }
    }
    else
    {
        if(_verbose >= 5)
            TIMEMORY_PRINTF_WARNING(
                stderr, "[omnitrace-causal][%i] program terminated abnormally.\n", _pid);
        ret = EXIT_FAILURE;
    }

    return ret;
}

std::string
get_realpath(const std::string& _v)
{
    auto* _tmp = realpath(_v.c_str(), nullptr);
    auto  _ret = std::string{ _tmp };
    free(_tmp);
    return _ret;
}

void
print_command(const std::vector<char*>& _argv)
{
    if(verbose >= 1)
        stream(std::cout, color::info())
            << "Executing '" << join(array_config{ " " }, _argv) << "'...\n";
}

std::vector<char*>
get_initial_environment()
{
    std::vector<char*> _env;
    if(environ != nullptr)
    {
        int idx = 0;
        while(environ[idx] != nullptr)
        {
            auto* _v = environ[idx++];
            original_envs.emplace(_v);
            _env.emplace_back(strdup(_v));
        }
    }

    update_env(_env, "LD_PRELOAD",
               get_realpath(get_internal_libpath("libomnitrace-dl.so")), true);

    auto* _dl_libpath =
        realpath(get_internal_libpath("libomnitrace-dl.so").c_str(), nullptr);
    auto* _omni_libpath =
        realpath(get_internal_libpath("libomnitrace.so").c_str(), nullptr);

    update_env(_env, "OMNITRACE_USE_SAMPLING", true);
    update_env(_env, "OMNITRACE_USE_CAUSAL", true);
    update_env(_env, "OMNITRACE_USE_PERFETTO", false);
    update_env(_env, "OMNITRACE_USE_TIMEMORY", false);
    update_env(_env, "OMNITRACE_USE_PROCESS_SAMPLING", false);
    update_env(_env, "OMNITRACE_CRITICAL_TRACE", false);

    // update_env(_env, "OMNITRACE_USE_PID", false);
    // update_env(_env, "OMNITRACE_TIME_OUTPUT", false);
    // update_env(_env, "OMNITRACE_OUTPUT_PATH", "omnitrace-output/%tag%/%launch_time%");

#if defined(OMNITRACE_USE_ROCTRACER) || defined(OMNITRACE_USE_ROCPROFILER)
    update_env(_env, "HSA_TOOLS_LIB", _dl_libpath);
    if(!getenv("HSA_TOOLS_REPORT_LOAD_FAILURE"))
        update_env(_env, "HSA_TOOLS_REPORT_LOAD_FAILURE", "1");
#endif

#if defined(OMNITRACE_USE_ROCPROFILER)
    update_env(_env, "ROCP_TOOL_LIB", _omni_libpath);
    if(!getenv("ROCP_HSA_INTERCEPT")) update_env(_env, "ROCP_HSA_INTERCEPT", "1");
#endif

#if defined(OMNITRACE_USE_OMPT)
    if(!getenv("OMP_TOOL_LIBRARIES"))
        update_env(_env, "OMP_TOOL_LIBRARIES", _dl_libpath, true);
#endif

    free(_dl_libpath);
    free(_omni_libpath);

    return _env;
}

std::string
get_internal_libpath(const std::string& _lib)
{
    auto _exe = std::string_view{ realpath("/proc/self/exe", nullptr) };
    auto _pos = _exe.find_last_of('/');
    auto _dir = std::string{ "./" };
    if(_pos != std::string_view::npos) _dir = _exe.substr(0, _pos);
    return omnitrace::common::join("/", _dir, "..", "lib", _lib);
}

void
print_updated_environment(std::vector<char*> _env)
{
    // if(get_env<int>("OMNITRACE_VERBOSE", 0) < 0 &&
    //   !get_env<bool>("OMNITRACE_DEBUG", false))
    //    return;

    std::sort(_env.begin(), _env.end(), [](auto* _lhs, auto* _rhs) {
        if(!_lhs) return false;
        if(!_rhs) return true;
        return std::string_view{ _lhs } < std::string_view{ _rhs };
    });

    std::vector<char*> _updates = {};
    std::vector<char*> _general = {};

    for(auto* itr : _env)
    {
        if(itr == nullptr) continue;

        auto _is_omni = (std::string_view{ itr }.find("OMNITRACE") == 0);
        auto _updated = false;
        for(const auto& vitr : updated_envs)
        {
            if(std::string_view{ itr }.find(vitr) == 0)
            {
                _updated = true;
                break;
            }
        }

        if(_updated)
            _updates.emplace_back(itr);
        else if(verbose >= 1 && _is_omni)
            _general.emplace_back(itr);
    }

    if(_general.size() + _updates.size() == 0 || verbose < 0) return;

    std::cerr << std::endl;

    for(auto& itr : _general)
        stream(std::cerr, color::source()) << itr << "\n";
    for(auto& itr : _updates)
        stream(std::cerr, color::source()) << itr << "\n";

    std::cerr << std::endl;
}

template <typename Tp>
void
update_env(std::vector<char*>& _environ, std::string_view _env_var, Tp&& _env_val,
           bool _append, std::string_view _join_delim)
{
    updated_envs.emplace(_env_var);

    auto _key = join("", _env_var, "=");
    for(auto& itr : _environ)
    {
        if(!itr) continue;
        if(std::string_view{ itr }.find(_key) == 0)
        {
            if(_append)
            {
                if(std::string_view{ itr }.find(join("", _env_val)) ==
                   std::string_view::npos)
                {
                    auto _val = std::string{ itr }.substr(_key.length());
                    free(itr);
                    itr = strdup(
                        join('=', _env_var, join(_join_delim, _env_val, _val)).c_str());
                }
            }
            else
            {
                free(itr);
                itr = strdup(omnitrace::common::join('=', _env_var, _env_val).c_str());
            }
            return;
        }
    }
    _environ.emplace_back(
        strdup(omnitrace::common::join('=', _env_var, _env_val).c_str()));
}

void
remove_env(std::vector<char*>& _environ, std::string_view _env_var)
{
    auto _key   = join("", _env_var, "=");
    auto _match = [&_key](auto itr) { return std::string_view{ itr }.find(_key) == 0; };

    _environ.erase(std::remove_if(_environ.begin(), _environ.end(), _match),
                   _environ.end());

    for(const auto& itr : original_envs)
    {
        if(std::string_view{ itr }.find(_key) == 0)
            _environ.emplace_back(strdup(itr.c_str()));
    }
}

std::vector<char*>
parse_args(int argc, char** argv, std::vector<char*>& _env,
           std::vector<std::map<std::string_view, std::string>>& _causal_envs)
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

    auto* _dl_libpath =
        realpath(get_internal_libpath("libomnitrace-dl.so").c_str(), nullptr);
    auto* _omni_libpath =
        realpath(get_internal_libpath("libomnitrace.so").c_str(), nullptr);

    auto parser = parser_t(argv[0]);

    parser.on_error([](parser_t&, const parser_err_t& _err) {
        stream(std::cerr, color::fatal()) << _err << "\n";
        exit(EXIT_FAILURE);
    });

    auto _realtime_reqs = (get_env("HSA_ENABLE_INTERRUPT", std::string{}, false).empty())
                              ? std::vector<std::string>{ "hsa-interrupt" }
                              : std::vector<std::string>{};

#if OMNITRACE_USE_ROCTRACER == 0 && OMNITRACE_USE_ROCPROFILER == 0
    _realtime_reqs.clear();
#endif

    const auto* _trace_policy_desc =
        R"(Policy for new data when the buffer size limit is reached:
    %{INDENT}%- discard     : new data is ignored
    %{INDENT}%- ring_buffer : new data overwrites oldest data)";

    auto _add_separator = [&](std::string _v, const std::string& _desc) {
        parser.add_argument({ "" }, "");
        parser
            .add_argument({ join("", "[", _v, "]") },
                          (_desc.empty()) ? _desc : join({ "", "(", ")" }, _desc))
            .color(tim::log::color::info());
        parser.add_argument({ "" }, "");
    };

    parser.enable_help();

    auto _cols = std::get<0>(tim::utility::console::get_columns());
    if(_cols > parser.get_help_width() + 8)
        parser.set_description_width(
            std::min<int>(_cols - parser.get_help_width() - 8, 120));

    _add_separator("DEBUG OPTIONS", "");
    parser.add_argument({ "--monochrome" }, "Disable colorized output")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            auto _colorized = !p.get<bool>("monochrome");
            colorized()     = _colorized;
            p.set_use_color(_colorized);
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

    _add_separator("GENERAL OPTIONS", "");
    parser.add_argument({ "-c", "--config" }, "Base configuration file")
        .min_count(0)
        .dtype("filepath")
        .action([&](parser_t& p) {
            update_env(
                _env, "OMNITRACE_CONFIG_FILE",
                join(array_config{ ":" }, p.get<std::vector<std::string>>("config")));
        });

    int64_t _niterations      = 1;
    auto    _virtual_speedups = std::vector<std::string>{};
    auto    _fileline_scopes  = std::vector<std::string>{};
    auto    _function_scopes  = std::vector<std::string>{};
    auto    _binary_scopes    = std::vector<std::string>{};
    auto    _source_scopes    = std::vector<std::string>{};

    _add_separator("CAUSAL PROFILING OPTIONS", "");

    parser.add_argument({ "-m", "--mode" }, "Causal profiling mode")
        .count(1)
        .dtype("string")
        .choices({ "function", "line" })
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_CAUSAL_MODE", p.get<std::string>("mode"));
        });

    parser
        .add_argument({ "-o", "--output-name" },
                      "Output filename of causal profiling data w/o extension")
        .min_count(1)
        .dtype("filename")
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_CAUSAL_FILE", p.get<std::string>("output-name"));
        });

    parser
        .add_argument({ "-e", "--end-to-end" },
                      "Single causal experiment for the entire application runtime")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_CAUSAL_END_TO_END", p.get<bool>("end-to-end"));
        });

    parser
        .add_argument({ "-n", "--iterations" }, "Number of times to repeat the variants")
        .count(1)
        .dtype("int")
        .action([&](parser_t& p) { _niterations = p.get<int64_t>("iterations"); });

    parser
        .add_argument({ "-s", "--speedups" },
                      "Pool of virtual speedups to sample from during experimentation. "
                      "Each space designates a group and multiple speedups can be "
                      "grouped together by commas, e.g. -s 0 0,10,20-50 is two groups: "
                      "group #1 is '0' and group #2 is '0 10 20 25 30 35 40 45 50'")
        .min_count(0)
        .max_count(-1)
        .dtype("integers")
        .action([&](parser_t& p) {
            _virtual_speedups = p.get<std::vector<std::string>>("speedups");
        });

    parser
        .add_argument({ "-L", "--fileline-scope" },
                      "Restricts causal experiments to the <file>:<line> combos matching "
                      "the list of "
                      "regular expressions. Each space designates a group and multiple "
                      "scopes can be grouped together with a semi-colon")
        .min_count(0)
        .max_count(-1)
        .dtype("regex-list")
        .action([&](parser_t& p) {
            _fileline_scopes = p.get<std::vector<std::string>>("fileline-scope");
        });

    parser
        .add_argument(
            { "-F", "--function-scope" },
            "Restricts causal experiments to the functions matching the list of "
            "regular expressions. Each space designates a group and multiple "
            "scopes can be grouped together with a semi-colon")
        .min_count(0)
        .max_count(-1)
        .dtype("regex-list")
        .action([&](parser_t& p) {
            _function_scopes = p.get<std::vector<std::string>>("function-scope");
        });

    parser
        .add_argument({ "-B", "--binary-scope" },
                      "Restricts causal experiments to the binaries matching the list of "
                      "regular expressions. Each space designates a group and multiple "
                      "scopes can be grouped together with a semi-colon")
        .min_count(0)
        .max_count(-1)
        .dtype("integers")
        .action([&](parser_t& p) {
            _binary_scopes = p.get<std::vector<std::string>>("binary-scope");
        });

    parser
        .add_argument(
            { "-S", "--source-scope" },
            "Restricts causal experiments to the source files matching the list of "
            "regular expressions. Each space designates a group and multiple "
            "scopes can be grouped together with a semi-colon")
        .min_count(0)
        .max_count(-1)
        .dtype("integers")
        .action([&](parser_t& p) {
            _source_scopes = p.get<std::vector<std::string>>("source-scope");
        });

#if OMNITRACE_HIP_VERSION < 50300
    update_env(_env, "HSA_ENABLE_INTERRUPT", 0);
#endif

    auto _inpv = std::vector<char*>{};
    auto _outv = std::vector<char*>{};
    bool _hash = false;
    for(int i = 0; i < argc; ++i)
    {
        if(_hash)
        {
            _outv.emplace_back(argv[i]);
        }
        else if(std::string_view{ argv[i] } == "--")
        {
            _hash = true;
        }
        else
        {
            _inpv.emplace_back(argv[i]);
        }
    }

    auto _cerr = parser.parse_args(_inpv.size(), _inpv.data());
    if(help_check(parser, argc, argv))
        help_action(parser);
    else if(_cerr)
        throw std::runtime_error(_cerr.what());

    free(_dl_libpath);
    free(_omni_libpath);

    if(_niterations < 1) _niterations = 1;
    auto _get_size = [](const auto& _v) { return std::max<size_t>(_v.size(), 1); };
    _causal_envs.resize(_niterations * _get_size(_virtual_speedups) *
                        _get_size(_fileline_scopes) * _get_size(_function_scopes) *
                        _get_size(_binary_scopes) * _get_size(_source_scopes));

    auto _fill = [&_causal_envs](std::string_view _env_var, const auto& _data) {
        if(_data.empty()) return;
        auto   _modulo = _data.size();
        size_t _n      = 0;
        for(auto& eitr : _causal_envs)
            eitr[_env_var] = _data.at(_n++ % _modulo);
    };

    _fill("OMNITRACE_CAUSAL_FIXED_SPEEDUP", _virtual_speedups);
    _fill("OMNITRACE_CAUSAL_FILELINE_SCOPE", _fileline_scopes);
    _fill("OMNITRACE_CAUSAL_FUNCTION_SCOPE", _function_scopes);
    _fill("OMNITRACE_CAUSAL_BINARY_SCOPE", _binary_scopes);
    _fill("OMNITRACE_CAUSAL_SOURCE_SCOPE", _source_scopes);

    return _outv;
}

// explicit instantiation for usage in omnitrace-causal.cpp
template void
update_env(std::vector<char*>&, std::string_view, const std::string& _env_val,
           bool _append, std::string_view);
