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

#include "omnitrace-run.hpp"

#include "common/defines.h"
#include "common/delimit.hpp"
#include "common/environment.hpp"
#include "common/join.hpp"
#include "common/setup.hpp"
#include "core/argparse.hpp"
#include "core/config.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"

#include <timemory/environment.hpp>
#include <timemory/log/color.hpp>
#include <timemory/settings/vsettings.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/console.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>

#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace color    = ::tim::log::color;
namespace filepath = ::tim::filepath;  // NOLINT
namespace console  = ::tim::utility::console;
namespace argparse = ::tim::argparse;
using namespace timemory::join;
using tim::get_env;
using tim::log::stream;

namespace std
{
std::string
to_string(bool _v)
{
    return (_v) ? "true" : "false";
}
}  // namespace std

namespace
{
int  verbose       = 0;
auto updated_envs  = std::set<std::string_view>{};
auto original_envs = std::set<std::string>{};
auto child_pids    = std::set<pid_t>{};
auto launcher      = std::string{};

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
        TIMEMORY_PRINTF_WARNING(stderr, "Killing pid=%i with signal %i...\n", itr, sig);
        kill(itr, sig);
        diagnose_status(itr, wait_pid(itr));
    }
    signal(sig, SIG_DFL);
    kill(getpid(), sig);
}
}  // namespace

int
get_verbose()
{
    verbose = get_env("OMNITRACE_CAUSAL_VERBOSE",
                      get_env<int>("OMNITRACE_VERBOSE", verbose, false));
    auto _debug =
        get_env("OMNITRACE_CAUSAL_DEBUG", get_env<bool>("OMNITRACE_DEBUG", false, false));
    if(_debug) verbose += 8;
    return verbose;
}

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
    auto _verbose = get_verbose();
    if(_verbose >= 3)
    {
        fflush(stderr);
        fflush(stdout);
        std::cout << std::flush;
        std::cerr << std::flush;
    }

    bool _normal_exit      = (WIFEXITED(_status) > 0);
    bool _unhandled_signal = (WIFSIGNALED(_status) > 0);
    bool _core_dump        = (WCOREDUMP(_status) > 0);
    bool _stopped          = (WIFSTOPPED(_status) > 0);
    int  _exit_status      = WEXITSTATUS(_status);
    int  _stop_signal      = (_stopped) ? WSTOPSIG(_status) : 0;
    int  _ec               = (_unhandled_signal) ? WTERMSIG(_status) : 0;

    if(_verbose >= 4)
    {
        TIMEMORY_PRINTF_INFO(
            stderr,
            "diagnosing status for process %i :: status: %i... normal exit: %s, "
            "unhandled signal: %s, core dump: %s, stopped: %s, exit status: %i, stop "
            "signal: %i, exit code: %i\n",
            _pid, _status, std::to_string(_normal_exit).c_str(),
            std::to_string(_unhandled_signal).c_str(), std::to_string(_core_dump).c_str(),
            std::to_string(_stopped).c_str(), _exit_status, _stop_signal, _ec);
    }
    else if(_verbose >= 3)
    {
        TIMEMORY_PRINTF_INFO(stderr,
                             "diagnosing status for process %i :: status: %i ...\n", _pid,
                             _status);
    }

    if(!_normal_exit)
    {
        if(_ec == 0) _ec = EXIT_FAILURE;
        if(_verbose >= 0)
        {
            TIMEMORY_PRINTF_FATAL(
                stderr, "process %i terminated abnormally. exit code: %i\n", _pid, _ec);
        }
    }

    if(_stopped)
    {
        if(_verbose >= 0)
        {
            TIMEMORY_PRINTF_FATAL(stderr,
                                  "process %i stopped with signal %i. exit code: %i\n",
                                  _pid, _stop_signal, _ec);
        }
    }

    if(_core_dump)
    {
        if(_verbose >= 0)
        {
            TIMEMORY_PRINTF_FATAL(
                stderr, "process %i terminated and produced a core dump. exit code: %i\n",
                _pid, _ec);
        }
    }

    if(_unhandled_signal)
    {
        if(_verbose >= 0)
        {
            TIMEMORY_PRINTF_FATAL(stderr,
                                  "process %i terminated because it received a signal "
                                  "(%i) that was not handled. exit code: %i\n",
                                  _pid, _ec, _ec);
        }
    }

    if(!_normal_exit && _exit_status > 0)
    {
        if(_verbose >= 0)
        {
            if(_exit_status == 127)
            {
                TIMEMORY_PRINTF_FATAL(
                    stderr, "execv in process %i failed. exit code: %i\n", _pid, _ec);
            }
            else
            {
                TIMEMORY_PRINTF_FATAL(
                    stderr,
                    "process %i terminated with a non-zero status. exit code: %i\n", _pid,
                    _ec);
            }
        }
    }

    return _ec;
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
print_command(const std::vector<char*>& _argv, std::string_view _prefix)
{
    if(verbose >= 1)
        stream(std::cout, color::info())
            << _prefix << "Executing '" << join(array_config{ " " }, _argv) << "'...\n";

    std::cerr << color::end() << std::flush;
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

    return _env;
}

void
prepare_command_for_run(char* _exe, std::vector<char*>& _argv)
{
    if(!launcher.empty())
    {
        bool _injected = false;
        auto _new_argv = std::vector<char*>{};
        for(auto* itr : _argv)
        {
            if(!_injected && std::regex_search(itr, std::regex{ launcher }))
            {
                _new_argv.emplace_back(_exe);
                _new_argv.emplace_back(strdup("--"));
                _injected = true;
            }
            _new_argv.emplace_back(itr);
        }

        if(!_injected)
        {
            throw std::runtime_error(
                join("", "omnitrace-run was unable to match \"", launcher,
                     "\" to any arguments on the command line: \"",
                     join(array_config{ " ", "", "" }, _argv), "\""));
        }

        std::swap(_argv, _new_argv);
    }
}

void
prepare_environment_for_run(std::vector<char*>& _env)
{
    if(launcher.empty())
    {
        update_env(_env, "LD_PRELOAD",
                   join(":", get_realpath(get_internal_libpath("libomnitrace-dl.so"))),
                   true);
    }
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
print_updated_environment(std::vector<char*> _env, std::string_view _prefix)
{
    if(get_verbose() < 0) return;

    std::sort(_env.begin(), _env.end(), [](auto* _lhs, auto* _rhs) {
        if(!_lhs) return false;
        if(!_rhs) return true;
        return std::string_view{ _lhs } < std::string_view{ _rhs };
    });

    std::vector<std::string_view> _updates = {};
    std::vector<std::string_view> _general = {};

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
        stream(std::cerr, color::source()) << _prefix << itr << "\n";
    for(auto& itr : _updates)
        stream(std::cerr, color::source()) << _prefix << itr << "\n";

    std::cerr << color::end() << std::flush;
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
                    if(_env_var == "LD_PRELOAD")
                    {
                        itr =
                            strdup(join('=', _env_var, join(_join_delim, _val, _env_val))
                                       .c_str());
                    }
                    else
                    {
                        itr =
                            strdup(join('=', _env_var, join(_join_delim, _env_val, _val))
                                       .c_str());
                    }
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

template <typename Tp>
void
add_default_env(std::vector<char*>& _environ, std::string_view _env_var, Tp&& _env_val)
{
    auto _key = join("", _env_var, "=");
    for(auto& itr : _environ)
    {
        if(!itr) continue;
        if(std::string_view{ itr }.find(_key) == 0) return;
    }

    updated_envs.emplace(_env_var);
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
parse_args(int argc, char** argv, std::vector<char*>& _env)
{
    using parser_t     = argparse::argument_parser;
    using parser_err_t = typename parser_t::result_type;

    omnitrace::set_state(omnitrace::State::Init);
    omnitrace::config::configure_settings(false);

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

    const auto* _desc = R"desc(
    Command line interface to omnitrace configuration.
    )desc";

    auto parser = parser_t{ basename(argv[0]), _desc };

    parser.on_error([](parser_t&, const parser_err_t& _err) {
        stream(std::cerr, color::fatal()) << _err << "\n";
        exit(EXIT_FAILURE);
    });

    parser.enable_help();
    parser.enable_version("omnitrace-run", "v" OMNITRACE_VERSION_STRING,
                          OMNITRACE_GIT_DESCRIBE, OMNITRACE_GIT_REVISION);

    auto _cols = std::get<0>(console::get_columns());
    if(_cols > parser.get_help_width() + 8)
        parser.set_description_width(
            std::min<int>(_cols - parser.get_help_width() - 8, 120));

    /*
    auto _added = std::set<std::string>{};

    auto _get_name = [](const std::shared_ptr<tim::vsettings>& itr) {
        auto _name = itr->get_name();
        auto _pos  = std::string::npos;
        while((_pos = _name.find('_')) != std::string::npos)
            _name = _name.replace(_pos, 1, "-");
        return _name;
    };

    auto _add_option = [&parser, &_env,
                        &_added](const std::string&                     _name,
                                 const std::shared_ptr<tim::vsettings>& itr) {
        if(_added.find(_name) != _added.end()) return false;

        _added.emplace(_name);

        auto _opt_name = std::string{ "--" } + _name;
        itr->set_command_line({ _opt_name });
        auto* _arg = static_cast<parser_t::argument*>(itr->add_argument(parser));
        if(_arg)
        {
            _arg->action([&](parser_t& p) {
                using namespace timemory::join;
                update_env(_env, itr->get_env_name(),
                           join(array_config{ " ", "", "" },
                                p.get<std::vector<std::string>>(_name)));
            });
        }
        else
        {
            TIMEMORY_PRINTF_WARNING(stderr, "Warning! Option %s (%s) is not enabled\n",
                                    _name.c_str(), itr->get_env_name().c_str());
            parser.add_argument({ _opt_name }, itr->get_description())
                .action([&](parser_t& p) {
                    using namespace timemory::join;
                    update_env(_env, itr->get_env_name(),
                               join(array_config{ " ", "", "" },
                                    p.get<std::vector<std::string>>(_name)));
                });
        }
        return true;
    };

    auto _category_count_map = std::unordered_map<std::string, uint32_t>{};
    auto _settings           = std::vector<std::shared_ptr<tim::vsettings>>{};
    for(auto& itr : *omnitrace::settings::instance())
    {
        if(itr.second->get_categories().count("omnitrace") == 0) continue;
        if(itr.second->get_categories().count("deprecated") > 0) continue;
        if(itr.second->get_hidden()) continue;

        itr.second->set_enabled(true);
        _settings.emplace_back(itr.second);

        if(itr.second->get_name() == "papi_events")
        {
            auto _choices = itr.second->get_choices();
            _choices.erase(
                std::remove_if(_choices.begin(), _choices.end(),
                               [](const auto& itr) {
                                   return std::regex_search(
                                              itr,
                                              std::regex{ "[A-Za-z0-9]:([A-Za-z_]+)" }) ||
                                          std::regex_search(itr, std::regex{ "io:::" });
                               }),
                _choices.end());
            _choices.emplace_back(
                "... run `omnitrace-avail -H -c CPU` for full list ...");
            itr.second->set_choices(_choices);
        }

        for(const auto& citr : itr.second->get_categories())
        {
            if(std::regex_search(
                   citr, std::regex{
                             "omnitrace|timemory|^(native|custom|advanced|analysis)$" }))
                continue;
            _category_count_map[citr] += 1;
        }
    }

    auto _category_count_vec = std::vector<std::string>{};
    for(const auto& itr : _category_count_map)
        _category_count_vec.emplace_back(itr.first);

    std::sort(_category_count_vec.begin(), _category_count_vec.end(),
              [&_category_count_map](const auto& _lhs, const auto& _rhs) {
                  auto _lhs_v = _category_count_map.at(_lhs);
                  auto _rhs_v = _category_count_map.at(_rhs);
                  if(_lhs_v == _rhs_v) return _lhs < _rhs;
                  return _lhs_v > _rhs_v;
              });

    auto _groups =
        std::unordered_map<std::string, std::vector<std::shared_ptr<tim::vsettings>>>{};
    for(const auto& citr : _category_count_vec)
    {
        _groups[citr] = {};
        for(const auto& itr : _settings)
        {
            if(itr->get_categories().count(citr) > 0) _groups[citr].emplace_back(itr);
        }
        _settings.erase(std::remove_if(_settings.begin(), _settings.end(),
                                       [&citr](const auto& itr) {
                                           return itr->get_categories().count(citr) > 0;
                                       }),
                        _settings.end());
    }

    for(const auto& citr : _category_count_vec)
    {
        auto _group = _groups.at(citr);
        if(_group.empty()) continue;

        auto _group_name = citr;
        for(auto& c : _group_name)
            c = toupper(c);

        std::sort(_group.begin(), _group.end(), [](const auto& _lhs, const auto& _rhs) {
            auto _lhs_v = _lhs->get_name();
            auto _rhs_v = _rhs->get_name();
            if(_lhs_v.length() > 4 && _rhs_v.length() > 4 &&
               _lhs_v.substr(0, 4) == _rhs_v.substr(0, 4))
                return _lhs_v < _rhs_v;
            return _lhs_v.length() < _rhs_v.length();
        });

        parser.start_group(_group_name);
        for(const auto& itr : _group)
            _add_option(_get_name(itr), itr);
        parser.end_group();
    }
    */

    using parser_data_t = omnitrace::argparse::parser_data;

    auto _parser_data = parser_data_t{};
    omnitrace::argparse::init_parser(_parser_data);
    omnitrace::argparse::add_core_arguments(parser, _parser_data);
    omnitrace::argparse::add_extended_arguments(parser, _parser_data);

    auto _inpv = std::vector<char*>{};
    auto _outv = std::vector<char*>{};
    bool _hash = false;
    for(int i = 0; i < argc; ++i)
    {
        if(argv[i] == nullptr)
        {
            continue;
        }
        else if(_hash)
        {
            _outv.emplace_back(strdup(argv[i]));
        }
        else if(std::string_view{ argv[i] } == "--")
        {
            _hash = true;
        }
        else
        {
            _inpv.emplace_back(strdup(argv[i]));
        }
    }

    std::cerr << "Input: ";
    for(auto& itr : _inpv)
        std::cerr << " " << itr;
    std::cerr << std::endl;

    auto _cerr = parser.parse_args(_inpv.size(), _inpv.data());
    if(help_check(parser, argc, argv))
        help_action(parser);
    else if(_cerr)
        throw std::runtime_error(_cerr.what());

    _env          = _parser_data.current;
    updated_envs  = _parser_data.updated;
    original_envs = _parser_data.initial;

    return _outv;
}

// explicit instantiation for usage in omnitrace-run.cpp
template void
update_env(std::vector<char*>&, std::string_view, const std::string& _env_val,
           bool _append, std::string_view);
