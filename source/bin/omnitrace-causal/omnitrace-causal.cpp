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

#include <timemory/log/macros.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <string_view>
#include <unistd.h>

int
main(int argc, char** argv)
{
    auto _base_env   = get_initial_environment();
    auto _causal_env = std::vector<std::map<std::string_view, std::string>>{};

    bool _has_double_hyphen = false;
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string_view{ argv[i] };
        if(_arg == "--" || _arg == "-?" || _arg == "-h" || _arg == "--help")
            _has_double_hyphen = true;
    }

    std::vector<char*> _argv = {};
    if(_has_double_hyphen)
    {
        _argv = parse_args(argc, argv, _base_env, _causal_env);
    }
    else
    {
        _argv.reserve(argc);
        for(int i = 1; i < argc; ++i)
            _argv.emplace_back(argv[i]);
        _causal_env.resize(1);
    }

    if(!_argv.empty())
    {
        forward_signals({ SIGINT, SIGTERM, SIGQUIT });
        for(auto& citr : _causal_env)
        {
            auto _pid = fork();
            TIMEMORY_PRINTF_WARNING(
                stderr, "[omnitrace-causal][%i] PID=%i returned %i from fork...\n",
                getppid(), getpid(), _pid);
            if(_pid == 0)
            {
                auto _env = _base_env;
                for(const auto& eitr : citr)
                    update_env(_env, eitr.first, eitr.second);
                print_updated_environment(_env);
                print_command(_argv);
                _argv.emplace_back(nullptr);
                _env.emplace_back(nullptr);
                return execvpe(_argv.front(), _argv.data(), _env.data());
            }
            else
            {
                add_child_pid(_pid);
                auto _status = wait_pid(_pid);
                auto _ret    = diagnose_status(_pid, _status);
                remove_child_pid(_pid);
                if(_ret != 0) return _ret;
            }
        }
    }
}
