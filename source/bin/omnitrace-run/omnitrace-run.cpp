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

#include <timemory/log/macros.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string_view>
#include <unistd.h>

int
main(int argc, char** argv)
{
    auto _base_env = get_initial_environment();

    bool _has_double_hyphen = false;
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string_view{ argv[i] };
        if(_arg == "--" || _arg == "-?" || _arg == "-h" || _arg == "--help" ||
           _arg == "--version")
            _has_double_hyphen = true;
    }

    std::vector<char*> _argv = {};
    if(_has_double_hyphen)
    {
        _argv = parse_args(argc, argv, _base_env);
    }
    else
    {
        _argv.reserve(argc);
        for(int i = 1; i < argc; ++i)
            _argv.emplace_back(argv[i]);
    }

    prepare_command_for_run(argv[0], _argv);
    prepare_environment_for_run(_base_env);

    if(!_argv.empty())
    {
        forward_signals({ SIGINT, SIGTERM, SIGQUIT });

        print_updated_environment(_base_env, "OMNITRACE: ");
        print_command(_argv, "OMNITRACE: ");
        _argv.emplace_back(nullptr);
        _base_env.emplace_back(nullptr);
        return execvpe(_argv.front(), _argv.data(), _base_env.data());
    }
}
