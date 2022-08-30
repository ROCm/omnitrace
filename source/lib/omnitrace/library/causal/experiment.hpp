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

#pragma once

#include "library/causal/data.hpp"
#include "library/causal/progress_point.hpp"
#include "library/defines.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/utility/unwind.hpp>

#include <map>

namespace omnitrace
{
namespace causal
{
struct experiment
: comp::empty_base
, concepts::component
{
    using progress_points_t = std::map<tim::hash_value_t, progress_point::result>;

    static std::string label();
    static std::string description();

    TIMEMORY_DEFAULT_OBJECT(experiment)

    void start();
    void stop();

    bool              active          = false;
    int8_t            virtual_speedup = 0;  // 0-100 in multiples of 5
    uint64_t          start_time      = 0;
    uint64_t          duration        = 0;
    progress_stack    selection       = {};
    progress_points_t progress        = {};
};
}  // namespace causal
}  // namespace omnitrace
