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

/// @file
/// This provides generic functionality for constraining data collection within
/// a windows of time. E.g., delay, delay + duration, (delay + duration) * nrepeat
///
/// @todo Migrate delay/duration for sampling, process sampling, and causal profiling
/// to use this
///

#include "library/defines.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace omnitrace
{
namespace constraint
{
struct spec;

struct stages
{
    using functor_t = std::function<bool(const spec&)>;

    stages();
    OMNITRACE_DEFAULT_COPY_MOVE(stages)

    functor_t init    = [](const spec&) { return true; };
    functor_t wait    = [](const spec&) { return true; };
    functor_t start   = [](const spec&) { return true; };
    functor_t collect = [](const spec&) { return true; };
    functor_t stop    = [](const spec&) { return true; };
};

struct spec
{
    spec(double, double, uint64_t = 0, uint64_t = 1);
    spec(const std::string&);
    OMNITRACE_DEFAULT_OBJECT(spec)

    void operator()(const stages&) const;

    double   delay    = 0.0;
    double   duration = 0.0;
    uint64_t count    = 0;
    uint64_t repeat   = 1;
};

std::vector<spec>
get_trace_specs();

stages
get_trace_stages();
}  // namespace constraint
}  // namespace omnitrace
