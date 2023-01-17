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

#include <cstdint>
#include <string>

namespace omnitrace
{
namespace binary
{
struct scope_filter
{
    enum filter_mode : uint8_t
    {
        FILTER_INCLUDE = 0,
        FILTER_EXCLUDE
    };

    enum filter_scope : uint8_t
    {
        UNIVERSAL_FILTER = (1 << 0),
        BINARY_FILTER    = (1 << 1),
        SOURCE_FILTER    = (1 << 2),
        FUNCTION_FILTER  = (1 << 3)
    };

    filter_mode  mode       = FILTER_INCLUDE;
    filter_scope scope      = UNIVERSAL_FILTER;
    std::string  expression = {};

    bool operator()(const std::string& _value) const;
};
}  // namespace binary
}  // namespace omnitrace
