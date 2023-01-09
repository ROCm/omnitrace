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

#include "library/common.hpp"
#include "library/concepts.hpp"
#include "library/defines.hpp"

#include <memory>
#include <optional>
#include <utility>

namespace omnitrace
{
template <typename Tp>
struct generate
{
    using type = Tp;

    template <typename... Args>
    auto operator()(Args&&... _args) const
    {
        if constexpr(concepts::is_unique_pointer<Tp>::value)
        {
            using value_type = typename type::element_type;
            if constexpr(std::is_constructible<value_type, Args...>::value)
            {
                return type{ new value_type{ std::forward<Args>(_args)... } };
            }
            else
            {
                return type{ new value_type{ invoke(std::forward<Args>(_args))... } };
            }
        }
        else
        {
            if constexpr(std::is_constructible<type, Args...>::value)
            {
                return type{ std::forward<Args>(_args)... };
            }
            else
            {
                return type{ invoke(std::forward<Args>(_args))... };
            }
        }
    }

private:
    template <typename Up>
    static auto invoke(Up&& _v, int,
                       std::enable_if_t<std::is_invocable<Up>::value, int> = 0)
        -> decltype(std::forward<Up>(_v)())
    {
        return std::forward<Up>(_v)();
    }

    template <typename Up>
    static auto&& invoke(Up&& _v, long)
    {
        return std::forward<Up>(_v);
    }

    template <typename Up>
    static decltype(auto) invoke(Up&& _v)
    {
        return invoke(std::forward<Up>(_v), 0);
    }
};

struct construct_on_thread
{
    int64_t index = threading::get_id();
};
}  // namespace omnitrace
