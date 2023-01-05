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

#include <timemory/utility/demangle.hpp>

#include <array>
#include <atomic>
#include <cstdlib>

namespace omnitrace
{
namespace container
{
template <typename Tp, size_t N>
struct static_vector
{
    static_vector()                         = default;
    static_vector(const static_vector&)     = default;
    static_vector(static_vector&&) noexcept = default;
    static_vector& operator=(const static_vector&) = default;
    static_vector& operator=(static_vector&&) noexcept = default;

    static_vector(size_t _n, Tp _v = {})
    {
        m_size.store(_n);
        m_data.fill(_v);
    }

    static_vector& operator=(std::initializer_list<Tp>&& _v)
    {
        reset();
        for(auto itr : _v)
        {
            OMNITRACE_CONDITIONAL_THROW(m_size == N, "Error! %s has reached its capacity",
                                        demangle<static_vector>().c_str());
            m_data[m_size++] = itr;
        }
        purge(Tp{});
        return *this;
    }

    template <typename Up>
    auto& emplace_back(Up&& _v)
    {
        OMNITRACE_CONDITIONAL_THROW(m_size == N, "Error! %s has reached its capacity",
                                    demangle<static_vector>().c_str());
        auto _idx    = m_size++;
        m_data[_idx] = std::forward<Up>(_v);
        return m_data[_idx];
    }

    // reset the size but do not clear data
    void reset() { m_size.store(0); }

    // purge old values
    void purge(Tp _v = {})
    {
        for(size_t i = m_size; i < N; ++i)
        {
            m_data.at(i) = _v;
        }
    }

    bool empty() const { return (m_size.load() == 0); }
    auto size() const { return m_size.load(); }
    auto begin() { return m_data.begin(); }
    auto end() { return m_data.begin() + m_size.load(); }
    auto begin() const { return m_data.begin(); }
    auto end() const { return m_data.begin() + m_size.load(); }

    decltype(auto) at(size_t _idx) { return m_data.at(_idx); }
    decltype(auto) at(size_t _idx) const { return m_data.at(_idx); }

private:
    std::atomic<size_t> m_size = 0;
    std::array<Tp, N>   m_data = {};
};
}  // namespace container
}  // namespace omnitrace
