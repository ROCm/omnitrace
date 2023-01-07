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
    using this_type = static_vector<Tp, N>;

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
        clear();
        for(auto&& itr : _v)
        {
            OMNITRACE_CONDITIONAL_THROW(m_size == N, "Error! %s has reached its capacity",
                                        demangle<static_vector>().c_str());
            m_data[m_size++] = itr;
        }
        return *this;
    }

    template <typename... Args>
    auto& emplace_back(Args&&... _v)
    {
        OMNITRACE_CONDITIONAL_THROW(m_size == N, "Error! %s has reached its capacity",
                                    demangle<static_vector>().c_str());
        auto _idx    = m_size++;
        m_data[_idx] = { std::forward<Args>(_v)... };
        return m_data[_idx];
    }

    template <typename Up>
    decltype(auto) push_back(Up&& _v)
    {
        return emplace_back(Tp{ std::forward<Up>(_v) });
    }

    void pop_back() { --m_size; }

    void clear() { m_size.store(0); }
    void reserve(size_t) noexcept {}
    void shrink_to_fit() noexcept {}
    auto capacity() noexcept { return N; }

    bool empty() const { return (m_size.load() == 0); }
    auto size() const { return m_size.load(); }

    auto begin() { return m_data.begin(); }
    auto begin() const { return m_data.begin(); }
    auto cbegin() const { return m_data.cbegin(); }

    auto end() { return m_data.begin() + m_size.load(); }
    auto end() const { return m_data.begin() + m_size.load(); }
    auto cend() const { return m_data.cbegin() + m_size.load(); }

    decltype(auto) operator[](size_t _idx) { return m_data[_idx]; }
    decltype(auto) operator[](size_t _idx) const { return m_data[_idx]; }

    decltype(auto) at(size_t _idx) { return m_data.at(_idx); }
    decltype(auto) at(size_t _idx) const { return m_data.at(_idx); }

    decltype(auto) front() { return m_data.front(); }
    decltype(auto) front() const { return m_data.front(); }
    decltype(auto) back() { return *(m_data.begin() + m_size - 1); }
    decltype(auto) back() const { return *(m_data.begin() + m_size - 1); }

    void swap(this_type& _v)
    {
        std::swap(m_data, _v.m_data);
        std::swap(m_size, _v.m_size);
    }

    friend void swap(this_type& _lhs, this_type& _rhs) { _lhs.swap(_rhs); }

private:
    std::atomic<size_t> m_size = 0;
    std::array<Tp, N>   m_data = {};
};
}  // namespace container
}  // namespace omnitrace
