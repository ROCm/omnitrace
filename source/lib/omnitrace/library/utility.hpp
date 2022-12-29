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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace omnitrace
{
namespace utility
{
/// provides an alternative thread index for when using threading::get_id() is not
/// desirable
inline auto
get_thread_index()
{
    static std::atomic<int64_t> _c{ 0 };
    static thread_local int64_t _v = _c++;
    return _v;
}

/// fills any array with the result of the functor
template <size_t N, typename FuncT>
inline auto
get_filled_array(FuncT&& _func)
{
    using Tp = std::decay_t<decltype(_func())>;
    std::array<Tp, N> _v{};
    for(auto& itr : _v)
        itr = std::move(_func());
    return _v;
}

/// returns a vector with a preallocated buffer
template <typename... Tp>
inline auto
get_reserved_vector(size_t _n)
{
    std::vector<Tp...> _v{};
    _v.reserve(_n);
    return _v;
}

template <typename Tp, size_t Offset>
struct offset_index_sequence;

template <size_t Idx, size_t Offset>
struct offset_index_value
{
    static constexpr size_t value = Idx + Offset;
};

template <size_t Offset, size_t... Idx>
struct offset_index_sequence<std::index_sequence<Idx...>, Offset>
{
    using type = std::integer_sequence<size_t, offset_index_value<Idx, Offset>::value...>;
};

template <size_t N, size_t OffsetN>
using make_offset_index_sequence =
    offset_index_sequence<std::make_index_sequence<N>, OffsetN>;

template <size_t StartN, size_t EndN>
using make_index_sequence_range =
    typename offset_index_sequence<std::make_index_sequence<(EndN - StartN)>,
                                   StartN>::type;

template <typename Tp>
struct c_array
{
    // Construct an array wrapper from a base pointer and array size
    c_array(Tp* _base, size_t _size)
    : m_base{ _base }
    , m_size{ _size }
    {}

    ~c_array()              = default;
    c_array(const c_array&) = default;
    c_array& operator=(const c_array&) = default;
    c_array& operator=(c_array&&) noexcept = default;

    // Get the size of the wrapped array
    size_t size() const { return m_size; }

    // Access an element by index
    Tp& operator[](size_t i) { return m_base[i]; }

    // Access an element by index
    const Tp& operator[](size_t i) const { return m_base[i]; }

    // Access an element by index with bounds check
    Tp& at(size_t i)
    {
        if(i < m_size) return m_base[i];
        throw std::out_of_range(std::string{ typeid(*this).name() } + std::to_string(i) +
                                " exceeds size " + std::to_string(m_size));
    }

    // Access an element by index with bounds check
    const Tp& at(size_t i) const
    {
        if(i < m_size) return m_base[i];
        throw std::out_of_range(std::string{ typeid(*this).name() } + std::to_string(i) +
                                " exceeds size " + std::to_string(m_size));
    }

    // Get a slice of this array, from a start index (inclusive) to end index (exclusive)
    c_array<Tp> slice(size_t start, size_t end)
    {
        return c_array<Tp>(&m_base[start], end - start);
    }

    operator Tp*() const { return m_base; }

    // Iterator class for convenient range-based for loop support
    template <typename Up>
    struct iterator
    {
        // Start the iterator at a given pointer
        iterator(Tp* p)
        : m_ptr{ p }
        {}

        // Advance to the next element
        void operator++() { ++m_ptr; }
        void operator++(int) { m_ptr++; }

        // Get the current element
        Up& operator*() const { return *m_ptr; }

        // Compare iterators
        bool operator==(const iterator& rhs) const { return m_ptr == rhs.m_ptr; }
        bool operator!=(const iterator& rhs) const { return m_ptr != rhs.m_ptr; }

    private:
        Tp* m_ptr = nullptr;
    };

    // Get an iterator positioned at the beginning of the wrapped array
    iterator<Tp>       begin() { return iterator<Tp>{ m_base }; }
    iterator<const Tp> begin() const { return iterator<const Tp>{ m_base }; }

    // Get an iterator positioned at the end of the wrapped array
    iterator<Tp>       end() { return iterator<Tp>{ &m_base[m_size] }; }
    iterator<const Tp> end() const { return iterator<const Tp>{ &m_base[m_size] }; }

private:
    Tp*    m_base = nullptr;
    size_t m_size = 0;
};

// Function for automatic template argument deduction
template <typename Tp>
c_array<Tp>
wrap_c_array(Tp* base, size_t size)
{
    return c_array<Tp>(base, size);
}
}  // namespace utility
}  // namespace omnitrace
