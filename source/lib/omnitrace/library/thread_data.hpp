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

#include "api.hpp"
#include "library/common.hpp"
#include "library/concepts.hpp"
#include "library/config.hpp"
#include "library/defines.hpp"
#include "library/state.hpp"
#include "library/timemory.hpp"

#include <timemory/utility/types.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

namespace omnitrace
{
ThreadState set_thread_state(ThreadState);

// bundle of components used in instrumentation
using instrumentation_bundle_t =
    tim::component_bundle<project::omnitrace, comp::wall_clock*,
                          comp::user_global_bundle*>;

// allocator for instrumentation_bundle_t
using bundle_allocator_t = tim::data::ring_buffer_allocator<instrumentation_bundle_t>;

template <typename Tp>
struct thread_deleter;

// unique ptr type for omnitrace
template <typename Tp>
using unique_ptr_t = std::unique_ptr<Tp, thread_deleter<Tp>>;

static constexpr size_t max_supported_threads = OMNITRACE_MAX_THREADS;

template <>
struct thread_deleter<void>
{
    void operator()() const;
};

extern template struct thread_deleter<void>;

template <typename Tp>
struct thread_deleter
{
    void operator()(Tp* ptr) const
    {
        thread_deleter<void>{}();
        delete ptr;
    }
};

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
            return type{ new value_type{ invoke(std::forward<Args>(_args), 0)... } };
        }
        else
        {
            return type{ invoke(std::forward<Args>(_args), 0)... };
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
};

using construct_on_init = std::true_type;

struct construct_on_thread
{
    int64_t index = threading::get_id();
};

template <typename Tp, typename Tag = void, size_t MaxThreads = max_supported_threads>
struct thread_data
{
    using value_type        = unique_ptr_t<Tp>;
    using instance_array_t  = std::array<value_type, MaxThreads>;
    using construct_on_init = std::true_type;

    template <typename... Args>
    static void              construct(construct_on_thread&&, Args&&...);
    static value_type&       instance();
    static instance_array_t& instances();
    template <typename... Args>
    static value_type& instance(construct_on_thread&&, Args&&...);
    template <typename... Args>
    static instance_array_t& instances(construct_on_init, Args&&...);

    template <typename... Args>
    static void construct(Args&&... args)
    {
        construct(construct_on_thread{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static value_type& instance(Args&&... args)
    {
        return instance(construct_on_thread{}, std::forward<Args>(args)...);
    }

    static constexpr size_t size() { return MaxThreads; }

    decltype(auto) begin() { return instances().begin(); }
    decltype(auto) end() { return instances().end(); }

    decltype(auto) begin() const { return instances().begin(); }
    decltype(auto) end() const { return instances().end(); }
};

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
void
thread_data<Tp, Tag, MaxThreads>::construct(construct_on_thread&& _t, Args&&... _args)
{
    // construct outside of lambda to prevent data-race
    static auto& _instances = instances();
    if(!_instances.at(_t.index))
        _instances.at(_t.index) = generate<value_type>{}(std::forward<Args>(_args)...);
}

template <typename Tp, typename Tag, size_t MaxThreads>
unique_ptr_t<Tp>&
thread_data<Tp, Tag, MaxThreads>::instance()
{
    return instances().at(threading::get_id());
}

template <typename Tp, typename Tag, size_t MaxThreads>
typename thread_data<Tp, Tag, MaxThreads>::instance_array_t&
thread_data<Tp, Tag, MaxThreads>::instances()
{
    static auto _v = instance_array_t{};
    return _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
unique_ptr_t<Tp>&
thread_data<Tp, Tag, MaxThreads>::instance(construct_on_thread&& _t, Args&&... _args)
{
    construct(construct_on_thread{ _t }, std::forward<Args>(_args)...);
    return instances().at(_t.index);
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
typename thread_data<Tp, Tag, MaxThreads>::instance_array_t&
thread_data<Tp, Tag, MaxThreads>::instances(construct_on_init, Args&&... _args)
{
    static auto& _v = [&]() -> instance_array_t& {
        auto& _internal = instances();
        for(size_t i = 0; i < MaxThreads; ++i)
            _internal.at(i) = generate<value_type>{}(std::forward<Args>(_args)...);
        return _internal;
    }();
    return _v;
}

//--------------------------------------------------------------------------------------//
//
//          thread_data with std::optional
//
//--------------------------------------------------------------------------------------//

template <typename Tp, typename Tag, size_t MaxThreads>
struct thread_data<std::optional<Tp>, Tag, MaxThreads>
{
    using value_type       = std::optional<Tp>;
    using instance_array_t = std::array<value_type, MaxThreads>;

    template <typename... Args>
    static void              construct(construct_on_thread&&, Args&&...);
    static value_type&       instance();
    static instance_array_t& instances();
    template <typename... Args>
    static value_type& instance(construct_on_thread&&, Args&&...);
    template <typename... Args>
    static instance_array_t& instances(construct_on_init, Args&&...);

    static constexpr size_t size() { return MaxThreads; }

    decltype(auto) begin() { return instances().begin(); }
    decltype(auto) end() { return instances().end(); }

    decltype(auto) begin() const { return instances().begin(); }
    decltype(auto) end() const { return instances().end(); }
};

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
void
thread_data<std::optional<Tp>, Tag, MaxThreads>::construct(construct_on_thread&& _t,
                                                           Args&&... _args)
{
    // construct outside of lambda to prevent data-race
    static auto& _instances = instances();
    if(!_instances.at(_t.index))
        _instances.at(_t.index) = generate<value_type>{}(std::forward<Args>(_args)...);
}

template <typename Tp, typename Tag, size_t MaxThreads>
std::optional<Tp>&
thread_data<std::optional<Tp>, Tag, MaxThreads>::instance()
{
    return instances().at(threading::get_id());
}

template <typename Tp, typename Tag, size_t MaxThreads>
typename thread_data<std::optional<Tp>, Tag, MaxThreads>::instance_array_t&
thread_data<std::optional<Tp>, Tag, MaxThreads>::instances()
{
    static auto _v = instance_array_t{};
    return _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
std::optional<Tp>&
thread_data<std::optional<Tp>, Tag, MaxThreads>::instance(construct_on_thread&& _t,
                                                          Args&&... _args)
{
    construct(construct_on_thread{ _t }, std::forward<Args>(_args)...);
    return instances().at(_t.index);
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
typename thread_data<std::optional<Tp>, Tag, MaxThreads>::instance_array_t&
thread_data<std::optional<Tp>, Tag, MaxThreads>::instances(construct_on_init,
                                                           Args&&... _args)
{
    static auto& _v = [&]() -> instance_array_t& {
        auto& _internal = instances();
        for(size_t i = 0; i < MaxThreads; ++i)
            _internal.at(i) = generate<value_type>{}(std::forward<Args>(_args)...);
        return _internal;
    }();
    return _v;
}

//--------------------------------------------------------------------------------------//
//
//          thread_data with raw data (no pointer)
//
//--------------------------------------------------------------------------------------//

using tim::identity;
using tim::identity_t;

template <typename Tp, typename Tag, size_t MaxThreads>
struct thread_data<identity<Tp>, Tag, MaxThreads>
{
    using value_type       = Tp;
    using instance_array_t = std::array<value_type, MaxThreads>;

    template <typename... Args>
    static void              construct(construct_on_thread&&, Args&&...);
    static value_type&       instance();
    static instance_array_t& instances();
    template <typename... Args>
    static value_type& instance(construct_on_thread&&, Args&&...);
    template <typename... Args>
    static instance_array_t& instances(construct_on_init, Args&&...);

    template <typename... Args>
    static void construct(Args&&... args)
    {
        construct(construct_on_thread{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static value_type& instance(Args&&... args)
    {
        return instance(construct_on_thread{}, std::forward<Args>(args)...);
    }

    static constexpr size_t size() { return MaxThreads; }

    decltype(auto) begin() { return instances().begin(); }
    decltype(auto) end() { return instances().end(); }

    decltype(auto) begin() const { return instances().begin(); }
    decltype(auto) end() const { return instances().end(); }
};

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
void
thread_data<identity<Tp>, Tag, MaxThreads>::construct(construct_on_thread&& _t,
                                                      Args&&... _args)
{
    // construct outside of lambda to prevent data-race
    static auto& _instances   = instances();
    static auto  _constructed = std::array<bool, MaxThreads>{ false };
    if(!_constructed.at(_t.index))
        _constructed.at(_t.index) = (_instances.at(_t.index) = generate<value_type>{}(
                                         std::forward<Args>(_args)...),
                                     true);
}

template <typename Tp, typename Tag, size_t MaxThreads>
Tp&
thread_data<identity<Tp>, Tag, MaxThreads>::instance()
{
    return instances().at(threading::get_id());
}

template <typename Tp, typename Tag, size_t MaxThreads>
typename thread_data<identity<Tp>, Tag, MaxThreads>::instance_array_t&
thread_data<identity<Tp>, Tag, MaxThreads>::instances()
{
    static auto _v = instance_array_t{};
    return _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
Tp&
thread_data<identity<Tp>, Tag, MaxThreads>::instance(construct_on_thread&& _t,
                                                     Args&&... _args)
{
    construct(construct_on_thread{ _t }, std::forward<Args>(_args)...);
    return instances().at(_t.index);
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
typename thread_data<identity<Tp>, Tag, MaxThreads>::instance_array_t&
thread_data<identity<Tp>, Tag, MaxThreads>::instances(construct_on_init, Args&&... _args)
{
    static auto& _v = [&]() -> instance_array_t& {
        auto& _internal = instances();
        for(size_t i = 0; i < MaxThreads; ++i)
            _internal.at(i) = generate<value_type>{}(std::forward<Args>(_args)...);
        return _internal;
    }();
    return _v;
}

//--------------------------------------------------------------------------------------//

// there are currently some strange things that happen with
// vector<instrumentation_bundle_t> so using vector<instrumentation_bundle_t*> and
// timemory's ring_buffer_allocator to create contiguous memory-page aligned instances of
// the bundle
template <typename... Tp>
struct component_bundle_cache
{
    using bundle_type    = tim::component_bundle<project::omnitrace, Tp...>;
    using this_type      = component_bundle_cache<Tp...>;
    using allocator_type = tim::data::ring_buffer_allocator<bundle_type>;
    using instance_type =
        std::array<component_bundle_cache<Tp...>, max_supported_threads>;

    using iterator         = typename std::vector<bundle_type*>::iterator;
    using const_iterator   = typename std::vector<bundle_type*>::const_iterator;
    using reverse_iterator = typename std::vector<bundle_type*>::reverse_iterator;

    allocator_type            allocator = {};
    std::vector<bundle_type*> bundles   = {};

    bool empty() const { return bundles.empty(); }

    auto& front() { return bundles.front(); }
    auto& front() const { return bundles.front(); }

    auto& back() { return bundles.back(); }
    auto& back() const { return bundles.back(); }

    auto begin() { return bundles.begin(); }
    auto end() { return bundles.end(); }

    auto rbegin() { return bundles.rbegin(); }
    auto rend() { return bundles.rend(); }

    auto begin() const { return bundles.begin(); }
    auto end() const { return bundles.end(); }

    auto size() const { return bundles.size(); }

    auto&       at(size_t _idx) { return bundles.at(_idx); }
    const auto& at(size_t _idx) const { return bundles.at(_idx); }

    static auto& instances()
    {
        static auto _v = instance_type{};
        return _v;
    }

    static auto& instance(int64_t _tid) { return instances().at(_tid); }

    template <typename... Args>
    bundle_type* construct(Args&&... args)
    {
        bundle_type* _v = allocator.allocate(1);
        allocator.construct(_v, std::forward<Args>(args)...);
        return bundles.emplace_back(_v);
    }

    void destroy(bundle_type* _v, size_t _idx)
    {
        allocator.destroy(_v);
        allocator.deallocate(_v, 1);
        bundles.erase(bundles.begin() + _idx);
    }

    void pop_back()
    {
        bundle_type* _v = bundles.back();
        allocator.destroy(_v);
        allocator.deallocate(_v, 1);
        bundles.pop_back();
    }

    template <typename IterT>
    void destroy(IterT _v)
    {
        iterator itr = begin();
        if constexpr(std::is_same<IterT, reverse_iterator>::value)
        {
            if(_v == rend()) return;
            std::advance(itr, std::distance(rbegin(), _v));
        }
        else
        {
            if(_v == end()) return;
            itr = _v;
        }
        allocator.destroy(*itr);
        allocator.deallocate(*itr, 1);
        bundles.erase(itr);
    }
};

template <typename... Tp>
struct component_bundle_cache<tim::component_bundle<project::omnitrace, Tp...>>
: component_bundle_cache<Tp...>
{
    using base_type = component_bundle_cache<Tp...>;

    using base_type::allocator;
    using base_type::bundles;
    using base_type::instances;
};

using instrumentation_bundles = component_bundle_cache<instrumentation_bundle_t>;
extern template struct component_bundle_cache<instrumentation_bundle_t>;
}  // namespace omnitrace
