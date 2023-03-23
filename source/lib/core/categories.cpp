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

#include "core/categories.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/constraint.hpp"
#include "core/debug.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"

#include <set>
#include <string>
#include <type_traits>

namespace omnitrace
{
namespace categories
{
namespace
{
template <typename Tp, typename ScopeT>
void
configure_categories(bool _enable, ScopeT&&, const std::set<std::string>& _categories)
{
    auto _name = trait::name<Tp>::value;
    if(_categories.count(_name) > 0)
    {
        OMNITRACE_VERBOSE_F(3, "%s category: %s\n", (_enable) ? "Enabling" : "Disabling",
                            _name);
        if constexpr(std::is_same<ScopeT, scope::process_scope>::value)
            trait::runtime_enabled<Tp>::set(_enable);
        else
        {
            static_assert(
                std::is_same<ScopeT, scope::thread_scope>::value,
                "Error! unsupported scope type :: must be process_scope or thread_scope");
            trait::runtime_enabled<Tp>::set(ScopeT{}, _enable);
        }
    }
}

template <typename ScopeT, size_t... Idx>
void
configure_categories(bool _enable, ScopeT&&, const std::set<std::string>& _categories,
                     std::index_sequence<Idx...>)
{
    (configure_categories<category_type_id_t<Idx>>(_enable, ScopeT{}, _categories), ...);
}

void
configure_categories(bool _enable, const std::set<std::string>& _categories)
{
    OMNITRACE_VERBOSE_F(1, "%s categories...\n", (_enable) ? "Enabling" : "Disabling");

    configure_categories(
        _enable, scope::process_scope{}, _categories,
        utility::make_index_sequence_range<1, OMNITRACE_CATEGORY_LAST>{});
}
}  // namespace

void
enable_categories(const std::set<std::string>& _categories)
{
    configure_categories(
        true, scope::process_scope{}, _categories,
        utility::make_index_sequence_range<1, OMNITRACE_CATEGORY_LAST>{});
}

void
disable_categories(const std::set<std::string>& _categories)
{
    configure_categories(
        false, scope::process_scope{}, _categories,
        utility::make_index_sequence_range<1, OMNITRACE_CATEGORY_LAST>{});
}

void
enable_categories(scope::thread_scope, const std::set<std::string>& _categories)
{
    configure_categories(
        true, scope::thread_scope{}, _categories,
        utility::make_index_sequence_range<1, OMNITRACE_CATEGORY_LAST>{});
}

void
disable_categories(scope::thread_scope, const std::set<std::string>& _categories)
{
    configure_categories(
        false, scope::thread_scope{}, _categories,
        utility::make_index_sequence_range<1, OMNITRACE_CATEGORY_LAST>{});
}

void
setup()
{
    // disable specified categories
    disable_categories();

    auto _trace_specs = constraint::get_trace_specs();

    if(!_trace_specs.empty())
    {
        auto _trace_stages = constraint::get_trace_stages();

        _trace_stages.init = [](const constraint::spec& _spec) {
            if(_spec.delay > 1.0e-3) disable_categories(config::get_enabled_categories());
            return get_state() < State::Finalized;
        };

        _trace_stages.start = [](const constraint::spec&) {
            enable_categories(config::get_enabled_categories());
            return get_state() < State::Finalized;
        };

        _trace_stages.stop = [](const constraint::spec&) {
            // only disable categories if not finalized since this might run in background
            // during finalization and disable output of data in those categories
            if(get_state() < State::Finalized)
                disable_categories(config::get_enabled_categories());
            return get_state() < State::Finalized;
        };

        auto _promise = std::promise<void>();
        std::thread{ [_trace_specs, _trace_stages](std::promise<void>* _prom) {
                        // ensure all categories are disabled before proceeding
                        // if a delay is requested
                        if(_trace_specs.front().delay > 1.0e-3)
                            disable_categories(config::get_enabled_categories());
                        _prom->set_value();
                        for(const auto& itr : _trace_specs)
                            itr(_trace_stages);
                    },
                     &_promise }
            .detach();

        _promise.get_future().wait_for(std::chrono::seconds{ 1 });
    }
}

void
shutdown()
{
    disable_categories(config::get_enabled_categories());
}
}  // namespace categories
}  // namespace omnitrace
