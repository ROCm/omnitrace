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
#include "library/common.hpp"
#include "library/components/fwd.hpp"
#include "library/defines.hpp"
#include "library/timemory.hpp"
#include "timemory/utility/unwind.hpp"

#include <timemory/components/base.hpp>
#include <timemory/macros/language.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/units.hpp>

#include <chrono>
#include <cstdint>

namespace omnitrace
{
namespace component
{
struct backtrace_causal
: tim::component::empty_base
, tim::concepts::component
{
    using value_type = void;
    struct sample_data : unwind::processed_entry
    {
        using base_type = unwind::processed_entry;

        using base_type::operator==;
        using base_type::operator!=;
        using base_type::operator<;
        using base_type::operator>;
        using base_type::operator<=;
        using base_type::operator>=;

        mutable uint64_t count = 0;

        template <typename ArchiveT>
        void serialize(ArchiveT& ar, const unsigned version)
        {
            base_type::serialize(ar, version);
            ar(tim::cereal::make_nvp("count", count));
        }
    };

    static std::string label() { return "backtrace_causal"; }
    static std::string description()
    {
        return "Causal profiling data collected in backtrace";
    }

    backtrace_causal()                            = default;
    ~backtrace_causal()                           = default;
    backtrace_causal(const backtrace_causal&)     = default;
    backtrace_causal(backtrace_causal&&) noexcept = default;

    backtrace_causal& operator=(const backtrace_causal&) = default;
    backtrace_causal& operator=(backtrace_causal&&) noexcept = default;

    static void start();
    static void stop() {}

    void sample(int = -1);

    auto is_valid() const { return m_valid; }
    auto get_internal() const { return m_internal; }
    auto get_selected() const { return m_selected; }
    auto get_index() const { return m_index; }
    auto get_stack() const { return m_stack; }

    template <typename Tp = uint64_t>
    static Tp get_period(uint64_t _units = units::nsec);

    using sample_data_set_t = std::set<sample_data>;

    static std::map<uint32_t, sample_data_set_t> get_samples();
    static std::set<sample_data>                 get_samples(uint32_t);

    static void add_sample(uint32_t, const sample_data::base_type&);
    static void add_samples(uint32_t, const std::vector<sample_data::base_type>&);

private:
    bool                   m_valid    = false;
    bool                   m_internal = false;
    bool                   m_selected = false;
    uint32_t               m_index    = 0;
    causal::unwind_stack_t m_stack    = {};
};
}  // namespace component
}  // namespace omnitrace
