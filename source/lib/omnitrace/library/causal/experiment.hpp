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

#include <libunwind-x86_64.h>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/utility/unwind.hpp>

#include <atomic>
#include <cstdint>
#include <unordered_map>

namespace omnitrace
{
namespace causal
{
using hash_value_t = ::tim::hash_value_t;

struct experiment
{
    using progress_points_t = std::unordered_map<tim::hash_value_t, progress_point>;
    using experiments_t     = std::vector<experiment>;

    static std::string                     label();
    static std::string                     description();
    static const std::atomic<experiment*>& get_current_experiment();

    TIMEMORY_DEFAULT_OBJECT(experiment)

    bool        start();
    void        wait() const;
    bool        stop();
    std::string as_string() const;

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned version);

    // in nanoseconds
    static uint64_t      get_delay();
    static bool          is_selected(unwind_stack_t);
    static void          add_selected();
    static experiments_t get_experiments();

    static void save_experiments();
    static void load_experiments();
    static void save_experiments(std::string, settings::compose_filename_config);
    static void load_experiments(std::string, settings::compose_filename_config);

    bool              active          = false;
    bool              running         = false;
    uint16_t          virtual_speedup = 0;  // 0-100 in multiples of 5
    uint64_t          start_time      = 0;
    uint64_t          duration        = 0;
    uint64_t          end_time        = 0;
    uint64_t          sample_delay    = 0;
    uint64_t          total_delay     = 0;
    uint64_t          selected        = 0;
    unwind_stack_t    selection       = {};
    progress_points_t start_progress  = {};
    progress_points_t end_progress    = {};
};

template <typename ArchiveT>
void
experiment::serialize(ArchiveT& ar, const unsigned)
{
    namespace cereal = ::tim::cereal;

    auto _selection = std::vector<std::string>{};
    if constexpr(concepts::is_output_archive<ArchiveT>::value)
    {
        _selection.reserve(selection.size());
        for(auto itr : selection)
        {
            if(itr) _selection.emplace_back(demangle(itr->get_name(selection.context)));
        }
    }
    ar(cereal::make_nvp("active", active),
       cereal::make_nvp("virtual_speedup", virtual_speedup),
       cereal::make_nvp("start_time", start_time), cereal::make_nvp("duration", duration),
       cereal::make_nvp("end_time", end_time),
       cereal::make_nvp("sample_delay", sample_delay),
       cereal::make_nvp("total_delay", total_delay),
       cereal::make_nvp("selection", _selection),
       cereal::make_nvp("start_progress", start_progress),
       cereal::make_nvp("end_progress", end_progress));

    if constexpr(concepts::is_input_archive<ArchiveT>::value)
    {
        // selection.name = tim::get_hash_identifier(tim::add_hash_id(_selection));
    }
}
}  // namespace causal
}  // namespace omnitrace
