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
#include "library/components/backtrace_causal.hpp"
#include "library/defines.hpp"
#include "library/utility.hpp"

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
    using filename_config_t = settings::compose_filename_config;
    using sample_data_t     = component::backtrace_causal::sample_data;
    using sample_dataset_t  = std::set<sample_data_t>;
    using period_stats_t    = tim::statistics<int64_t>;

    static std::string                     label();
    static std::string                     description();
    static const std::atomic<experiment*>& get_current_experiment();

    TIMEMORY_DEFAULT_OBJECT(experiment)

    bool        start();
    bool        wait() const;  // returns false if interrupted
    bool        stop();
    std::string as_string() const;

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned version);

    // in nanoseconds
    static uint64_t      get_delay();
    static double        get_delay_scaling();
    static uint32_t      get_index();
    static bool          is_active();
    static bool          is_selected(unwind_stack_t);
    static bool          is_selected(utility::c_array<void*>);
    static void          add_selected();
    static experiments_t get_experiments();

    static void save_experiments();
    static void load_experiments(bool _throw_on_err = true);
    static void save_experiments(std::string, const filename_config_t&);
    static void load_experiments(std::string, const filename_config_t&, bool = true);

    bool              running         = false;
    uint16_t          virtual_speedup = 0;   /// 0-100 in multiples of 5
    uint32_t          index           = 0;   /// experiment number
    uint64_t          sampling_period = 0;   /// period b/t samples [nsec]
    uint64_t          start_time      = 0;   /// start of experiment [nsec]
    uint64_t          end_time        = 0;   /// end of experiment [nsec]
    uint64_t          experiment_time = 0;   /// how long the experiment ran [nsec]
    uint64_t          duration        = 0;   /// runtime - delays [nsec]
    uint64_t          batch_size      = 10;  /// batch factor for experiment/cooloff
    uint64_t          scaling_factor  = 50;  /// scaling factor for experiment time
    uint64_t          sample_delay    = 0;   /// how long to delay [nsec]
    uint64_t          total_delay     = 0;   /// total delays [nsec]
    uint64_t          selected        = 0;   /// num times selected line sampled
    uint64_t          global_delay    = 0;
    double            delay_scaling   = 0.0;  /// virtual_speedup / 100.
    selected_entry    selection       = {};   /// which line was selected
    progress_points_t init_progress   = {};   /// progress points at start
    progress_points_t fini_progress   = {};   /// progress points at end
    sample_dataset_t  samples         = {};   /// data sampled during experiment
    period_stats_t    period_stats    = {};   /// stats for sampling period
};

template <typename ArchiveT>
void
experiment::serialize(ArchiveT& ar, const unsigned)
{
    namespace cereal = ::tim::cereal;

    ar(cereal::make_nvp("index", index),
       cereal::make_nvp("virtual_speedup", virtual_speedup),
       cereal::make_nvp("sampling_period", sampling_period),
       cereal::make_nvp("start_time", start_time), cereal::make_nvp("end_time", end_time),
       cereal::make_nvp("experiment_time", experiment_time),
       cereal::make_nvp("batch_size", batch_size), cereal::make_nvp("duration", duration),
       cereal::make_nvp("scaling_factor", scaling_factor),
       cereal::make_nvp("selected", selected),
       cereal::make_nvp("sample_delay", sample_delay),
       cereal::make_nvp("delay_scaling", delay_scaling),
       cereal::make_nvp("total_delay", total_delay),
       cereal::make_nvp("global_delay", global_delay),
       cereal::make_nvp("selection", selection),
       cereal::make_nvp("init_progress", init_progress),
       cereal::make_nvp("fini_progress", fini_progress),
       cereal::make_nvp("period_stats", period_stats),
       cereal::make_nvp("samples", samples));

    if constexpr(concepts::is_input_archive<ArchiveT>::value)
    {
        // selection.name = tim::get_hash_identifier(tim::add_hash_id(_selection));
    }
}
}  // namespace causal
}  // namespace omnitrace
