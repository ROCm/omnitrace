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

#include "library/causal/experiment.hpp"
#include "common/defines.h"
#include "library/causal/data.hpp"
#include "library/causal/delay.hpp"
#include "library/causal/progress_point.hpp"
#include "library/code_object.hpp"
#include "library/components/backtrace_causal.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/state.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <timemory/components/timing/backends.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/policy.hpp>
#include <timemory/tpls/cereal/archives.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/tpls/cereal/cereal/archives/json.hpp>
#include <timemory/tpls/cereal/types.hpp>

#include <chrono>
#include <ratio>
#include <regex>
#include <thread>
#include <vector>

namespace omnitrace
{
namespace causal
{
namespace
{
using backtrace_causal = omnitrace::component::backtrace_causal;
namespace cereal       = ::tim::cereal;

auto                        current_experiment_value = experiment{};
std::atomic<uint64_t>       current_selected_count{ 0 };
std::atomic<experiment*>    current_experiment{ nullptr };
std::vector<experiment>     experiment_history = {};
std::vector<unwind_stack_t> ignored_stacks     = {};
int64_t                     global_scaling     = 1;
}  // namespace

std::string
experiment::label()
{
    return "casual_experiment";
}

std::string
experiment::description()
{
    return "Records an experiment for casual profiling";
}

const std::atomic<experiment*>&
experiment::get_current_experiment()
{
    return current_experiment;
}

bool
experiment::start()
{
    if(running && tracing::now() < start_time + experiment_time) return false;

    selection = sample_selection();
    if(!selection) return false;

    // sampling period in nanoseconds
    sampling_period = backtrace_causal::get_period(units::nsec);
    // adjust for the real sampling period
    period_stats = component::backtrace_causal::get_period_stats();
    if(period_stats.get_count() > 10) sampling_period = period_stats.get_mean();

    index           = experiment_history.size() + 1;
    experiment_time = global_scaling * scaling_factor * sampling_period * batch_size;
    virtual_speedup = sample_virtual_speedup();
    delay_scaling   = virtual_speedup / 100.0;
    sample_delay    = sampling_period * delay_scaling;
    total_delay     = delay::sync();
    init_progress   = get_progress_points();
    start_time      = tracing::now();

    OMNITRACE_VERBOSE(0, "Starting causal experiment #%u: %s\n", index,
                      as_string().c_str());

    current_experiment_value = *this;
    current_selected_count.store(0);
    current_experiment.store(this);
    return true;
}

bool
experiment::wait() const
{
    auto _now  = tracing::now();
    auto _wait = experiment_time - (_now - start_time);
    auto _end  = _now + _wait;
    auto _incr = std::min<uint64_t>(_wait / 100, 1000000);
    while(tracing::now() < _end && get_state() < State::Finalized)
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _incr });
    }
    return (tracing::now() >= _end);
}

bool
experiment::stop()
{
    auto _now = tracing::now();
    if(_now < start_time + experiment_time) return false;

    current_experiment.store(nullptr);
    selected        = current_selected_count.load();
    running         = false;
    end_time        = _now;
    experiment_time = (end_time - start_time);
    global_delay    = delay::compute_total_delay(0);
    total_delay     = (global_delay - total_delay);
    duration      = (experiment_time > total_delay) ? (experiment_time - total_delay) : 0;
    fini_progress = get_progress_points();
    period_stats  = component::backtrace_causal::get_period_stats();

    // sync data
    delay::sync();
    // component::backtrace_causal::reset_period_stats();

    int64_t _num = 0;
    for(auto fitr : fini_progress)
        _num = std::max<int64_t>(_num, fitr.second.get_laps() -
                                           init_progress[fitr.first].get_laps());

    if(_num < 5)
        global_scaling *= 2;
    else if(_num > 10)
        global_scaling /= 2;

    if(_num > 0) experiment_history.emplace_back(*this);

    std::this_thread::sleep_for(std::chrono::nanoseconds{ sampling_period });
    return true;
}

std::string
experiment::as_string() const
{
    std::stringstream _ss{};
    auto _dur = static_cast<double>(experiment_time) / static_cast<double>(units::sec);
    _ss << std::boolalpha << "virtual speed-up: " << std::setw(3) << virtual_speedup
        << "%, delay: " << std::setw(6) << sample_delay << ", duration: " << std::setw(6)
        << std::fixed << std::setprecision(3) << _dur << " sec";
    return _ss.str();
}

// in nanoseconds
uint64_t
experiment::get_delay()
{
    if(!current_experiment.load()) return 0;
    return current_experiment_value.sample_delay;
}

double
experiment::get_delay_scaling()
{
    if(!current_experiment.load()) return 0;
    return current_experiment_value.delay_scaling;
}

uint32_t
experiment::get_index()
{
    if(!is_active()) return 0;
    return current_experiment_value.index;
}

bool
experiment::is_active()
{
    return (current_experiment.load(std::memory_order_relaxed) != nullptr);
}

bool
experiment::is_selected(unwind_stack_t _stack)
{
    if(is_active())
    {
        for(auto itr : _stack)
            if(current_experiment_value.selection.contains(itr->address())) return true;
    }
    return false;
}

bool
experiment::is_selected(utility::c_array<void*> _stack)
{
    if(is_active())
    {
        for(auto* itr : _stack)
            if(itr && current_experiment_value.selection.contains(
                          reinterpret_cast<uintptr_t>(itr)))
                return true;
    }
    return false;
}

void
experiment::add_selected()
{
    if(current_experiment.load() == nullptr) return;
    ++current_selected_count;
}

std::vector<experiment>
experiment::get_experiments()
{
    return experiment_history;
}

void
experiment::save_experiments()
{
    auto _cfg         = settings::compose_filename_config{};
    _cfg.subdirectory = "causal";
    save_experiments("experiments", _cfg);
}

void
experiment::load_experiments(bool _throw_on_error)
{
    auto _cfg         = settings::compose_filename_config{};
    _cfg.subdirectory = "causal";
    load_experiments("experiments", _cfg, _throw_on_error);
}

void
experiment::save_experiments(std::string _fname, const filename_config_t& _cfg)
{
    using cache_type = typename unwind_stack_t::cache_type;

    auto        _fname_orig = _fname;
    const auto& _info0      = thread_info::get(0, InternalTID);

    // if(experiment_history.size() > 1)
    //    experiment_history.erase(experiment_history.begin());

    auto _merge_samples = [](auto& _dst, const auto& _src) {
        for(const auto& sitr : _src)
        {
            if(!_dst.emplace(sitr).second)
            {
                auto titr = _dst.find(sitr);
                titr->count += sitr.count;
            }
        }
    };

    // update sample data
    {
        auto _samples = component::backtrace_causal::get_samples();
        for(auto& itr : experiment_history)
            _merge_samples(itr.samples, _samples[itr.index]);
    }

    auto _total_samples = sample_dataset_t{};
    for(const auto& itr : experiment_history)
        _merge_samples(_total_samples, itr.samples);

    save_line_info(_cfg);

    if(experiment_history.empty()) return;

    auto _saved_experiment_history = std::vector<experiment>{};
    auto _saved_samples_history    = std::map<uint32_t, sample_dataset_t>{};
    _fname = tim::settings::compose_output_filename(_fname_orig, "json", _cfg);
    {
        std::ifstream ifs{ _fname };
        if(ifs)
        {
            auto ar = tim::policy::input_archive<cereal::JSONInputArchive>::get(ifs);

            ar->setNextName("omnitrace");
            ar->startNode();
            ar->setNextName("causal");
            ar->startNode();
            (*ar)(cereal::make_nvp("experiments", _saved_experiment_history));
            ar->finishNode();
            ar->finishNode();
        }
    }

    for(auto itr : experiment_history)
        _saved_experiment_history.emplace_back(itr);

    std::stringstream oss{};
    {
        auto ar = tim::policy::output_archive<cereal::PrettyJSONOutputArchive>::get(oss);

        ar->setNextName("omnitrace");
        ar->startNode();
        ar->setNextName("causal");
        ar->startNode();
        (*ar)(cereal::make_nvp("experiments", _saved_experiment_history));
        ar->finishNode();
        ar->finishNode();
    }
    {
        std::ofstream ofs{};
        if(tim::filepath::open(ofs, _fname))
        {
            if(get_verbose() >= 0)
                operation::file_output_message<experiment>{}(
                    _fname, std::string{ "causal_experiments" });
            ofs << oss.str() << "\n";
        }
        else
        {
            OMNITRACE_THROW("Error opening causal experiments output file: %s",
                            _fname.c_str());
        }
    }

    _fname = tim::settings::compose_output_filename(_fname_orig, "coz", _cfg);
    std::stringstream _existing{};
    {
        std::ifstream ifs{ _fname };
        if(ifs)
        {
            while(ifs && ifs.good())
            {
                std::string _line;
                std::getline(ifs, _line);
                _existing << _line << "\n";
            }
        }
    }
    std::ofstream ofs{};
    ofs.setf(std::ios::fixed);
    if(tim::filepath::open(ofs, _fname))
    {
        if(get_verbose() >= 0)
            operation::file_output_message<experiment>{}(
                _fname, std::string{ "causal_experiments" });
        auto _cache = cache_type{ true };
        ofs << _existing.str();
        ofs << "startup\ttime=" << _info0->lifetime.first << "\n";

        uint64_t _beg_runtime = std::numeric_limits<uint64_t>::max();
        uint64_t _end_runtime = std::numeric_limits<uint64_t>::min();
        for(auto& itr : experiment_history)
        {
            if(itr.duration == 0) continue;
            if(itr.experiment_time == 0) continue;
            _beg_runtime = std::min<uint64_t>(_beg_runtime, itr.start_time);
            _end_runtime = std::max<uint64_t>(_end_runtime, itr.end_time);
        }
        uint64_t _runtime = (_end_runtime - _beg_runtime);

        uint64_t _duration_sum = 0;
        for(auto& itr : experiment_history)
        {
            if(itr.duration == 0) continue;
            OMNITRACE_VERBOSE(0, "\n");
            auto& _selection = itr.selection;
            auto& _line_info = _selection.info;

            _duration_sum += itr.duration;

            std::string _name =
                (_selection.symbol_address > 0) ? _line_info.func : _line_info.name();

            OMNITRACE_CONDITIONAL_THROW(
                _name.empty(),
                "Error! causal experiment selection has no name: address=%s, file=%s, "
                "line=%u, func=%s",
                JOIN("", "0x", std::hex, _line_info.address).c_str(),
                _line_info.file.c_str(), _line_info.line, _line_info.func.c_str());

            ofs << "experiment\tselected=" << demangle(_name)
                << "\tspeedup=" << std::setprecision(2)
                << static_cast<double>(itr.virtual_speedup / 100.0)
                << "\tduration=" << itr.duration << "\tselected-samples=" << itr.selected
                << "\tduration_sum=" << _duration_sum
                << "\texperiment_time=" << itr.experiment_time
                << "\tglobal_delay=" << itr.global_delay
                << "\ttotal_delay=" << itr.total_delay << "\n";

            auto ppts = itr.fini_progress;
            for(auto pitr : itr.init_progress)
                ppts[pitr.first] -= pitr.second;

            for(auto pitr : ppts)
            {
                if(pitr.second.get_laps() == 0) continue;
                if(get_causal_end_to_end() && pitr.second.get_laps() > 1) continue;
                ofs << "throughput-point\tname="
                    << tim::demangle(tim::get_hash_identifier(pitr.first))
                    << "\tdelta=" << pitr.second.get_laps() << "\n";
                // ofs << "latency-point\tname=" << tim::get_hash_identifier(pitr.first)
                //    << "\tarrivals=" << pitr.second.get_laps()
                //    << "\tdepartures=" << pitr.second.get_laps()
                //    << "\tdifference=" << pitr.second.get_accum() << "\n";
                if(get_causal_end_to_end()) break;
            }
        }

        auto _prediction =
            100.0 * ((_runtime - _duration_sum) / static_cast<double>(_runtime));
        ofs << "runtime\ttime=" << _runtime
            << "\tremainder=" << (_runtime - _duration_sum)
            << "\tprediction=" << _prediction << "\n";

        // sort the samples alphabetically
        auto _aggregated_total_samples = std::map<std::string, uint64_t>{};
        for(const auto& itr : _total_samples)
        {
            for(const auto& iitr : get_line_info(itr.address, false))
                _aggregated_total_samples[iitr.name()] += itr.count;
        }

        for(const auto& itr : _aggregated_total_samples)
        {
            if(itr.second > 0)
                ofs << "samples\tlocation=" << itr.first << "\tcount=" << itr.second
                    << "\n";
        }
    }
    else
    {
        OMNITRACE_THROW("Error opening causal experiments output file: %s",
                        _fname.c_str());
    }
}

void
experiment::load_experiments(std::string _fname, const filename_config_t& _cfg,
                             bool _throw_on_error)
{
    _fname = tim::settings::compose_input_filename(_fname, "json", _cfg);

    std::ifstream           ifs{};
    std::vector<experiment> _experiments{};
    if(tim::filepath::open(ifs, _fname))
    {
        auto ar = tim::policy::input_archive<cereal::JSONInputArchive>::get(ifs);

        ar->setNextName("omnitrace");
        ar->startNode();
        ar->setNextName("causal");
        ar->startNode();
        (*ar)(cereal::make_nvp("experiments", _experiments));
        ar->finishNode();
        ar->finishNode();
    }
    else
    {
        if(_throw_on_error)
        {
            OMNITRACE_THROW("Error opening causal experiments input file: %s",
                            _fname.c_str());
        }
        return;
    }

    experiment_history.reserve(experiment_history.size() + _experiments.size());
    for(auto& itr : _experiments)
        experiment_history.emplace_back(std::move(itr));
}
}  // namespace causal
}  // namespace omnitrace
