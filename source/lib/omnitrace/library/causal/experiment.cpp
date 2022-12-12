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
#include "library/state.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <timemory/components/timing/backends.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/policy.hpp>
#include <timemory/tpls/cereal/archives.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/tpls/cereal/types.hpp>

#include <chrono>
#include <ratio>
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
    static uint32_t _experiment_count = 0;

    auto _now = tracing::now();
    if(running && _now < start_time + duration) return false;

    // sampling period in nanoseconds
    auto _sampling_period = backtrace_causal::get_period(units::nsec);
    if(_sampling_period == 0) return false;

    active          = sample_enabled();
    running         = true;
    virtual_speedup = (active) ? sample_virtual_speedup() : 0;
    start_time      = _now;
    duration        = std::max<uint64_t>(1000 * _sampling_period, 100 * units::msec);
    duration        = std::min<uint64_t>(duration, 1 * units::sec);
    sample_delay    = _sampling_period * (virtual_speedup / 100.0);
    total_delay     = delay::sync();
    start_progress  = get_progress_points();
    selection       = sample_selection();

    if(selection.empty()) return false;

    index = ++_experiment_count;

    OMNITRACE_VERBOSE(1, "Starting causal experiment: %s\n", as_string().c_str());

    current_experiment_value = *this;
    current_selected_count.store(0);
    current_experiment.store(this);
    return true;
}

bool
experiment::wait() const
{
    auto _now  = tracing::now();
    auto _wait = duration - (_now - start_time);
    auto _end  = _now + _wait;
    auto _incr = std::min<uint64_t>(_wait / 100, 1000);
    while(tracing::now() < _end && get_state() < State::Finalized)
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _incr });
    }
    return (tracing::now() >= _end);
}

bool
experiment::stop(bool _store)
{
    auto _now = tracing::now();
    if(_now < start_time + duration) return false;

    current_experiment.store(nullptr);
    running      = false;
    end_time     = _now;
    selected     = current_selected_count.load();
    total_delay  = delay::compute_total_delay(total_delay);
    end_progress = get_progress_points();
    delay::sync();

    if(_store) experiment_history.emplace_back(*this);
    return _store;
}

std::string
experiment::as_string() const
{
    std::stringstream _ss{};
    auto _dur = static_cast<double>(duration) / static_cast<double>(units::sec);
    _ss << std::boolalpha << "active: " << std::setw(5) << active
        << ", virtual speed-up: " << std::setw(3) << virtual_speedup
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
        auto&& _v = current_experiment_value.selection.get_entry();
        if(!_v)
        {
            OMNITRACE_UNLIKELY;
            return false;
        }
        auto _addr = _v->address();
        for(auto itr : _stack)
            if(_addr == itr->address()) return true;
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
experiment::load_experiments()
{
    auto _cfg         = settings::compose_filename_config{};
    _cfg.subdirectory = "causal";
    load_experiments("experiments", _cfg);
}

void
experiment::save_experiments(std::string _fname, const filename_config_t& _cfg)
{
    using cache_type = typename unwind_stack_t::cache_type;

    if(experiment_history.empty()) return;

    auto        _fname_orig = _fname;
    auto        _samples    = component::backtrace_causal::get_samples();
    const auto& _info0      = thread_info::get(0, InternalTID);
    auto        _rt0 = (_info0) ? (_info0->lifetime.second - _info0->lifetime.first) : 0;

    _fname = tim::settings::compose_output_filename(_fname_orig, "json", _cfg);
    std::stringstream oss{};
    {
        auto ar = tim::policy::output_archive<cereal::PrettyJSONOutputArchive>::get(oss);

        ar->setNextName("omnitrace");
        ar->startNode();
        ar->setNextName("causal");
        ar->startNode();
        (*ar)(cereal::make_nvp("experiments", experiment_history));
        (*ar)(cereal::make_nvp("runtime", _rt0));
        (*ar)(cereal::make_nvp("samples", _samples));
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
    if(tim::filepath::open(ofs, _fname))
    {
        if(get_verbose() >= 0)
            operation::file_output_message<experiment>{}(
                _fname, std::string{ "causal_experiments" });
        auto _cache = cache_type{ true };
        ofs << _existing.str();
        ofs << "startup\ttime=" << _info0->lifetime.first << "\n";
        for(auto& itr : experiment_history)
        {
            OMNITRACE_VERBOSE(0, "\n");
            auto _idx       = itr.selection.index;
            auto _selection = itr.selection.get(&_cache, false);
            if(_idx >= _selection.size())
                throw std::runtime_error(JOIN("", "Error! selection had index (", _idx,
                                              ") >= selection size (", _selection.size(),
                                              ")"));

            auto        _addr = (_selection.begin() + _idx)->address;
            std::string _name = {};
            for(auto& ditr : _selection)
            {
                if(!_name.empty()) break;
                auto _line_info_v = find_line_info(ditr.address);
                if(_line_info_v.is_valid())
                    _name = JOIN(":", _line_info_v.file, _line_info_v.line);
            }

            if(_name.empty())
            {
                for(auto& ditr : _selection)
                {
                    OMNITRACE_VERBOSE(0, "name=%s, loc=%s\n", ditr.name.c_str(),
                                      ditr.location.c_str());
                    if(ditr.lineinfo)
                    {
                        auto _lines = ditr.lineinfo.lines;
                        // std::reverse(_lines.begin(), _lines.end());
                        for(const auto& litr : _lines)
                        {
                            OMNITRACE_VERBOSE(0, "name=%s, loc=%s, line=%u\n",
                                              litr.name.c_str(), litr.location.c_str(),
                                              litr.line);
                            if(!litr.location.empty() && litr.line > 0)
                            {
                                _name = JOIN(":", litr.location, litr.line);
                                break;
                            }
                        }
                    }
                    if(!_name.empty() && _name.find("error") == std::string::npos) break;
                }
            }

            if(_name.empty())
            {
                throw std::runtime_error(JOIN("", "Error! selection has no name (", _idx,
                                              "). Address = 0x", std::hex, _addr));
                continue;
            }

            ofs << "experiment\tselected=" << demangle(_name)
                << "\tspeedup=" << static_cast<double>(itr.virtual_speedup / 100.0)
                << "\tduration=" << (itr.duration - itr.total_delay)
                << "\tselected-samples=" << itr.selected << "\n";
            ofs << "runtime\ttime=" << _rt0 << "\n";
            progress_points_t ppts = itr.end_progress;
            for(auto pitr : itr.start_progress)
                ppts[pitr.first] -= pitr.second;
            for(size_t i = 1; i < ppts.size(); ++i)
            {
                auto _init = ppts.begin();
                auto _next = ppts.begin();
                std::advance(_init, i - 1);
                std::advance(_next, i);
                _next->second.set_laps(_next->second.get_laps() -
                                       _init->second.get_laps());
            }

            for(auto pitr : ppts)
            {
                if(pitr.second.get_laps() == 0) continue;
                ofs << "throughput-point\tname="
                    << tim::demangle(tim::get_hash_identifier(pitr.first))
                    << "\tdelta=" << pitr.second.get_laps() << "\n";
                // ofs << "latency-point\tname=" << tim::get_hash_identifier(pitr.first)
                //    << "\tarrivals=" << pitr.second.get_laps()
                //    << "\tdepartures=" << pitr.second.get_laps()
                //    << "\tdifference=" << pitr.second.get_accum() << "\n";
            }

            for(const auto& sitr : _samples[itr.index])
            {
                if(sitr.lineinfo)
                {
                    for(const auto& litr : sitr.lineinfo.lines)
                    {
                        if(!litr.location.empty() && litr.line > 0)
                        {
                            ofs << "samples\tlocation="
                                << JOIN(":", litr.location, litr.line)
                                << "\tcount=" << sitr.count << "\n";
                        }
                    }
                }
                else
                {
                    ofs << "samples\tlocation="
                        << JOIN(":", tim::demangle(sitr.name), sitr.location)
                        << "\tcount=" << sitr.count << "\n";
                }
            }
        }
    }
    else
    {
        OMNITRACE_THROW("Error opening causal experiments output file: %s",
                        _fname.c_str());
    }
}

void
experiment::load_experiments(std::string _fname, const filename_config_t& _cfg)
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
        OMNITRACE_THROW("Error opening causal experiments input file: %s",
                        _fname.c_str());
    }

    experiment_history.reserve(experiment_history.size() + _experiments.size());
    for(auto& itr : _experiments)
        experiment_history.emplace_back(std::move(itr));
}
}  // namespace causal
}  // namespace omnitrace
