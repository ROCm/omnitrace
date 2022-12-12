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

#include "library/causal/data.hpp"
#include "library/causal/delay.hpp"
#include "library/causal/experiment.hpp"
#include "library/code_object.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
#include "library/state.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/utility.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/unwind/processed_entry.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace std
{
template <size_t N>
struct hash<tim::unwind::stack<N>>
{
    size_t operator()(const tim::unwind::stack<N>& _v) const { return _v.hash(); }
};

template <>
struct hash<omnitrace::causal::selected_entry>
{
    size_t operator()(const omnitrace::causal::selected_entry& _v) const
    {
        if(_v.index == omnitrace::causal::unwind_depth) return 0;
        return _v.stack.hash(_v.index);
    }
};

template struct hash<tim::unwind::stack<8>>;
}  // namespace std

namespace omnitrace
{
namespace causal
{
namespace
{
using random_engine_t    = std::mt19937_64;
using progress_bundles_t = component_bundle_cache<progress_point>;

uint16_t              increment            = 5;
std::vector<uint16_t> virtual_speedup_dist = []() {
    size_t                _n = std::max<size_t>(1, 100 / increment);
    std::vector<uint16_t> _v(_n, uint16_t{ 0 });
    std::generate(_v.begin(), _v.end(),
                  [_value = 0]() mutable { return (_value += increment); });
    OMNITRACE_CI_THROW(_v.back() > 100, "Error! last value is too large: %i\n",
                       (int) _v.back());
    return _v;
}();
std::vector<size_t> seeds = {};

auto
get_seed()
{
    static auto       _v = std::random_device{};
    static std::mutex _mutex;
    std::unique_lock  _lk{ _mutex };
    seeds.emplace_back(_v());
    return seeds.back();
}

auto&
get_engine(int64_t _tid = utility::get_thread_index())
{
    using thread_data_t = thread_data<identity<random_engine_t>, experiment>;
    static auto& _v =
        thread_data_t::instances(construct_on_init{}, []() { return get_seed(); });
    return _v.at(_tid);
}

std::atomic<int64_t>&
get_max_thread_index()
{
    static auto _v = std::atomic<int64_t>{ 1 };
    return _v;
}

void
update_max_thread_index(int64_t _tid)
{
    bool _success = false;
    while(!_success)
    {
        auto _v = get_max_thread_index().load();
        if(_tid <= _v) break;
        _success = get_max_thread_index().compare_exchange_strong(
            _v, _tid, std::memory_order_relaxed);
    }
}

int64_t
sample_thread_index(int64_t _tid)
{
    return std::uniform_int_distribution<int64_t>{ 0, get_max_thread_index().load() }(
        get_engine(_tid));
}

void
perform_experiment_impl()
{
    thread_info::init(true);
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto _experim = experiment{};
    // loop until started or finalized
    while(!_experim.start())
    {
        if(get_state() == State::Finalized) return;
    }

    // wait for the experiment to complete
    _experim.wait();

    while(!_experim.stop())
    {
        if(get_state() == State::Finalized) return;
    }

    if(get_state() == State::Finalized) return;

    // cool-down / inactive period
    std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });

    // start a new experiment
    tasking::general::get_task_group().exec(perform_experiment_impl);
}

auto&
get_unwind_cache()
{
    static auto _v = unwind_cache_t{ true };
    return _v;
}

using line_info_map_t = std::map<code_object::procfs::maps, code_object::basic_line_info>;

std::pair<line_info_map_t, line_info_map_t>&
get_cached_line_info()
{
    static auto _v = []() {
        auto _discarded = line_info_map_t{};
        auto _requested = code_object::get_basic_line_info(&_discarded);
        return std::make_pair(_requested, _discarded);
    }();
    return _v;
}

auto
compute_eligible_pcs()
{
    /*
    auto _write = [](std::ofstream& _ofs, const auto& _data) {
        auto _as_hex = [](auto&& _v) {
            std::stringstream _ss{};
            _ss.fill('0');
            _ss << "0x" << std::hex << std::setw(16) << _v;
            return _ss.str();
        };

        for(auto& itr : _data)
        {
            _ofs << itr.first.pathname << " [" << _as_hex(itr.first.start_address)
                 << " - " << _as_hex(itr.first.end_address) << "]\n";

            for(auto& ditr : itr.second)
            {
                auto _addr     = ditr.address;
                auto _addr_off = ditr.address + itr.first.start_address;
                _ofs << "    " << _as_hex(_addr_off) << " [" << _as_hex(_addr)
                     << "] :: " << ditr.file << ":" << ditr.line;
                if(!ditr.func.empty()) _ofs << " [" << tim::demangle(ditr.func) << "]";
                _ofs << "\n";
            }
        }

        _ofs << "\n" << std::flush;
    };

    {
        std::string ofname = "codemap.txt";
        std::ofstream _ofs{ ofname };
        if(!_ofs) throw std::runtime_error("Error opening " + ofname);
    }*/

    auto        _v              = std::unordered_set<uintptr_t>{};
    auto        _specific_lines = config::get_causal_fixed_line();
    const auto& _line_info      = get_cached_line_info().first;
    if(!_specific_lines.empty())
    {
        for(const auto& itr : _specific_lines)
        {
            auto _entry = tim::delimit(itr, ":");
            if(_entry.size() == 2)
            {
                auto _file_re = std::regex{ _entry.front() + "$" };
                auto _line_no = std::stoul(_entry.back());
                for(const auto& litr : _line_info)
                {
                    for(const auto& ditr : litr.second)
                    {
                        if(ditr.line == _line_no &&
                           std::regex_search(ditr.file, _file_re))
                        {
                            auto _addr_off = litr.first.start_address + ditr.address;
                            _v.emplace(_addr_off);
                        }
                    }
                }
            }
            else if(_entry.size() == 1)
            {
                auto _file_re = std::regex{ _entry.front() + "$" };
                for(const auto& litr : _line_info)
                {
                    for(const auto& ditr : litr.second)
                    {
                        if(std::regex_search(ditr.file, _file_re))
                        {
                            auto _addr_off = litr.first.start_address + ditr.address;
                            _v.emplace(_addr_off);
                        }
                    }
                }
            }
        }
    }
    else
    {
        for(const auto& litr : _line_info)
        {
            for(const auto& ditr : litr.second)
            {
                auto _addr_off = litr.first.start_address + ditr.address;
                _v.emplace(_addr_off);
            }
        }
    }
    return _v;
}

auto&
get_eligible_pcs()
{
    static auto _v = compute_eligible_pcs();
    return _v;
}

auto latest_progress_point = std::unordered_set<selected_entry>{};
auto latest_progress_mutex = std::mutex{};
}  // namespace

//--------------------------------------------------------------------------------------//

void
set_current_selection(unwind_stack_t _stack)
{
    if(experiment::is_active()) return;

    size_t _idx      = 0;
    auto   _prog_pts = std::unordered_set<selected_entry>{};
    for(auto itr : _stack)
    {
        if(itr && get_eligible_pcs().count(itr->address()) > 0)
            _prog_pts.emplace(selected_entry{ _idx, _stack });
        ++_idx;
    }

    if(!_prog_pts.empty())
    {
        std::scoped_lock _lk{ latest_progress_mutex };
        for(auto itr : _prog_pts)
            latest_progress_point.emplace(itr);
    }
}

selected_entry
sample_selection(size_t _nitr, size_t _wait_ns)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    size_t _n = 0;

    while(_n++ < _nitr)
    {
        {
            std::scoped_lock _lk{ latest_progress_mutex };
            auto&            _lpp = latest_progress_point;
            if(!_lpp.empty())
            {
                auto _dist = std::uniform_int_distribution<size_t>{ 0, _lpp.size() - 1 };
                auto itr   = _lpp.begin();
                auto _idx  = _dist(get_engine());
                std::advance(itr, _idx);
                return *itr;
            }
        }

        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _wait_ns });
    }

    return selected_entry{};
}

void
push_progress_point(std::string_view _name)
{
    auto  _hash   = tim::add_hash_id(_name);
    auto& _data   = progress_bundles_t::instance(utility::get_thread_index());
    auto* _bundle = _data.construct(_hash);
    _bundle->push();
    _bundle->start();
}

void
pop_progress_point(std::string_view _name)
{
    auto  _hash = tim::add_hash_id(_name);
    auto& _data = progress_bundles_t::instance(utility::get_thread_index());
    for(auto itr = _data.rbegin(); itr != _data.rend(); ++itr)
    {
        if((*itr)->get_hash() == _hash)
        {
            (*itr)->stop();
            (*itr)->pop();
            _data.destroy(itr);
            return;
        }
    }
}

bool
sample_enabled(int64_t _tid)
{
    using thread_data_t =
        thread_data<identity<std::uniform_int_distribution<int>>, experiment>;
    static auto& _v = thread_data_t::instances(construct_on_init{}, 0, 1);
    return (_v.at(_tid)(get_engine(_tid)) == 1);
}

uint16_t
sample_virtual_speedup(int64_t _tid)
{
    using thread_data_t =
        thread_data<identity<std::uniform_int_distribution<size_t>>, experiment>;
    static auto& _v = thread_data_t::instances(construct_on_init{}, size_t{ 0 },
                                               virtual_speedup_dist.size() - 1);
    return virtual_speedup_dist.at(_v.at(_tid)(get_engine(_tid)));
}

void
start_experimenting()
{
    thread_info::init(true);

    // wait until fully activated before proceeding
    while(get_state() < State::Active)
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
    }

    if(get_state() != State::Finalized)
    {
        // start a new experiment
        tasking::general::get_task_group().exec(perform_experiment_impl);
    }
}

const std::map<code_object::procfs::maps, code_object::basic_line_info>&
get_causal_line_info()
{
    return get_cached_line_info().first;
}

const std::map<code_object::procfs::maps, code_object::basic_line_info>&
get_causal_ignored_line_info()
{
    return get_cached_line_info().second;
}

code_object::basic::line_info
find_line_info(uintptr_t _addr, bool _include_ignored)
{
    auto _find_line_info = [](auto _address, auto& _line_info) {
        for(const auto& litr : _line_info)
        {
            for(const auto& eitr : litr.second)
            {
                auto _addr_off = litr.first.start_address + eitr.address;
                if(_addr_off == _address)
                {
                    return eitr;
                }
            }
        }
        return code_object::basic::line_info{};
    };

    auto _v = _find_line_info(_addr, get_cached_line_info().first);
    if(_include_ignored && !_v.is_valid())
        _v = _find_line_info(_addr, get_cached_line_info().second);
    return _v;
}

}  // namespace causal
}  // namespace omnitrace
