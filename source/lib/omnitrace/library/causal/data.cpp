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
#include "library/components/backtrace_causal.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
#include "library/state.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/utility.hpp"

#include <timemory/data/atomic_ring_buffer.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/log/logger.hpp>
#include <timemory/unwind/dlinfo.hpp>
#include <timemory/unwind/processed_entry.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <random>
#include <regex>
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
        return _v.hash();
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

auto speedup_seeds     = std::vector<size_t>{};
auto speedup_divisions = get_env<uint16_t>("OMNITRACE_CAUSAL_SPEEDUP_DIVISIONS", 5);
auto speedup_dist      = []() {
    size_t                _n = std::max<size_t>(1, 100 / speedup_divisions);
    std::vector<uint16_t> _v(_n, uint16_t{ 0 });
    std::generate(_v.begin(), _v.end(),
                  [_value = 0]() mutable { return (_value += speedup_divisions); });
    // approximately 25% of bins should be zero speedup
    size_t _nzero = std::ceil(_v.size() / 4.0);
    _v.resize(_v.size() + _nzero, 0);
    std::sort(_v.begin(), _v.end());
    OMNITRACE_CI_THROW(_v.back() > 100, "Error! last value is too large: %i\n",
                       (int) _v.back());
    return _v;
}();

auto
get_seed()
{
    static auto       _v = std::random_device{};
    static std::mutex _mutex;
    std::scoped_lock  _lk{ _mutex };
    speedup_seeds.emplace_back(_v());
    return speedup_seeds.back();
}

auto&
get_engine(int64_t _tid = utility::get_thread_index())
{
    using thread_data_t = thread_data<identity<random_engine_t>, experiment>;
    static auto& _v =
        thread_data_t::instances(construct_on_init{}, []() { return get_seed(); });
    return _v.at(_tid);
}

void
perform_experiment_impl()
{
    const auto& _thr_info = thread_info::init(true);
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    OMNITRACE_CONDITIONAL_THROW(!_thr_info->is_offset,
                                "Error! causal profiling thread should be offset");

    auto _impl_count = 0;
    while(get_state() < State::Finalized)
    {
        auto _impl_no = _impl_count++;
        auto _experim = experiment{};

        // loop until started or finalized
        while(!_experim.start())
        {
            if(get_state() == State::Finalized)
            {
                OMNITRACE_CONDITIONAL_THROW(_impl_no == 0, "experiment never started\n");
                return;
            }
        }

        // wait for the experiment to complete
        if(config::get_causal_end_to_end())
        {
            while(get_state() < State::Finalized)
            {
                std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
            }
        }
        else
        {
            _experim.wait();
        }

        while(!_experim.stop())
        {
            if(get_state() == State::Finalized) return;
        }
    }
}

auto&
get_unwind_cache()
{
    static auto _v = unwind_cache_t{ true };
    return _v;
}

std::set<code_object::address_range>&
get_eligible_address_ranges()
{
    static auto _v = std::set<code_object::address_range>{};
    return _v;
}

using line_info_map_t = code_object::line_info_map<code_object::ext_line_info>;

std::pair<line_info_map_t, line_info_map_t>&
get_cached_line_info()
{
    static auto _v = []() {
        auto _discarded = line_info_map_t{};
        auto _requested = code_object::get_ext_line_info(&_discarded);
        return std::make_pair(_requested, _discarded);
    }();
    return _v;
}

auto
compute_eligible_lines()
{
    auto _v =
        std::unordered_map<uintptr_t,
                           std::vector<std::unique_ptr<code_object::basic::line_info>>>{};
    auto        _specific_lines = config::get_causal_fixed_line();
    auto        _specific_funcs = config::get_causal_fixed_function();
    const auto& _line_info      = get_cached_line_info().first;

    auto _add_line_info = [&_v](auto _ip_val, const auto& _li_val) {
        for(const auto& vitr : _v[_ip_val])
            if(*vitr == _li_val) return;
        _v[_ip_val].emplace_back(
            std::make_unique<code_object::basic::line_info>(_li_val));
    };

    if(!_specific_lines.empty())
    {
        for(const auto& itr : _specific_lines)
        {
            auto _re = std::regex{ itr + "$" };
            for(const auto& litr : _line_info)
            {
                auto _range = code_object::address_range{ litr.first.load_address,
                                                          litr.first.last_address };
                for(const auto& ditr : litr.second)
                {
                    auto _li =
                        std::visit([](auto&& _val) { return _val.get_basic(); }, ditr);
                    // match to <file> or <file>:<line>
                    if(std::regex_search(_li.file, _re) == false &&
                       std::regex_search(JOIN(':', _li.file, _li.line), _re) == false)
                        continue;
                    // map the instruction pointer address to the line info
                    auto _ip = litr.first.load_address + _li.address;
                    _add_line_info(_ip, _li);
                    get_eligible_address_ranges().emplace(_range);
                }
            }
        }
    }

    if(!_specific_funcs.empty())
    {
        for(const auto& itr : _specific_funcs)
        {
            auto _re = std::regex{ itr, std::regex_constants::optimize };
            for(const auto& litr : _line_info)
            {
                auto _range = code_object::address_range{ litr.first.load_address,
                                                          litr.first.last_address };
                for(const auto& ditr : litr.second)
                {
                    auto _li =
                        std::visit([](auto&& _val) { return _val.get_basic(); }, ditr);
                    // match to <file> or <file>:<line>
                    if(std::regex_search(_li.func, _re) == false &&
                       std::regex_search(demangle(_li.func), _re) == false)
                        continue;
                    // map the instruction pointer address to the line info
                    auto _ip = litr.first.load_address + _li.address;
                    _add_line_info(_ip, _li);
                    get_eligible_address_ranges().emplace(_range);
                }
            }
        }
    }

    if(_v.empty())
    {
        for(const auto& litr : _line_info)
        {
            auto _range = code_object::address_range{ litr.first.load_address,
                                                      litr.first.last_address };
            for(const auto& ditr : litr.second)
            {
                auto _li = std::visit([](auto&& _val) { return _val.get_basic(); }, ditr);
                auto _ip = litr.first.load_address + _li.address;
                _add_line_info(_ip, _li);
                get_eligible_address_ranges().emplace(_range);
            }
        }
    }
    return _v;
}

auto&
get_eligible_lines()
{
    static auto _v = compute_eligible_lines();
    return _v;
}

// thread-safe read/write ring-buffer via atomics
using pc_ring_buffer_t = tim::data_storage::atomic_ring_buffer<uintptr_t>;
// latest_eligible_pcs is an array of unwind_depth size -> samples will
// use lowest indexes for most recent functions address in the call-stack
auto latest_eligible_pc = []() {
    auto _arr = std::array<std::unique_ptr<pc_ring_buffer_t>, unwind_depth>{};
    for(auto& itr : _arr)
        itr = std::make_unique<pc_ring_buffer_t>(units::get_page_size() /
                                                 (sizeof(uintptr_t) + 1));
    return _arr;
}();
}  // namespace

//--------------------------------------------------------------------------------------//

template <typename Tp>
std::string
as_hex(Tp _v, size_t _width)
{
    std::stringstream _ss;
    _ss.fill('0');
    _ss << "0x" << std::hex << std::setw(_width) << _v;
    return _ss.str();
}

template std::string as_hex<int32_t>(int32_t, size_t);
template std::string as_hex<uint32_t>(uint32_t, size_t);
template std::string as_hex<int64_t>(int64_t, size_t);
template std::string as_hex<uint64_t>(uint64_t, size_t);
template std::string
as_hex<void*>(void*, size_t);

bool
is_eligible_address(uintptr_t _v)
{
    return get_eligible_address_ranges().count(_v) > 0;
}

void
save_line_info(const settings::compose_filename_config& _cfg)
{
    auto _write_impl = [](std::ofstream& _ofs, const auto& _data) {
        for(auto& itr : _data)
        {
            _ofs << itr.first.pathname << " [" << as_hex(itr.first.load_address) << " - "
                 << as_hex(itr.first.last_address) << "]\n";

            for(auto& ditr : itr.second)
            {
                auto _li   = std::visit([](auto&& _v) { return _v.get_basic(); }, ditr);
                auto _addr = _li.address;
                auto _addr_off = _li.address + itr.first.load_address;
                _ofs << "    " << as_hex(_addr_off) << " [" << as_hex(_addr)
                     << "] :: " << _li.file << ":" << _li.line;
                if(!_li.func.empty()) _ofs << " [" << tim::demangle(_li.func) << "]";
                _ofs << "\n";
            }
        }

        _ofs << "\n" << std::flush;
    };

    auto _write = [&_write_impl](const std::string& ofname, const auto& _data) {
        auto _ofs = std::ofstream{};
        if(tim::filepath::open(_ofs, ofname))
        {
            if(config::get_verbose() >= 0)
                operation::file_output_message<code_object::basic_line_info>{}(
                    ofname, std::string{ "causal_line_info" });
            _write_impl(_ofs, _data);
        }
        else
        {
            throw std::runtime_error("Error opening " + ofname);
        }
    };

    _write(tim::settings::compose_output_filename("line-info-included", "txt", _cfg),
           get_cached_line_info().first);
    _write(tim::settings::compose_output_filename("line-info-discarded", "txt", _cfg),
           get_cached_line_info().second);
}

void
set_current_selection(utility::c_array<void*> _stack)
{
    if(experiment::is_active()) return;

    size_t _n = 0;
    for(auto* itr : _stack)
    {
        auto& _pcs  = latest_eligible_pc.at(_n);
        auto  _addr = reinterpret_cast<uintptr_t>(itr);
        if(_pcs && is_eligible_address(_addr))
        {
            _pcs->write(&_addr);
            // increment after valid found -> first valid pc for call-stack
            ++_n;
        }
    }
}

void
set_current_selection(unwind_stack_t _stack)
{
    if(experiment::is_active()) return;

    size_t _n = 0;
    for(auto itr : _stack)
    {
        auto& _pcs  = latest_eligible_pc.at(_n);
        auto  _addr = itr->address();
        if(_pcs && is_eligible_address(_addr))
        {
            _pcs->write(&_addr);
            // increment after valid found -> first valid pc for call-stack
            ++_n;
        }
    }
}

selected_entry
sample_selection(size_t _nitr, size_t _wait_ns)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto&  _eligible_pcs   = get_eligible_lines();
    size_t _n              = 0;
    auto   _select_address = [&](auto& _addresses) {
        if(_addresses.empty()) return selected_entry{};
        auto _dist = std::uniform_int_distribution<size_t>{ 0, _addresses.size() - 1 };
        auto _idx  = _dist(get_engine());
        auto _addr = _addresses.at(_idx);
        if(get_causal_mode() == CausalMode::Function)
        {
            auto      _dl_info  = unwind::dlinfo::construct(_addr);
            uintptr_t _sym_addr = (_dl_info.symbol) ? _dl_info.symbol.address() : _addr;
            auto&     pitrs     = _eligible_pcs.at(_sym_addr);
            for(auto ritr = pitrs.rbegin(); ritr != pitrs.rend(); ++ritr)
            {
                auto& pitr = *ritr;
                // skip lambdas since these provide no relevant info
                if(pitr->inlined && demangle(pitr->func).find("operator()") == 0)
                    continue;
                OMNITRACE_VERBOSE(-1, "Selected address %s ('%s') for experiment...\n",
                                  as_hex(_sym_addr).c_str(),
                                  demangle(pitr->func).c_str());
                return selected_entry{ _addr, _sym_addr, *pitr };
            }
            auto& pitr = _eligible_pcs.at(_sym_addr).back();
            OMNITRACE_VERBOSE(-1, "Selected address %s ('%s') for experiment...\n",
                              as_hex(_sym_addr).c_str(), demangle(pitr->func).c_str());
            return selected_entry{ _addr, _sym_addr, *pitr };
        }
        else if(get_causal_mode() == CausalMode::Line)
        {
            auto& pitrs = _eligible_pcs.at(_addr);
            for(auto ritr = pitrs.rbegin(); ritr != pitrs.rend(); ++ritr)
            {
                auto& pitr = *ritr;
                // skip lambdas since these provide no relevant info
                if(pitr->inlined && demangle(pitr->func).find("operator()") == 0)
                    continue;
                OMNITRACE_VERBOSE(0,
                                  "Selected address %s (%s) for experiment from %zu "
                                  "eligible addresses...\n",
                                  as_hex(_addr).c_str(), demangle(pitr->func).c_str(),
                                  _eligible_pcs.size());
                return selected_entry{ _addr, 0, *pitr };
            }
        }
        return selected_entry{};
    };

    while(_n++ < _nitr)
    {
        auto _addresses = std::vector<uintptr_t>{};
        for(auto& aitr : latest_eligible_pc)
        {
            if(!aitr) continue;
            auto _naddrs = aitr->count();
            _addresses.reserve(_addresses.size() + _naddrs);
            for(size_t i = 0; i < _naddrs; ++i)
            {
                uintptr_t _addr = 0;
                if(aitr->read(&_addr) != nullptr)
                {
                    if(_eligible_pcs.count(_addr) > 0) _addresses.emplace_back(_addr);
                }
            }

            if(!_addresses.empty()) return _select_address(_addresses);
        }

        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _wait_ns });
    }

    return selected_entry{};
}

std::vector<code_object::basic::line_info>
get_line_info(uintptr_t _addr, bool include_discarded)
{
    auto _data          = std::vector<code_object::basic::line_info>{};
    auto _get_line_info = [&](const auto& _info) {
        for(const auto& litr : _info)
        {
            for(const auto& ditr : litr.second)
            {
                auto _li = std::visit([](auto&& _v) { return _v.get_basic(); }, ditr);
                auto _ip = litr.first.load_address + _li.address;
                if(_ip == _addr) _data.emplace_back(_li);
            }
        }
    };
    _get_line_info(get_cached_line_info().first);
    if(include_discarded) _get_line_info(get_cached_line_info().second);
    return _data;
}

void
push_progress_point(std::string_view _name)
{
    if(!config::get_causal_end_to_end() && !experiment::is_active()) return;

    auto  _hash   = tim::add_hash_id(_name);
    auto& _data   = progress_bundles_t::instance(utility::get_thread_index());
    auto* _bundle = _data.construct(_hash);
    _bundle->push();
    _bundle->start();
}

void
pop_progress_point(std::string_view _name)
{
    if(!config::get_causal_end_to_end() && !experiment::is_active()) return;

    auto& _data = progress_bundles_t::instance(utility::get_thread_index());
    if(_data.empty()) return;
    if(_name.empty())
    {
        auto* itr = _data.back();
        itr->stop();
        itr->pop();
        _data.pop_back();
        return;
    }
    else
    {
        auto _hash = tim::add_hash_id(_name);
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
}

uint16_t
sample_virtual_speedup(int64_t _tid)
{
    using thread_data_t =
        thread_data<identity<std::uniform_int_distribution<size_t>>, experiment>;
    static auto& _v = thread_data_t::instances(construct_on_init{}, size_t{ 0 },
                                               speedup_dist.size() - 1);
    return speedup_dist.at(_v.at(_tid)(get_engine(_tid)));
}

void
start_experimenting()
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto _user_speedup_dist = config::get_causal_fixed_speedup();
    if(!_user_speedup_dist.empty())
    {
        speedup_dist.clear();
        for(auto itr : _user_speedup_dist)
        {
            OMNITRACE_CONDITIONAL_ABORT_F(itr > 100,
                                          "Virtual speedups must be in range [0, 100]. "
                                          "Invalid virtual speedup: %lu\n",
                                          itr);
            speedup_dist.emplace_back(static_cast<uint16_t>(itr));
        }
    }

    (void) get_eligible_lines();

    // wait until fully activated before proceeding
    while(utility::get_thread_index() > 0 && get_state() < State::Active)
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
    }

    if(get_state() < State::Finalized) std::thread{ perform_experiment_impl }.detach();
}
}  // namespace causal
}  // namespace omnitrace
