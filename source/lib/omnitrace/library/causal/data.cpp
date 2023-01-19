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
#include "library/binary/address_multirange.hpp"
#include "library/binary/analysis.hpp"
#include "library/binary/basic_line_info.hpp"
#include "library/binary/bfd_line_info.hpp"
#include "library/binary/fwd.hpp"
#include "library/binary/line_info.hpp"
#include "library/binary/link_map.hpp"
#include "library/binary/scope_filter.hpp"
#include "library/causal/delay.hpp"
#include "library/causal/experiment.hpp"
#include "library/causal/sampling.hpp"
#include "library/causal/selected_entry.hpp"
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
#include <timemory/mpl/concepts.hpp>
#include <timemory/units.hpp>
#include <timemory/unwind/dlinfo.hpp>
#include <timemory/unwind/processed_entry.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <random>
#include <regex>
#include <thread>
#include <utility>
#include <vector>

namespace omnitrace
{
namespace causal
{
namespace
{
using random_engine_t    = std::mt19937_64;
using progress_bundles_t = component_bundle_cache<component::progress_point>;

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

auto perform_experiment_impl_completed = std::unique_ptr<std::promise<void>>{};

template <typename ContextT>
auto&
get_engine()
{
    static auto _seed = []() -> hash_value_t {
        auto _seed_v =
            config::get_setting_value<uint64_t>("OMNITRACE_CAUSAL_RANDOM_SEED").second;
        if(_seed_v == 0) _seed_v = std::random_device{}();
        return _seed_v;
    }();

    static thread_local auto _v =
        random_engine_t{ tim::get_combined_hash_id(_seed, utility::get_thread_index()) };
    return _v;
}

binary::address_multirange&
get_eligible_address_ranges()
{
    static auto _v = binary::address_multirange{};
    return _v;
}

auto
get_filters()
{
    using sf      = binary::scope_filter;
    auto _filters = std::vector<binary::scope_filter>{};

    // exclude internal libraries used by omnitrace
    _filters.emplace_back(
        sf{ sf::FILTER_EXCLUDE, sf::BINARY_FILTER,
            "lib(omnitrace[-\\.]|dyninst|tbbmalloc|gotcha\\.|unwind\\.so\\.99)" });

    // in function mode, it generally doesn't help to experiment on main function since
    // telling the user to "make the main function" faster is literally useless since it
    // contains everything that could be made faster
    if(config::get_causal_mode() == CausalMode::Function)
        _filters.emplace_back(
            sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER, "( main\\(|^main$)" });

    bool _use_default_excludes =
        config::get_setting_value<bool>("OMNITRACE_CAUSAL_FUNCTION_EXCLUDE_DEFAULTS")
            .second;

    if(_use_default_excludes)
    {
        // symbols starting with leading underscore are generally system functions
        _filters.emplace_back(sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER, "^_" });

        if(config::get_causal_mode() == CausalMode::Function)
        {
            // exclude STL implementation functions
            _filters.emplace_back(sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER, "::_M" });
        }
    }

    // in function mode, it generally doesn't help to claim
    // "make main function" faster since it contains everything
    // that could be made faster
    if(config::get_causal_mode() == CausalMode::Function)
    {
        _filters.emplace_back(
            sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER, "(^main$|int main\\()" });
    }

    using utility::get_regex_or;

    auto _source_end_converter = [](const std::string& _v) { return _v + "$"; };

    // include handling
    {
        auto _binary_include = get_regex_or(config::get_causal_binary_scope(), "");
        auto _source_include =
            get_regex_or(config::get_causal_source_scope(), _source_end_converter, "");
        auto _function_include = get_regex_or(config::get_causal_function_scope(), "");

        auto _current_include =
            std::make_tuple(_binary_include, _source_include, _function_include);
        static auto _former_include = decltype(_current_include){};

        if(_former_include != _current_include)
        {
            if(!_binary_include.empty())
                OMNITRACE_VERBOSE(0, "[causal] binary scope     : %s\n",
                                  _binary_include.c_str());
            if(!_source_include.empty())
                OMNITRACE_VERBOSE(0, "[causal] source scope     : %s\n",
                                  _source_include.c_str());
            if(!_function_include.empty())
                OMNITRACE_VERBOSE(0, "[causal] function scope   : %s\n",
                                  _function_include.c_str());
            _former_include = _current_include;
        }

        if(!_binary_include.empty())
            _filters.emplace_back(
                sf{ sf::FILTER_INCLUDE, sf::BINARY_FILTER, _binary_include });

        if(!_source_include.empty())
            _filters.emplace_back(
                sf{ sf::FILTER_INCLUDE, sf::SOURCE_FILTER, _source_include });

        if(!_function_include.empty())
            _filters.emplace_back(
                sf{ sf::FILTER_INCLUDE, sf::FUNCTION_FILTER, _function_include });
    }

    // exclude handling
    {
        auto _binary_exclude = get_regex_or(config::get_causal_binary_exclude(), "");
        auto _source_exclude =
            get_regex_or(config::get_causal_source_exclude(), _source_end_converter, "");
        auto _function_exclude = get_regex_or(config::get_causal_function_exclude(), "");

        auto _current_exclude =
            std::make_tuple(_binary_exclude, _source_exclude, _function_exclude);
        static auto _former_exclude = decltype(_current_exclude){};

        if(_former_exclude != _current_exclude)
        {
            if(!_binary_exclude.empty())
                OMNITRACE_VERBOSE(0, "[causal] binary exclude   : %s\n",
                                  _binary_exclude.c_str());
            if(!_source_exclude.empty())
                OMNITRACE_VERBOSE(0, "[causal] source exclude   : %s\n",
                                  _source_exclude.c_str());
            if(!_function_exclude.empty())
                OMNITRACE_VERBOSE(0, "[causal] function exclude : %s\n",
                                  _function_exclude.c_str());
            _former_exclude = _current_exclude;
        }

        if(!_binary_exclude.empty())
            _filters.emplace_back(
                sf{ sf::FILTER_EXCLUDE, sf::BINARY_FILTER, _binary_exclude });

        if(!_source_exclude.empty())
            _filters.emplace_back(
                sf{ sf::FILTER_EXCLUDE, sf::SOURCE_FILTER, _source_exclude });

        if(!_function_exclude.empty())
            _filters.emplace_back(
                sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER, _function_exclude });
    }

    return _filters;
}

std::pair<binary::line_info_t, binary::line_info_t>&
get_cached_line_info()
{
    static auto _v = []() {
        // get the linked binaries for the exe (excluding ones from libomnitrace)
        auto _link_map = binary::get_link_map();
        auto _files    = std::vector<std::string>{};
        _files.reserve(_link_map.size());
        for(const auto& itr : _link_map)
            _files.emplace_back(itr.real());

        auto _discarded = binary::line_info_t{};
        auto _requested = binary::get_line_info(_files, get_filters(), &_discarded);
        return std::make_pair(_requested, _discarded);
    }();
    return _v;
}

auto
compute_eligible_lines()
{
    const auto& _line_info = get_cached_line_info().first;
    auto        _v = std::unordered_map<uintptr_t, std::deque<binary::basic_line_info>>{};

    auto _add_line_info = [&_v](auto _ip_val, const auto& _bfd_val) {
        auto _basic_val = _bfd_val.get_basic();
        for(uintptr_t i = _ip_val.low; i < _ip_val.high + 1; ++i)
        {
            for(const auto& vitr : _v[i])
                if(vitr == _basic_val) continue;
            _v[i].emplace_back(_basic_val);
        }
    };

    auto _filters         = get_filters();
    auto satisfies_filter = [&_filters](auto _scope, const std::string& _value) {
        for(const auto& itr : _filters)  // NOLINT
        {
            // if the filter is for the specified scope and itr does not satisfy the
            // include/exclude mode, return false
            if((itr.scope & _scope) > 0 && !itr(_value)) return false;
        }
        return true;
    };

    using sf = binary::scope_filter;

    bool _use_custom_filters = false;
    for(const auto& itr : _filters)
    {
        if(itr.mode == sf::FILTER_INCLUDE &&
           (itr.scope == sf::FUNCTION_FILTER || itr.scope == sf::SOURCE_FILTER))
        {
            _use_custom_filters = true;
            break;
        }
    }

    auto& _eligible_ar = get_eligible_address_ranges();
    for(const auto& litr : _line_info)
    {
        bool _valid = false;
        auto _range = address_range_t{ litr.first.load_address, litr.first.last_address };
        for(const auto& ditr : litr.second)
        {
            if(_use_custom_filters)
            {
                // check both <file> and <file>:<line>
                if(!satisfies_filter(sf::SOURCE_FILTER, ditr.file) &&
                   !satisfies_filter(sf::SOURCE_FILTER, JOIN(':', ditr.file, ditr.line)))
                    continue;

                // only check demangled function name since things like ^_ can
                // accidentally catch mangled C++ function names
                if(!satisfies_filter(sf::FUNCTION_FILTER, demangle(ditr.func))) continue;
            }

            // map the instruction pointer address to the line info
            _add_line_info(ditr.ipaddr(), ditr);
            if(!_valid)
                _valid = (_eligible_ar +=
                          std::make_pair(binary::address_multirange::coarse{}, _range),
                          true);

            _eligible_ar += ditr.ipaddr();
        }
    }

    OMNITRACE_VERBOSE(0,
                      "[causal] eligible addresses: %zu, eligible address ranges: %zu, "
                      "total range: %zu [%s]\n",
                      _v.size(), _eligible_ar.size(), _eligible_ar.range_size(),
                      _eligible_ar.coarse_range.as_string().c_str());

    if(_eligible_ar.empty())
    {
        auto _cfg         = settings::compose_filename_config{};
        _cfg.subdirectory = "causal/line-info";
        save_line_info(_cfg, config::get_verbose());
    }

    OMNITRACE_CONDITIONAL_THROW(
        _eligible_ar.empty(),
        "Error! binary analysis (after filters) resulted in zero eligible instruction "
        "pointer addresses for causal experimentation");

    return _v;
}

auto&
get_eligible_lines()
{
    static auto _v = compute_eligible_lines();
    return _v;
}

void
perform_experiment_impl(std::shared_ptr<std::promise<void>> _started)  // NOLINT
{
    using clock_type      = std::chrono::high_resolution_clock;
    using duration_nsec_t = std::chrono::duration<double, std::nano>;
    using duration_sec_t  = std::chrono::duration<double, std::ratio<1>>;

    const auto& _thr_info = thread_info::init(true);
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    OMNITRACE_CONDITIONAL_THROW(!_thr_info->is_offset,
                                "Error! causal profiling thread should be offset");

    if(!perform_experiment_impl_completed)
        perform_experiment_impl_completed = std::make_unique<std::promise<void>>();

    perform_experiment_impl_completed->set_value_at_thread_exit();

    (void) get_eligible_lines();

    // notify that thread has started
    if(_started) _started->set_value();

    // pause at least one second to determine sampling rate
    std::this_thread::sleep_for(std::chrono::seconds{ 1 });

    auto _cfg         = settings::compose_filename_config{};
    _cfg.subdirectory = "causal/line-info";
    save_line_info(_cfg, config::get_verbose());

    double _delay_sec =
        config::get_setting_value<double>("OMNITRACE_CAUSAL_DELAY").second;
    double _duration_sec =
        config::get_setting_value<double>("OMNITRACE_CAUSAL_DURATION").second;
    auto _duration_nsec = duration_nsec_t{ _duration_sec * units::sec };

    if(_delay_sec > 0.0)
    {
        OMNITRACE_VERBOSE(1, "[causal] delaying experimentation for %.2f seconds...\n",
                          _delay_sec);
        uint64_t _delay_nsec = _delay_sec * units::sec;
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _delay_nsec });
    }

    auto _impl_count        = 0;
    auto _start_time        = clock_type::now();
    auto _exceeded_duration = [&]() {
        if(_duration_sec > 1.0e-3)
        {
            auto _elapsed = clock_type::now() - _start_time;
            if(_elapsed >= _duration_nsec)
            {
                OMNITRACE_VERBOSE(
                    1,
                    "[causal] stopping experimentation after %.2f seconds "
                    "(elapsed: %.2f seconds)...\n",
                    _duration_sec,
                    std::chrono::duration_cast<duration_sec_t>(_elapsed).count());
                causal::sampling::post_process();
                return true;
            }
        }
        return false;
    };

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
                if(_exceeded_duration()) return;
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

        if(_exceeded_duration()) return;
    }
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

bool
is_eligible_address(uintptr_t _v)
{
    return get_eligible_address_ranges().contains(_v);
}

void
save_line_info(const settings::compose_filename_config& _cfg, int _verbose)
{
    auto _write_impl = [](std::ofstream& _ofs, const binary::line_info_t& _data) {
        for(const auto& itr : _data)
        {
            _ofs << itr.first.pathname << " [" << as_hex(itr.first.load_address) << " - "
                 << as_hex(itr.first.last_address) << "]\n";

            for(const auto& ditr : itr.second)
            {
                auto _addr     = ditr.address;
                auto _addr_off = ditr.ipaddr();
                _ofs << "    " << as_hex(_addr_off) << " [" << as_hex(_addr)
                     << "] :: " << ditr.file << ":" << ditr.line;
                if(!ditr.func.empty()) _ofs << " [" << tim::demangle(ditr.func) << "]";
                _ofs << "\n";
            }
        }

        _ofs << "\n" << std::flush;
    };

    auto _write = [&_write_impl, _verbose](const std::string& ofname, const auto& _data) {
        auto _ifs  = std::ifstream{ join("/", "/proc", process::get_id(), "maps") };
        auto _maps = std::stringstream{};
        while(_ifs)
        {
            std::string _line{};
            getline(_ifs, _line);
            _maps << _line << "\n";
        }
        auto _ofs = std::ofstream{};
        if(tim::filepath::open(_ofs, ofname))
        {
            if(_verbose >= 0)
                operation::file_output_message<binary::basic_line_info>{}(
                    ofname, std::string{ "causal_line_info" });
            _ofs << _maps.str();
            _write_impl(_ofs, _data);
        }
        else
        {
            throw ::omnitrace::exception<std::runtime_error>("Error opening " + ofname);
        }
    };

    _write(tim::settings::compose_output_filename(
               config::get_causal_output_filename() + "-included", "txt", _cfg),
           get_cached_line_info().first);
    _write(tim::settings::compose_output_filename(
               config::get_causal_output_filename() + "-excluded", "txt", _cfg),
           get_cached_line_info().second);
}

void
set_current_selection(unwind_addr_t _stack)
{
    if(experiment::is_active()) return;

    size_t _n = 0;
    for(auto itr : _stack)
    {
        auto& _pcs = latest_eligible_pc.at(_n);
        if(_pcs && is_eligible_address(itr))
        {
            _pcs->write(&itr);
            // increment after valid found -> first valid pc for call-stack
            ++_n;
        }
    }
}

selected_entry
sample_selection(size_t _nitr, size_t _wait_ns)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto&  _eligible_pcs = get_eligible_lines();
    size_t _n            = 0;

    auto _select_address = [&](auto& _address_vec) {
        // this isn't necessary bc of check before calling this lambda but
        // kept because of size() - 1 in distribution range
        if(OMNITRACE_UNLIKELY(_address_vec.empty()))
        {
            OMNITRACE_WARNING(0, "no addresses for sample selection...\n");
            return selected_entry{};
        }
        // randomly select an address
        auto _dist = std::uniform_int_distribution<size_t>{ 0, _address_vec.size() - 1 };
        auto _idx  = _dist(get_engine<selected_entry>());

        uintptr_t _addr        = _address_vec.at(_idx);
        uintptr_t _sym_addr    = 0;
        uintptr_t _lookup_addr = _addr;
        auto      _dl_info     = unwind::dlinfo::construct(_addr);

        if(get_causal_mode() == CausalMode::Function)
            _sym_addr = (_dl_info.symbol) ? _dl_info.symbol.address() : _addr;

        // lookup the PC line info at either the address or the symbol address
        auto linfo = get_line_info(_lookup_addr, false);

        // unlikely this will be empty but just in case
        if(OMNITRACE_UNLIKELY(linfo.empty())) return selected_entry{};

        // debugging for continuous integration
        if(OMNITRACE_UNLIKELY(config::get_is_continuous_integration() ||
                              config::get_debug()))
        {
            auto _location =
                (_dl_info.location)
                    ? filepath::realpath(std::string{ _dl_info.location.name }, nullptr,
                                         false)
                    : std::string{};
            for(const auto& itr : linfo)
            {
                if(OMNITRACE_UNLIKELY(config::get_debug()))
                {
                    OMNITRACE_WARNING(0, "[%s][%s][%s][%s] %s [%s][%s:%i][%s][%zu]\n",
                                      as_hex(_lookup_addr).c_str(), as_hex(_addr).c_str(),
                                      as_hex(_sym_addr).c_str(),
                                      (_location.empty()) ? "" : _location.data(),
                                      demangle(itr.second.func).c_str(),
                                      itr.first.pathname.c_str(), itr.second.file.c_str(),
                                      itr.second.line,
                                      itr.second.address.as_string().c_str(),
                                      itr.second.address.size());
                }
                TIMEMORY_REQUIRE(OMNITRACE_UNLIKELY(
                    itr.first.pathname.find(filepath::basename(_location)) !=
                    std::string::npos))
                    << "Error! pathname (" << itr.first.pathname
                    << ") does not contain dlinfo location "
                    << filepath::basename(_location) << " (" << _location << ")";
            }
        }

        auto& _linfo_v = (config::get_causal_mode() == CausalMode::Function)
                             ? linfo.front()
                             : linfo.back();
        return selected_entry{ _addr, _sym_addr, _linfo_v.second };
    };

    while(_n++ < _nitr)
    {
        auto _addresses = std::vector<uintptr_t>{};
        for(auto& aitr : latest_eligible_pc)
        {
            if(OMNITRACE_UNLIKELY(!aitr))
            {
                OMNITRACE_WARNING(0, "invalid ring buffer...\n");
                continue;
            }

            auto _naddrs = aitr->count();
            if(_naddrs == 0) continue;

            _addresses.reserve(_addresses.size() + _naddrs);
            for(size_t i = 0; i < _naddrs; ++i)
            {
                uintptr_t _addr = 0;
                if(aitr->read(&_addr) != nullptr)
                {
                    // comment out below bc symbol address lookup happens when filling
                    uintptr_t _sym_addr = 0;
                    if(get_causal_mode() == CausalMode::Function)
                    {
                        auto _dl_info = unwind::dlinfo::construct(_addr);
                        _sym_addr     = (_dl_info.symbol) ? _dl_info.symbol.address() : 0;
                    }

                    if(_eligible_pcs.count(_addr) > 0)
                        _addresses.emplace_back(_addr);
                    else if(_sym_addr > 0 && _eligible_pcs.count(_sym_addr) > 0)
                        _addresses.emplace_back(_sym_addr);
                }
            }

            if(!_addresses.empty()) return _select_address(_addresses);
        }

        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _wait_ns });
    }

    return selected_entry{};
}

std::deque<line_mapping_info_t>
get_line_info(uintptr_t _addr, bool include_discarded)
{
    auto _data          = std::deque<line_mapping_info_t>{};
    auto _get_line_info = [&](const auto& _info) {
        // search for exact matches first
        for(const auto& litr : _info)
        {
            if(!address_range_t{ litr.first.load_address, litr.first.last_address }
                    .contains(_addr))
                continue;

            auto _local_data = std::deque<line_mapping_info_t>{};
            for(const auto& ditr : litr.second)
            {
                if(ditr.ipaddr().contains(_addr))
                    _local_data.emplace_back(litr.first, ditr.get_basic());
            }

            if(!_local_data.empty())
            {
                // combine and only allow first match
                utility::combine(_data, _local_data);
                break;
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

void
mark_progress_point(std::string_view _name)
{
    if(!config::get_causal_end_to_end() && !experiment::is_active()) return;

    auto  _hash   = tim::add_hash_id(_name);
    auto& _data   = progress_bundles_t::instance(utility::get_thread_index());
    auto* _bundle = _data.construct(_hash);
    _bundle->push();
    _bundle->mark();
    _bundle->pop();
    _data.pop_back();
}

uint16_t
sample_virtual_speedup()
{
    if(speedup_dist.empty())
        return 0;
    else if(speedup_dist.size() == 1)
        return speedup_dist.front();
    else
    {
        struct virtual_speedup
        {};
        auto _dist =
            std::uniform_int_distribution<size_t>{ size_t{ 0 }, speedup_dist.size() - 1 };
        return speedup_dist.at(_dist(get_engine<virtual_speedup>()));
    }
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

    if(get_state() < State::Finalized)
    {
        OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
        auto _promise = std::make_shared<std::promise<void>>();
        std::thread{ perform_experiment_impl, _promise }.detach();
        _promise->get_future().wait_for(std::chrono::seconds{ 1 });
    }
}

void
finish_experimenting()
{
    if(perform_experiment_impl_completed)
    {
        perform_experiment_impl_completed->get_future().wait_for(
            std::chrono::seconds{ 5 });
        perform_experiment_impl_completed.reset();
    }
    sampling::post_process();
    experiment::save_experiments();
}
}  // namespace causal
}  // namespace omnitrace
