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
#include "common/defines.h"
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

#include <timemory/data/atomic_ring_buffer.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/log/logger.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/unwind/dlinfo.hpp>
#include <timemory/unwind/processed_entry.hpp>

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

auto&
get_engine()
{
    static thread_local auto _v = random_engine_t{ std::random_device{}() };
    return _v;
}

struct domain_address_range
{};

struct coarse_address_range
{
    TIMEMORY_DEFAULT_OBJECT(coarse_address_range)

    coarse_address_range& operator+=(std::pair<domain_address_range, uintptr_t>&& _v)
    {
        domain = address_range_t{ std::min(domain.low, _v.second),
                                  std::max(domain.high, _v.second) };
        return *this;
    }

    coarse_address_range& operator+=(
        std::pair<domain_address_range, address_range_t>&& _v)
    {
        domain = address_range_t{ std::min(domain.low, _v.second.low),
                                  std::max(domain.high, _v.second.high) };

        return *this;
    }

    coarse_address_range& operator+=(uintptr_t _v)
    {
        *this += std::make_pair(domain_address_range{}, _v);

        for(auto&& itr : m_fine_ranges)
            if(itr.contains(_v)) return *this;

        m_fine_ranges.emplace(address_range_t{ _v });
        return *this;
    }

    coarse_address_range& operator+=(address_range_t _v)
    {
        *this += std::make_pair(domain_address_range{}, _v);

        for(auto&& itr : m_fine_ranges)
            if(itr.contains(_v)) return *this;

        m_fine_ranges.emplace(_v);
        return *this;
    }

    template <typename Tp>
    bool contains(Tp&& _v) const
    {
        using type = concepts::unqualified_type_t<Tp>;
        static_assert(std::is_integral<type>::value ||
                          std::is_same<type, address_range_t>::value,
                      "Error! operator+= supports only integrals or address_ranges");

        if(!domain.contains(_v)) return false;
        return std::any_of(m_fine_ranges.begin(), m_fine_ranges.end(),
                           [_v](auto&& itr) { return itr.contains(_v); });
    }

    address_range_t domain = {};

    auto    size() const { return m_fine_ranges.size(); }
    int64_t range_size() const { return domain.high - domain.low + 1; }

private:
    std::set<address_range_t> m_fine_ranges = {};
};

coarse_address_range&
get_eligible_address_ranges()
{
    static auto _v = coarse_address_range{};
    return _v;
}

using line_info_map_t = code_object::line_info_map<code_object::ext_line_info>;

auto
get_filters()
{
    using sf      = code_object::scope_filter;
    auto _filters = std::vector<code_object::scope_filter>{};

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
        // exclude _GLOBAL and cold symbols
        // symbols starting with leading underscore are generally system functions
        // and cold functions are functions (or parts of functions) which the compiler
        // or user designated as nowhere close to the hotpath
        _filters.emplace_back(
            sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER, "(^_|\\.cold(|\\.[0-9]+)$)" });

        // exclude lambdas, function call operators, and STL implementation functions
        _filters.emplace_back(sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER,
                                  "::(<|\\{)lambda\\(|operator\\(\\)|::_M" });

        _filters.emplace_back(sf{ sf::FILTER_EXCLUDE, sf::FUNCTION_FILTER, "^_" });
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

    auto _fileline_end_converter = [](const std::string& _v) { return _v + "$"; };

    // include handling
    {
        auto _binary_include   = get_regex_or(config::get_causal_binary_scope(), "");
        auto _source_include   = get_regex_or(config::get_causal_source_scope(), "");
        auto _function_include = get_regex_or(config::get_causal_function_scope(), "");
        auto _fileline_include = get_regex_or(config::get_causal_fileline_scope(),
                                              _fileline_end_converter, "");

        auto        _current_include = std::make_tuple(_binary_include, _source_include,
                                                _function_include, _fileline_include);
        static auto _former_include  = decltype(_current_include){};

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
            if(!_fileline_include.empty())
                OMNITRACE_VERBOSE(0, "[causal] fileline scope   : %s\n",
                                  _fileline_include.c_str());
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

        if(!_fileline_include.empty())
            _filters.emplace_back(
                sf{ sf::FILTER_INCLUDE, sf::FILELINE_FILTER, _fileline_include });
    }

    // exclude handling
    {
        auto _binary_exclude   = get_regex_or(config::get_causal_binary_exclude(), "");
        auto _source_exclude   = get_regex_or(config::get_causal_source_exclude(), "");
        auto _function_exclude = get_regex_or(config::get_causal_function_exclude(), "");
        auto _fileline_exclude = get_regex_or(config::get_causal_fileline_exclude(),
                                              _fileline_end_converter, "");

        auto        _current_exclude = std::make_tuple(_binary_exclude, _source_exclude,
                                                _function_exclude, _fileline_exclude);
        static auto _former_exclude  = decltype(_current_exclude){};

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
            if(!_fileline_exclude.empty())
                OMNITRACE_VERBOSE(0, "[causal] fileline exclude : %s\n",
                                  _fileline_exclude.c_str());
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

        if(!_fileline_exclude.empty())
            _filters.emplace_back(
                sf{ sf::FILTER_EXCLUDE, sf::FILELINE_FILTER, _fileline_exclude });
    }

    return _filters;
}

std::pair<line_info_map_t, line_info_map_t>&
get_cached_line_info()
{
    static auto _v = []() {
        auto _discarded = line_info_map_t{};
        auto _requested = code_object::get_ext_line_info(get_filters(), &_discarded);
        return std::make_pair(_requested, _discarded);
    }();
    return _v;
}

auto
compute_eligible_lines()
{
    const auto& _line_info = get_cached_line_info().first;
    auto _v = std::unordered_map<uintptr_t, std::deque<code_object::basic::line_info>>{};

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

    using sf = code_object::scope_filter;

    bool _use_custom_filters = false;
    for(const auto& itr : _filters)
    {
        if(itr.mode == sf::FILTER_INCLUDE &&
           (itr.scope == sf::FUNCTION_FILTER || itr.scope == sf::FILELINE_FILTER))
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
                if(!satisfies_filter(sf::FILELINE_FILTER, ditr.file) &&
                   !satisfies_filter(sf::FILELINE_FILTER,
                                     JOIN(':', ditr.file, ditr.line)))
                    continue;

                // check both mangled and demangled function name
                if(!satisfies_filter(sf::FUNCTION_FILTER, demangle(ditr.func))) continue;
            }

            // map the instruction pointer address to the line info
            auto _ip = ditr.address + litr.first.load_address;

            _add_line_info(_ip, ditr);
            if(!_valid)
                _valid = (_eligible_ar += std::make_pair(domain_address_range{}, _range),
                          true);

            _eligible_ar += _ip;
        }
    }

    int64_t _range_size = _eligible_ar.range_size();

    OMNITRACE_VERBOSE(0,
                      "[causal] eligible addresses: %zu, eligible address ranges: %zu, "
                      "total range: %zu, [%s]\n",
                      _v.size(), _eligible_ar.size(), _range_size,
                      _eligible_ar.domain.as_string().c_str());

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
    const auto& _thr_info = thread_info::init(true);
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    OMNITRACE_CONDITIONAL_THROW(!_thr_info->is_offset,
                                "Error! causal profiling thread should be offset");

    (void) get_eligible_lines();

    // notify that thread has started
    if(_started) _started->set_value();

    auto _cfg         = settings::compose_filename_config{};
    _cfg.subdirectory = "causal";
    save_line_info(_cfg);

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

hash_value_t
selected_entry::hash() const
{
    return tim::get_combined_hash_id(tim::hash_value_t{ address }, info.hash());
}

template <typename ArchiveT>
void
selected_entry::serialize(ArchiveT& ar, const unsigned int)
{
    using ::tim::cereal::make_nvp;
    ar(make_nvp("address", address), make_nvp("symbol_address", symbol_address),
       make_nvp("info", info));
}

template void
selected_entry::serialize<cereal::JSONInputArchive>(cereal::JSONInputArchive&,
                                                    const unsigned int);

template void
selected_entry::serialize<cereal::MinimalJSONOutputArchive>(
    cereal::MinimalJSONOutputArchive&, const unsigned int);

template void
selected_entry::serialize<cereal::PrettyJSONOutputArchive>(
    cereal::PrettyJSONOutputArchive&, const unsigned int);

bool
is_eligible_address(uintptr_t _v)
{
    return get_eligible_address_ranges().contains(_v);
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
                auto _addr     = ditr.address;
                auto _addr_off = ditr.address + itr.first.load_address;
                _ofs << "    " << as_hex(_addr_off) << " [" << as_hex(_addr)
                     << "] :: " << ditr.file << ":" << ditr.line;
                if(!ditr.func.empty()) _ofs << " [" << tim::demangle(ditr.func) << "]";
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
            throw ::omnitrace::exception<std::runtime_error>("Error opening " + ofname);
        }
    };

    _write(tim::settings::compose_output_filename("line-info-included", "txt", _cfg),
           get_cached_line_info().first);
    _write(tim::settings::compose_output_filename("line-info-discarded", "txt", _cfg),
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

    auto&  _eligible_pcs = get_eligible_lines();
    size_t _n            = 0;

    auto _select_address = [&](auto& _addresses) {
        // this isn't necessary bc of check before calling this lambda but
        // kept because of size() - 1 in distribution range
        if(OMNITRACE_UNLIKELY(_addresses.empty()))
        {
            OMNITRACE_WARNING(0, "no addresses for sample selection...\n");
            return selected_entry{};
        }
        // randomly select an address
        auto _dist = std::uniform_int_distribution<size_t>{ 0, _addresses.size() - 1 };
        auto _idx  = _dist(get_engine());

        uintptr_t _addr        = _addresses.at(_idx);
        uintptr_t _sym_addr    = 0;
        uintptr_t _lookup_addr = _addr;
        auto      _dl_info     = unwind::dlinfo::construct(_addr);
        auto      _location =
            (_dl_info.location) ? _dl_info.location.name : std::string_view{};

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
                    << ") does not contain dlinfo location ("
                    << filepath::basename(_location) << ")";
            }
        }

        return (config::get_causal_mode() == CausalMode::Function)
                   ? selected_entry{ _addr, _sym_addr, linfo.front().second }
                   : selected_entry{ _addr, _sym_addr, linfo.back().second };
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
                auto _ip = litr.first.load_address + ditr.address;
                if(_ip.contains(_addr))
                {
                    _local_data.emplace_back(litr.first, ditr.get_basic());
                }
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
    auto _dist =
        std::uniform_int_distribution<size_t>{ size_t{ 0 }, speedup_dist.size() - 1 };
    return speedup_dist.at(_dist(get_engine()));
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
        _promise->get_future().wait_for(std::chrono::seconds{ 2 });
    }
}
}  // namespace causal
}  // namespace omnitrace
