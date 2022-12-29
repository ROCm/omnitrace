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

#include "library/causal/progress_point.hpp"
#include "library/causal/experiment.hpp"
#include "library/common.hpp"
#include "library/debug.hpp"
#include "library/thread_data.hpp"
#include "library/timemory.hpp"
#include "timemory/mpl/type_traits.hpp"

#include <bits/stdint-intn.h>
#include <timemory/hash/types.hpp>
#include <timemory/units.hpp>

using progress_map_t =
    std::unordered_map<tim::hash_value_t, omnitrace::causal::progress_point*>;

namespace omnitrace
{
namespace causal
{
namespace
{
using thread_data_t        = thread_data<identity<progress_map_t>>;
using progress_allocator_t = tim::data::ring_buffer_allocator<progress_point>;

progress_map_t&
get_progress_map(int64_t _tid)
{
    static auto& _v = thread_data_t::instances();
    return _v.at(_tid);
}

auto&
get_progress_allocator(int64_t _tid)
{
    static auto& _v = thread_data<progress_allocator_t>::instances(construct_on_init{});
    return _v.at(_tid);
}

}  // namespace

std::string
progress_point::label()
{
    return "progress_point";
}

std::string
progress_point::description()
{
    return "Tracks progress point latency and throughput for casual profiling";
}

std::unordered_map<tim::hash_value_t, progress_point>
get_progress_points()
{
    auto _data = std::unordered_map<tim::hash_value_t, progress_point>{};
    for(const auto& titr : thread_data_t::instances())
    {
        for(const auto& itr : titr)
        {
            if(itr.second)
            {
                auto& ditr = _data[itr.first];
                ditr += *itr.second;
                ditr.set_hash(itr.second->get_hash());
                ditr.set_laps(ditr.get_laps() + itr.second->get_laps());
                itr.second->set_laps(0);
                itr.second->set_value(0);
                itr.second->set_accum(0);
            }
        }
    }
    return _data;
}

void
progress_point::print(std::ostream& os) const
{
    os << tim::get_hash_identifier(m_hash) << " :: ";
    tim::operation::base_printer<progress_point>(os, *this);
}
}  // namespace causal
}  // namespace omnitrace

namespace tim
{
namespace operation
{
namespace causal = omnitrace::causal;

void
push_node<causal::progress_point>::operator()(type&        _obj, scope::config,
                                              hash_value_t _hash, int64_t _tid) const
{
    auto itr = causal::get_progress_map(_tid).emplace(_hash, nullptr);
    if(itr.second && !itr.first->second)
    {
        auto& _alloc = causal::get_progress_allocator(_tid);
        auto* _val   = _alloc->allocate(1);
        _alloc->construct(_val);
        _val->set_hash(_hash);
        itr.first->second = _val;
    }
    _obj.set_hash(_hash);
    _obj.set_iterator(itr.first->second);
}

void
pop_node<causal::progress_point>::operator()(type& _obj, int64_t) const
{
    auto* itr = _obj.get_iterator();
    if(itr && !(_obj.get_is_invalid() || _obj.get_is_running()))
    {
        *itr += _obj;
        itr->set_laps(itr->get_laps() + _obj.get_laps());
    }
}
}  // namespace operation
}  // namespace tim
