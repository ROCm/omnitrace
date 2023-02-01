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

#include "library/constraint.hpp"
#include "library/config.hpp"
#include "library/state.hpp"
#include "library/utility.hpp"

#include <timemory/units.hpp>
#include <timemory/utility/delimit.hpp>

#include <chrono>
#include <string>
#include <thread>

namespace omnitrace
{
namespace constraint
{
namespace
{
namespace units = ::tim::units;

using clock_type    = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double, std::nano>;

auto
sleep(uint64_t _n)
{
    std::this_thread::sleep_for(std::chrono::nanoseconds{ _n });
}
}  // namespace

stages::stages()
: init{ [](const spec&) { return get_state() < State::Finalized; } }
, wait{ [](const spec& _spec) {
    sleep(std::min<uint64_t>(1 * units::msec, _spec.delay * units::sec));
    return get_state() < State::Finalized;
} }
, start{ [](const spec&) { return get_state() < State::Finalized; } }
, collect{ [](const spec& _spec) {
    sleep(std::min<uint64_t>(1 * units::msec, _spec.duration * units::sec));
    return get_state() < State::Finalized;
} }
, stop{ [](const spec&) { return get_state() < State::Finalized; } }
{}

spec::spec(double _delay, double _dur, uint64_t _n, uint64_t _rep)
: delay{ _delay }
, duration{ _dur }
, count{ _n }
, repeat{ _rep }
{}

spec::spec(const std::string& _line)
{
    auto _delim = tim::delimit(_line, ":");
    if(!_delim.empty()) delay = utility::convert<double>(_delim.at(0));
    if(_delim.size() > 1) duration = utility::convert<double>(_delim.at(1));
    if(_delim.size() > 2) repeat = utility::convert<uint64_t>(_delim.at(2));
}

void
spec::operator()(const stages& _stages) const
{
    auto _n = repeat;
    if(_n < 1) _n = std::numeric_limits<uint64_t>::max();

    while(get_state() < State::Active)
        sleep(1 * units::usec);

    for(uint64_t i = 0; i < _n; ++i)
    {
        auto _spec = spec{ delay, duration, i, repeat };
        auto _wait = [_spec](const auto& _func, auto _dur) {
            auto _ret = true;
            auto _end = clock_type::now() + duration_type{ _dur * units::sec };
            while(clock_type::now() < _end && (_ret = _func(_spec)))
            {}
            return _ret;
        };

        if(_stages.init(_spec) && _wait(_stages.wait, _spec.delay) &&
           _stages.start(_spec) && _wait(_stages.collect, _spec.duration) &&
           _stages.stop(_spec))
        {}
        else
        {
            break;
        }
    }
}

std::vector<spec>
get_trace_specs()
{
    auto _v = std::vector<constraint::spec>{};

    {
        auto _delay_v = config::get_setting_value<double>("OMNITRACE_TRACE_DELAY").second;
        auto _duration_v =
            config::get_setting_value<double>("OMNITRACE_TRACE_DURATION").second;
        if(_delay_v > 0.0 || _duration_v > 0.0)
        {
            _v.emplace_back(_delay_v, _duration_v);
        }
    }

    {
        auto _periods_v =
            config::get_setting_value<std::string>("OMNITRACE_TRACE_PERIODS").second;
        if(!_periods_v.empty())
        {
            for(auto itr : tim::delimit(_periods_v, " ;\t\n"))
                _v.emplace_back(itr);
        }
    }

    return _v;
}

stages
get_trace_stages()
{
    auto _v = stages{};

    _v.init = [](const spec&) { return get_state() < State::Finalized; };
    _v.wait = [](const spec& _spec) {
        sleep(std::min<uint64_t>(1 * units::msec, _spec.delay * units::sec));
        return get_state() < State::Finalized;
    };
    _v.start   = [](const spec&) { return get_state() < State::Finalized; };
    _v.collect = [](const spec& _spec) {
        sleep(std::min<uint64_t>(1 * units::msec, _spec.duration * units::sec));
        return get_state() < State::Finalized;
    };
    _v.stop = [](const spec&) { return get_state() < State::Finalized; };

    return _v;
}
}  // namespace constraint
}  // namespace omnitrace
