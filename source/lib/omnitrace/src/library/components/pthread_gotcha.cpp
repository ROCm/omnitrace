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

#include "library/components/pthread_gotcha.hpp"
#include "library/components/omnitrace.hpp"
#include "library/components/roctracer.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/utility/types.hpp>

#include <ostream>
#include <pthread.h>

namespace omnitrace
{
namespace sampling
{
std::set<int>
setup();
std::set<int>
shutdown();
}  // namespace sampling

namespace mpl = tim::mpl;

using bundle_t  = tim::lightweight_tuple<comp::wall_clock, comp::roctracer_data>;
using wall_pw_t = mpl::piecewise_select<comp::wall_clock>;  // only wall-clock
using main_pw_t = mpl::piecewise_ignore<comp::wall_clock>;  // exclude wall-clock
using omni_pw_t = mpl::piecewise_select<>;

namespace
{
std::map<int64_t, std::shared_ptr<bundle_t>> bundles = {};
std::mutex                                   bundles_mutex{};

inline void
start_bundle(bundle_t& _bundle)
{
    if(comp::roctracer::is_setup())
    {
        _bundle.push(main_pw_t{});
        _bundle.start();
    }
    else
    {
        _bundle.push(omni_pw_t{});
        _bundle.start(omni_pw_t{});
    }
}

inline void
stop_bundle(bundle_t& _bundle, int64_t _tid)
{
    _bundle.stop(wall_pw_t{});  // stop wall-clock so we can get the value
    // update roctracer_data
    _bundle.store(std::plus<double>{},
                  _bundle.get<comp::wall_clock>()->get() * units::sec);
    // stop all other components including roctracer_data after update
    _bundle.stop(main_pw_t{});
    // exclude popping wall-clock
    _bundle.pop(main_pw_t{}, _tid);
}

auto
get_thread_index()
{
    static std::atomic<int64_t> _c{ 0 };
    static thread_local int64_t _v = _c++;
    return _v;
}

auto&
get_sampling_on_child_threads_history(int64_t _idx = get_thread_index())
{
    static auto _v = std::array<std::vector<bool>, OMNITRACE_MAX_THREADS>{};
    return _v.at(_idx);
}
}  // namespace

pthread_gotcha::wrapper::wrapper(routine_t _routine, void* _arg, bool _enable_sampling,
                                 int64_t _parent, promise_t* _p)
: m_enable_sampling{ _enable_sampling }
, m_parent_tid{ _parent }
, m_routine{ _routine }
, m_arg{ _arg }
, m_promise{ _p }
{}

void*
pthread_gotcha::wrapper::operator()() const
{
    std::shared_ptr<bundle_t> _bundle{};
    std::set<int>             _signals{};
    auto                      _active      = (get_state() == omnitrace::State::Active);
    int64_t                   _tid         = -1;
    auto                      _is_sampling = false;
    auto                      _dtor        = scope::destructor{ [&]() {
        if(_is_sampling)
        {
            sampling::block_signals(_signals);
            sampling::shutdown();
        }

        if(_bundle)
        {
            std::unique_lock<std::mutex> _lk{ bundles_mutex };
            stop_bundle(*_bundle, _tid);
            _bundle.reset();
            bundles.erase(_tid);
        }
    } };

    if(_active) get_cpu_cid_stack(threading::get_id(), m_parent_tid);

    if(m_enable_sampling && _active)
    {
        _tid = threading::get_id();
        threading::set_thread_name(TIMEMORY_JOIN(" ", "Thread", _tid).c_str());
        // initialize thread-local statics
        (void) tim::get_unw_backtrace<12, 1, false>();
        {
            std::unique_lock<std::mutex> _lk{ bundles_mutex };
            if(comp::roctracer::is_setup())
                _bundle =
                    bundles.emplace(_tid, std::make_shared<bundle_t>("start_thread"))
                        .first->second;
        }
        if(_bundle) start_bundle(*_bundle);
        _is_sampling = true;
        push_enable_sampling_on_child_threads(false);
        _signals = sampling::setup();
        pop_enable_sampling_on_child_threads();
        sampling::unblock_signals();
    }

    if(m_promise) m_promise->set_value();

    // execute the original function
    return m_routine(m_arg);
}

void*
pthread_gotcha::wrapper::wrap(void* _arg)
{
    if(_arg == nullptr) return nullptr;

    // convert the argument
    wrapper* _wrapper = static_cast<wrapper*>(_arg);

    // execute the original function
    return (*_wrapper)();
}

void
pthread_gotcha::configure()
{
    pthread_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(pthread_gotcha_t, 0, pthread_create);
    };
}

void
pthread_gotcha::shutdown()
{
    std::unique_lock<std::mutex> _lk{ bundles_mutex };
    unsigned long                _ndangling = 0;
    for(auto itr : bundles)
    {
        if(itr.second)
        {
            stop_bundle(*itr.second, itr.first);
            ++_ndangling;
        }
        itr.second.reset();
    }

    OMNITRACE_CONDITIONAL_PRINT(
        (get_verbose() > 0 || get_debug()) && _ndangling > 0,
        "pthread_gotcha::shutdown() cleaned up %lu dangling bundles\n", _ndangling);
    bundles.clear();
}

bool
pthread_gotcha::sampling_enabled_on_child_threads()
{
    return sampling_on_child_threads();
}

bool
pthread_gotcha::push_enable_sampling_on_child_threads(bool _v)
{
    auto& _hist = get_sampling_on_child_threads_history();
    bool  _last = sampling_on_child_threads();
    _hist.emplace_back(_last);
    sampling_on_child_threads() = _v;
    return _last;
}

bool
pthread_gotcha::pop_enable_sampling_on_child_threads()
{
    auto& _hist = get_sampling_on_child_threads_history();
    if(!_hist.empty())
    {
        bool _restored = _hist.back();
        _hist.pop_back();
        sampling_on_child_threads() = _restored;
    }
    return sampling_on_child_threads();
}

void
pthread_gotcha::set_sampling_on_all_future_threads(bool _v)
{
    for(size_t i = 0; i < max_supported_threads; ++i)
        get_sampling_on_child_threads_history(i).emplace_back(_v);
}

bool&
pthread_gotcha::sampling_on_child_threads()
{
    static thread_local bool _v = get_sampling_on_child_threads_history().empty()
                                      ? false
                                      : get_sampling_on_child_threads_history().back();
    return _v;
}

// pthread_create
int
pthread_gotcha::operator()(pthread_t* thread, const pthread_attr_t* attr,
                           void* (*start_routine)(void*), void*     arg) const
{
    bundle_t _bundle{ "pthread_create" };
    auto     _enable_sampling = sampling_enabled_on_child_threads();
    auto     _active          = (get_state() == omnitrace::State::Active);
    int64_t  _tid             = (_active) ? threading::get_id() : 0;

    // ensure that cpu cid stack exists on the parent thread if active
    if(_active) get_cpu_cid_stack();

    if(!get_use_sampling() || !_enable_sampling)
    {
        // if(!get_use_sampling()) start_bundle(_bundle);
        auto* _obj = new wrapper(start_routine, arg, _enable_sampling, _tid, nullptr);
        // create the thread
        auto _ret =
            pthread_create(thread, attr, &wrapper::wrap, static_cast<void*>(_obj));
        // if(!get_use_sampling()) stop_bundle(_bundle, threading::get_id());
        return _ret;
    }

    // block the signals in entire process
    OMNITRACE_DEBUG("blocking signals...\n");
    tim::sampling::block_signals({ SIGALRM, SIGPROF },
                                 tim::sampling::sigmask_scope::process);

    start_bundle(_bundle);

    // promise set by thread when signal handler is configured
    auto  _promise = std::promise<void>{};
    auto  _fut     = _promise.get_future();
    auto* _wrap    = new wrapper(start_routine, arg, _enable_sampling, _tid, &_promise);

    // create the thread
    auto _ret = pthread_create(thread, attr, &wrapper::wrap, static_cast<void*>(_wrap));

    // wait for thread to set promise
    OMNITRACE_DEBUG("waiting for child to signal it is setup...\n");
    _fut.wait();

    stop_bundle(_bundle, threading::get_id());

    // unblock the signals in the entire process
    OMNITRACE_DEBUG("unblocking signals...\n");
    tim::sampling::unblock_signals({ SIGALRM, SIGPROF },
                                   tim::sampling::sigmask_scope::process);

    OMNITRACE_DEBUG("returning success...\n");
    return _ret;
}

}  // namespace omnitrace
