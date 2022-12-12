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

#include "library/causal/blocking_gotcha.hpp"
#include "library/causal/delay.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/runtime.hpp"
#include "library/state.hpp"

#include <timemory/components/macros.hpp>
#include <timemory/hash/types.hpp>

#include <csignal>
#include <cstdint>
#include <pthread.h>
#include <stdexcept>

namespace omnitrace
{
namespace causal
{
std::string
blocking_gotcha::label()
{
    return "causal_blocking_gotcha";
}

std::string
blocking_gotcha::description()
{
    return "Handles executing all necessary pauses before the thread performs some "
           "blocking function";
}

void
blocking_gotcha::preinit()
{
    configure();
}

void
blocking_gotcha::configure()
{
    blocking_gotcha_t::get_initializer() = []() {
        if(!config::get_use_causal()) return;

        blocking_gotcha_t::configure(
            comp::gotcha_config<0, int, pthread_mutex_t*>{ "pthread_mutex_lock" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<1, int, pthread_mutex_t*>{ "pthread_mutex_trylock" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<2, int, pthread_rwlock_t*>{ "pthread_rwlock_wrlock" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<3, int, pthread_rwlock_t*>{ "pthread_rwlock_trywrlock" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<4, int, pthread_spinlock_t*>{ "pthread_spin_lock" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<5, int, pthread_spinlock_t*>{ "pthread_spin_trylock" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<6, int, pthread_t, void**>{ "pthread_join" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<7, int, pthread_cond_t*, pthread_mutex_t*>{
                "pthread_cond_wait" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<8, int, pthread_cond_t*, pthread_mutex_t*,
                                const struct timespec*>{ "pthread_cond_timedwait" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<9, int, const sigset_t*, int*>{ "sigwait" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<10, int, const sigset_t*, int*, siginfo_t*>{
                "sigwaitinfo" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<11, int, const sigset_t*, int*, siginfo_t*,
                                const struct timespec*>{ "sigtimedwait" });

        blocking_gotcha_t::configure(
            comp::gotcha_config<12, int, const sigset_t*>{ "sigsuspend" });
    };
}

void
blocking_gotcha::shutdown()
{
    blocking_gotcha_t::disable();
}

void
blocking_gotcha::start()
{
    if(get_state() == ::omnitrace::State::Active &&
       get_thread_state() == ::omnitrace::ThreadState::Enabled)
        causal::delay::process();
}

void
blocking_gotcha::stop()
{
    if(get_state() == ::omnitrace::State::Active &&
       get_thread_state() == ::omnitrace::ThreadState::Enabled)
        causal::delay::process();
}

void
blocking_gotcha::set_data(const comp::gotcha_data& _data)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    auto   _hash  = tim::add_hash_id(_data.tool_id);
    auto&& _ident = tim::get_hash_identifier(_hash);
    if(_ident != _data.tool_id)
        throw std::runtime_error(JOIN("", "Error! resolving hash for \"", _data.tool_id,
                                      "\" (", _hash, ") returns ", _ident.c_str()));
#if defined(OMNITRACE_CI)
    OMNITRACE_VERBOSE_F(3, "data set for '%s'...\n", _data.tool_id.c_str());
#endif
}
}  // namespace causal
}  // namespace omnitrace

TIMEMORY_INVOKE_PREINIT(omnitrace::causal::blocking_gotcha)
