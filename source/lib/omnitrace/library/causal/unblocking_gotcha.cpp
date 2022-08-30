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

#include "library/causal/unblocking_gotcha.hpp"
#include "library/causal/delay.hpp"
#include "library/config.hpp"

#include <csignal>
#include <cstdint>
#include <pthread.h>
#include <stdexcept>

namespace omnitrace
{
namespace causal
{
std::string
unblocking_gotcha::label()
{
    return "causal_unblocking_gotcha";
}

std::string
unblocking_gotcha::description()
{
    return "Handles executing all necessary pauses before the thread performs some "
           "blocking function";
}

void
unblocking_gotcha::configure()
{
    if(!config::get_use_critical_trace()) return;

    unblocking_gotcha_t::get_initializer() = []() {
        unblocking_gotcha_t::configure(
            comp::gotcha_config<0, int, pthread_mutex_t*>{ "pthread_mutex_unlock" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<1, int, pthread_rwlock_t*>{ "pthread_rwlock_unlock" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<2, int, pthread_spinlock_t*>{ "pthread_spin_unlock" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<3, int, pthread_barrier_t*>{ "pthread_barrier_wait" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<4, int, pthread_cond_t*>{ "pthread_cond_signal" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<5, int, pthread_cond_t*>{ "pthread_cond_broadcast" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<6, int, pthread_t, int>{ "pthread_kill" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<7, void, void*>{ "pthread_exit" });
    };
}

void
unblocking_gotcha::shutdown()
{
    unblocking_gotcha_t::disable();
}

void
unblocking_gotcha::start()
{
    causal::delay::process();
}

void
unblocking_gotcha::stop()
{
    causal::delay::credit();
}
}  // namespace causal
}  // namespace omnitrace
