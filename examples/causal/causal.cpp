/*
 * Copyright (c) 2015, Charlie Curtsinger and Emery Berger,
 *                     University of Massachusetts Amherst
 * This file is part of the Coz project. See LICENSE.md file at the top-level
 * directory of this distribution and at http://github.com/plasma-umass/coz.
 */

#include <chrono>
#include <cmath>
#include <coz.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <sys/times.h>
#include <thread>
#include <unistd.h>

using mutex_t     = std::timed_mutex;
using auto_lock_t = std::unique_lock<mutex_t>;
using clock_type  = std::chrono::high_resolution_clock;
using nanosec     = std::chrono::nanoseconds;

mutex_t mtx{};

std::chrono::duration<double, std::milli> t_ms;
std::chrono::duration<double, std::milli> slow_ms;
std::chrono::duration<double, std::milli> fast_ms;

inline __attribute__((always_inline)) int64_t
clock_tick() noexcept
{
    static int64_t _val = ::sysconf(_SC_CLK_TCK);
    return _val;
}

template <typename Precision, typename Ret = int64_t>
inline __attribute__((always_inline)) Ret
clock_tick() noexcept
{
    return static_cast<Ret>(Precision::den) / static_cast<Ret>(clock_tick());
}

template <typename Tp = int64_t, typename Precision = std::nano>
inline __attribute__((always_inline)) Tp
get_clock_now(clockid_t clock_id) noexcept
{
    constexpr Tp    factor = Precision::den / static_cast<Tp>(std::nano::den);
    struct timespec ts;
    clock_gettime(clock_id, &ts);
    return (ts.tv_sec * std::nano::den + ts.tv_nsec) * factor;
}

template <typename Tp = int64_t, typename Precision = std::nano>
inline __attribute__((always_inline)) Tp
get_clock_cpu_now() noexcept
{
    return get_clock_now<Tp, Precision>(CLOCK_THREAD_CPUTIME_ID);
}

__attribute__((noinline)) void
slow_func(int64_t n);

__attribute__((noinline)) void
fast_func(int64_t n);

void
slow_func(int64_t n)
{
    auto _t       = clock_type::now();
    auto _cpu_now = get_clock_cpu_now();
    auto _cpu_end = _cpu_now + n;
    while(get_clock_cpu_now() < _cpu_end)
    {
        for(volatile int i = 0; i < 1000; ++i)
        {}
    }
    slow_ms += clock_type::now() - _t;
}

void
fast_func(int64_t n)
{
    auto _t       = clock_type::now();
    auto _cpu_now = get_clock_cpu_now();
    auto _cpu_end = _cpu_now + n;
    while(get_clock_cpu_now() < _cpu_end)
    {
        for(volatile int i = 0; i < 1000; ++i)
        {}
    }
    fast_ms += clock_type::now() - _t;
}

int
main(int argc, char** argv)
{
    int     nitr   = 20;
    double  frac   = 25;
    int64_t t_nsec = 100000000L;

    if(argc > 1) nitr = atoi(argv[1]);
    if(argc > 2) frac = fmod(atof(argv[2]), 100.);
    if(argc > 3) t_nsec = atoll(argv[3]);

    int64_t slow_nsec = t_nsec;
    int64_t fast_nsec = t_nsec - ((frac / 100.0) * t_nsec);

    if(argc > 4) fast_nsec = atoll(argv[4]);

    std::unique_lock<mutex_t> _lk{ mtx };

    printf("Iterations: %i :: slow = %zu, fast = %zu\n", nitr, slow_nsec, fast_nsec);

    std::string _name  = "iteration-" + std::to_string(static_cast<int64_t>(frac));
    std::string _phase = "phase-" + std::to_string(static_cast<int64_t>(frac));

    auto _t = clock_type::now();
    for(int i = 0; i < nitr; ++i)
    {
        printf("executing iteration: %i\n", i);
        auto _threads = std::array<std::optional<std::thread>, 2>{
            std::thread{ slow_func, slow_nsec }, std::thread{ fast_func, fast_nsec }
        };

        for(auto& itr : _threads)
            itr->join();

        for(auto& itr : _threads)
            itr.reset();

        // COZ_END(_phase.c_str());
        COZ_PROGRESS_NAMED(_name.c_str());
    }
    t_ms += clock_type::now() - _t;

    // printf("\n");
    printf("slow_func() took %8.3f ms\n", slow_ms.count());
    printf("fast_func() took %8.3f ms\n", fast_ms.count());
    printf("total is %8.3f ms\n", t_ms.count());
    printf("ratio is %8.3f %s\n", (1.0 - (fast_ms.count() / slow_ms.count())) * 100, "%");
}
