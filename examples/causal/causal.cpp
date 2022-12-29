
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/times.h>
#include <thread>
#include <unistd.h>

#define STR2(x) #x
#define STR(x)  STR2(x)
#define LABEL   __FILE__ ":" STR(__LINE__)

#if defined(USE_OMNI) && USE_OMNI > 0
#    include <omnitrace/user.h>
#    define CAUSAL_PROGRESS omnitrace_user_progress(LABEL)
#elif defined(USE_COZ) && USE_COZ > 0
#    include <coz.h>
#    define CAUSAL_PROGRESS COZ_PROGRESS_NAMED(LABEL)
#else
#    define CAUSAL_PROGRESS
#endif

using mutex_t     = std::timed_mutex;
using auto_lock_t = std::unique_lock<mutex_t>;
using clock_type  = std::chrono::high_resolution_clock;
using nanosec     = std::chrono::nanoseconds;

namespace
{
std::chrono::duration<double, std::milli> t_ms;
std::chrono::duration<double, std::milli> slow_ms;
std::chrono::duration<double, std::milli> fast_ms;

template <typename... Args>
inline void
consume_variables(Args&&...)
{}
}  // namespace

bool
rng_func_impl(int64_t n, uint64_t rseed);

bool
cpu_func_impl(int64_t n, int nloop);

int64_t
rng_slow_func(int64_t n, uint64_t rseed);

int64_t
rng_fast_func(int64_t n, uint64_t rseed);

void
cpu_slow_func(int64_t n, int nloop);

void
cpu_fast_func(int64_t n, int nloop);

#if USE_CPU > 0
#    define CPU_SLOW_FUNC(...) cpu_slow_func(__VA_ARGS__)
#    define CPU_FAST_FUNC(...) cpu_fast_func(__VA_ARGS__)
#else
#    define CPU_SLOW_FUNC(...) consume_variables(__VA_ARGS__)
#    define CPU_FAST_FUNC(...) consume_variables(__VA_ARGS__)
#endif

#if USE_RNG > 0
#    define RNG_SLOW_FUNC(...) rng_slow_func(__VA_ARGS__)
#    define RNG_FAST_FUNC(...) rng_fast_func(__VA_ARGS__)
#else
#    define RNG_SLOW_FUNC(...) consume_variables(__VA_ARGS__)
#    define RNG_FAST_FUNC(...) consume_variables(__VA_ARGS__)
#endif

int
main(int argc, char** argv)
{
    uint64_t rseed    = std::random_device{}();
    int      nitr     = 150;
    double   frac     = 70;
    int64_t  slow_val = 100000000L;

    if(argc > 1) frac = std::stod(argv[1]);
    if(argc > 2) nitr = std::stoi(argv[2]);
    if(argc > 3) rseed = std::stoul(argv[3]);
    if(argc > 4) slow_val = std::stol(argv[4]);

    int64_t fast_val = (frac / 100.0) * slow_val;
    double  rfrac    = (fast_val / static_cast<double>(slow_val));
    if(argc > 5) fast_val = std::stol(argv[5]);

    printf("\nIterations: %i, fraction: %6.2f, random seed: %lu :: slow = %zu, "
           "fast = %zu, expected ratio = %6.2f\n",
           nitr, frac, rseed, slow_val, fast_val, rfrac * 100.0);

    auto _t = clock_type::now();
    for(int i = 0; i < nitr; ++i)
    {
        if(i == 0 || i + 1 == nitr || i % (nitr / 5) == 0)
            printf("executing iteration: %i\n", i);
        auto&& _slow_func = [](auto _nsec, auto _seed, auto _nloop) {
            CPU_SLOW_FUNC(_nsec, _nloop);
            RNG_SLOW_FUNC(_nsec / 5, _seed);
        };
        auto&& _fast_func = [](auto _nsec, auto _seed, auto _nloop) {
            CPU_FAST_FUNC(_nsec, _nloop);
            RNG_FAST_FUNC(_nsec / 5, _seed);
        };
        auto _threads = std::array<std::thread, 2>{
            std::thread{ std::move(_slow_func), slow_val, rseed, 10000 },
            std::thread{ std::move(_fast_func), fast_val, rseed, 10000 }
        };
        //
        for(auto& itr : _threads)
            itr.join();

        CAUSAL_PROGRESS;
    }
    t_ms += clock_type::now() - _t;
    auto rms = (fast_ms.count() / slow_ms.count());
    printf("slow_func() took %8.3f ms\n", slow_ms.count());
    printf("fast_func() took %8.3f ms\n", fast_ms.count());
    printf("total is %12.3f ms\n", t_ms.count());
    printf("ratio is %12.3f %s\n", 100.0 * rms, "%");
    printf("rdiff is %12.3f %s\n", 100.0 * (rms - rfrac), "%");
}

//
//  This implementation works well for Omnitrace
//  while COZ makes poor predictions
//
int64_t
rng_slow_func(int64_t n, uint64_t rseed)
{
    auto _t = clock_type::now();
    // clang-format off
    while(!rng_func_impl(n, rseed)) {}
    // clang-format on
    slow_ms += clock_type::now() - _t;
    return slow_ms.count();
}

int64_t
rng_fast_func(int64_t n, uint64_t rseed)
{
    auto _t = clock_type::now();
    // clang-format off
    while(!rng_func_impl(n, rseed)) {}
    // clang-format on
    fast_ms += clock_type::now() - _t;
    return fast_ms.count();
}

//
//  This implementation works well for COZ
//  while Omnitrace makes poor predictions
//
void
cpu_slow_func(int64_t n, int nloop)
{
    auto _t = clock_type::now();
    // clang-format off
    while(!cpu_func_impl(n, nloop)) {}
    // clang-format on
    slow_ms += clock_type::now() - _t;
}

void
cpu_fast_func(int64_t n, int nloop)
{
    auto _t = clock_type::now();
    // clang-format off
    while(!cpu_func_impl(n, nloop)) {}
    // clang-format on
    fast_ms += clock_type::now() - _t;
}
