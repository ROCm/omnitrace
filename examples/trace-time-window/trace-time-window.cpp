#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ratio>
#include <string>
#include <thread>

#define NOINLINE __attribute__((noinline))

NOINLINE size_t
inner();

NOINLINE size_t
outer_a();

NOINLINE size_t
outer_b();

NOINLINE size_t
outer_c();

NOINLINE size_t
outer_d();

NOINLINE size_t
outer_e();

int
main(int argc, char** argv)
{
    int ncalls = 10;
    if(argc > 1) ncalls = atol(argv[1]);

    std::string _name = argv[0];
    auto        _pos  = _name.find_last_of('/');
    if(_pos != std::string::npos) _name = _name.substr(_pos + 1);

    size_t nitr = 0;
    nitr += outer_a();
    nitr += outer_b();
    nitr += outer_c();
    nitr += outer_d();
    nitr += outer_e();

    printf("[%s] number of calls made = %zu\n", _name.c_str(), nitr);
}

size_t
inner(size_t _duration)
{
    using clock_type = std::chrono::high_resolution_clock;
    auto   _end      = clock_type::now() + std::chrono::milliseconds{ _duration };
    size_t nitr      = 0;
    while(clock_type::now() < _end)
    {
        ++nitr;
    }
    return nitr;
}

#define OUTER_FUNCTION(TAG)                                                              \
    size_t outer_##TAG() { return inner(200); }

OUTER_FUNCTION(a)
OUTER_FUNCTION(b)
OUTER_FUNCTION(c)
OUTER_FUNCTION(d)
OUTER_FUNCTION(e)
