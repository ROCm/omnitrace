
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

std::atomic<long> total{ 0 };
long
fib(long n) __attribute__((noinline));
void
run(size_t nitr, long) __attribute__((noinline));

long
fib(long n)
{
    return (n < 2) ? n : fib(n - 1) + fib(n - 2);
}

void
run(size_t nitr, long n)
{
    long local = 0;
    for(size_t i = 0; i < nitr; ++i)
        local += fib(n);
    total += local;
}

int
main(int argc, char** argv)
{
    std::string _name = argv[0];
    auto        _pos  = _name.find_last_of('/');
    if(_pos != std::string::npos) _name = _name.substr(_pos + 1);

    size_t nthread = std::min<size_t>(16, std::thread::hardware_concurrency());
    size_t nitr    = 50000;
    long   nfib    = 10;

    if(argc > 1) nfib = atol(argv[1]);
    if(argc > 2) nthread = atol(argv[2]);
    if(argc > 3) nitr = atol(argv[3]);

    printf("[%s] Threads: %zu\n[%s] Iterations: %zu\n[%s] fibonacci(%li)...\n",
           _name.c_str(), nthread, _name.c_str(), nitr, _name.c_str(), nfib);

    std::vector<std::thread> threads{};
    for(size_t i = 0; i < nthread; ++i)
    {
        size_t _nitr = ((i % 2) == 1) ? (nitr - (0.1 * nitr)) : (nitr + (0.1 * nitr));
        _nitr        = std::max<size_t>(_nitr, 1);
        threads.emplace_back(&run, _nitr, nfib);
    }

    auto _nitr = std::max<size_t>(nitr - 0.25 * nitr, 1);
    run(_nitr, nfib - 0.1 * nfib);
    for(auto& itr : threads)
        itr.join();

    printf("[%s] fibonacci(%li) x %lu = %li\n", _name.c_str(), nfib, nthread,
           total.load());

    return 0;
}
