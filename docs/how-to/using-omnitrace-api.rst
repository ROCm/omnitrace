.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Using the Omnitrace API
****************************************************

The following example shows how a program can use the Omnitrace API for run-time analysis.

Omnitrace user API example program
========================================

You can use the Omnitrace API to define custom regions to profile and trace.
The following C++ program demonstrates this technique by calling several functions from the
Omnitrace API, such as ``omnitrace_user_push_region`` and
``omnitrace_user_stop_thread_trace``.

.. note::

   By default, when Omnitrace detects any ``omnitrace_user_start_*`` or
   ``omnitrace_user_stop_*`` function, instrumentation
   is disabled at start up, which means ``omnitrace_user_stop_trace()`` is not
   required at the beginning of ``main``. This behavior
   can be manually controlled by using the ``OMNITRACE_INIT_ENABLED`` environment variable.
   User-defined regions are always
   recorded, regardless of whether ``omnitrace_user_start_*`` or
   ``omnitrace_user_stop_*`` has been called.

.. code-block:: shell

   #include <omnitrace/categories.h>
   #include <omnitrace/types.h>
   #include <omnitrace/user.h>

   #include <atomic>
   #include <cassert>
   #include <cerrno>
   #include <cstdio>
   #include <cstdlib>
   #include <cstring>
   #include <sstream>
   #include <thread>
   #include <vector>

   std::atomic<long> total{ 0 };

   long
   fib(long n) __attribute__((noinline));

   void
   run(size_t nitr, long) __attribute__((noinline));

   int
   custom_push_region(const char* name);

   namespace
   {
   omnitrace_user_callbacks_t custom_callbacks   = OMNITRACE_USER_CALLBACKS_INIT;
   omnitrace_user_callbacks_t original_callbacks = OMNITRACE_USER_CALLBACKS_INIT;
   }  // namespace

   int
   main(int argc, char** argv)
   {
      custom_callbacks.push_region = &custom_push_region;
      omnitrace_user_configure(OMNITRACE_USER_UNION_CONFIG, custom_callbacks,
                              &original_callbacks);

      omnitrace_user_push_region(argv[0]);
      omnitrace_user_push_region("initialization");
      size_t nthread = std::min<size_t>(16, std::thread::hardware_concurrency());
      size_t nitr    = 50000;
      long   nfib    = 10;
      if(argc > 1) nfib = atol(argv[1]);
      if(argc > 2) nthread = atol(argv[2]);
      if(argc > 3) nitr = atol(argv[3]);
      omnitrace_user_pop_region("initialization");

      printf("[%s] Threads: %zu\n[%s] Iterations: %zu\n[%s] fibonacci(%li)...\n", argv[0],
            nthread, argv[0], nitr, argv[0], nfib);

      omnitrace_user_push_region("thread_creation");
      std::vector<std::thread> threads{};
      threads.reserve(nthread);
      // disable instrumentation for child threads
      omnitrace_user_stop_thread_trace();
      for(size_t i = 0; i < nthread; ++i)
      {
         threads.emplace_back(&run, nitr, nfib);
      }
      // re-enable instrumentation
      omnitrace_user_start_thread_trace();
      omnitrace_user_pop_region("thread_creation");

      omnitrace_user_push_region("thread_wait");
      for(auto& itr : threads)
         itr.join();
      omnitrace_user_pop_region("thread_wait");

      run(nitr, nfib);

      printf("[%s] fibonacci(%li) x %lu = %li\n", argv[0], nfib, nthread, total.load());
      omnitrace_user_pop_region(argv[0]);

      return 0;
   }

   long
   fib(long n)
   {
      return (n < 2) ? n : fib(n - 1) + fib(n - 2);
   }

   #define RUN_LABEL                                                                        \
      std::string{ std::string{ __FUNCTION__ } + "(" + std::to_string(n) + ") x " +        \
                  std::to_string(nitr) }                                                  \
         .c_str()

   void
   run(size_t nitr, long n)
   {
      omnitrace_user_push_region(RUN_LABEL);
      long local = 0;
      for(size_t i = 0; i < nitr; ++i)
         local += fib(n);
      total += local;
      omnitrace_user_pop_region(RUN_LABEL);
   }

   int
   custom_push_region(const char* name)
   {
      if(!original_callbacks.push_region || !original_callbacks.push_annotated_region)
         return OMNITRACE_USER_ERROR_NO_BINDING;

      printf("Pushing custom region :: %s\n", name);

      if(original_callbacks.push_annotated_region)
      {
         int32_t _err = errno;
         char*   _msg = nullptr;
         char    _buff[1024];
         if(_err != 0) _msg = strerror_r(_err, _buff, sizeof(_buff));

         omnitrace_annotation_t _annotations[] = {
               { "errno", OMNITRACE_INT32, &_err }, { "strerror", OMNITRACE_STRING, _msg }
         };

         errno = 0;  // reset errno
         return (*original_callbacks.push_annotated_region)(
               name, _annotations, sizeof(_annotations) / sizeof(omnitrace_annotation_t));
      }

      return (*original_callbacks.push_region)(name);
   }

Linking the Omnitrace libraries to another program
=======================================================

To link the ``omnitrace-user-library`` to another program,
use the following CMake and ``g++`` directives.

CMake
-------------------------------------------------------

.. code-block:: cmake

   find_package(omnitrace REQUIRED COMPONENTS user)
   add_executable(foo foo.cpp)
   target_link_libraries(foo PRIVATE omnitrace::omnitrace-user-library)

g++ compilation
-------------------------------------------------------

Assuming Omnitrace is installed in ``/opt/omnitrace``, use the ``g++`` compiler
to build the application.

.. code-block:: shell

   g++ -g -I/opt/omnitrace/include -L/opt/omnitrace/lib foo.cpp -o foo -lomnitrace-user

Output from the API example program
========================================

First, instrument and run the program.

.. code-block:: shell

   $ omnitrace-instrument -l --min-instructions=8 -E custom_push_region -o -- ./user-api
   ...
   $ omnitrace-run --profile --use-pid off --time-output off -- ./user-api.inst 10 12 1000
   OMNITRACE: HSA_TOOLS_LIB=/opt/omnitrace/lib/libomnitrace-dl.so.1.13.0
   OMNITRACE: HSA_TOOLS_REPORT_LOAD_FAILURE=1
   OMNITRACE: LD_PRELOAD=/opt/omnitrace/lib/libomnitrace-dl.so.1.13.0
   OMNITRACE: OMNITRACE_PROFILE=true
   OMNITRACE: OMNITRACE_TRACE=true
   OMNITRACE: OMNITRACE_VERBOSE=0
   OMNITRACE: OMP_TOOL_LIBRARIES=/opt/omnitrace/lib/libomnitrace-dl.so.1.13.0
   OMNITRACE: ROCP_HSA_INTERCEPT=1
   OMNITRACE: ROCP_TOOL_LIB=/opt/omnitrace/lib/libomnitrace.so.1.13.0
   [omnitrace][dl][3099154] omnitrace_main
   [omnitrace][3099154][omnitrace_init_tooling] Instrumentation mode: Trace


        ______   .___  ___. .__   __.  __  .___________..______          ___       ______  _______
       /  __  \  |   \/   | |  \ |  | |  | |           ||   _  \        /   \     /      ||   ____|
      |  |  |  | |  \  /  | |   \|  | |  | `---|  |----`|  |_)  |      /  ^  \   |  ,----'|  |__
      |  |  |  | |  |\/|  | |  . `  | |  |     |  |     |      /      /  /_\  \  |  |     |   __|
      |  `--'  | |  |  |  | |  |\   | |  |     |  |     |  |\  \----./  _____  \ |  `----.|  |____
       \______/  |__|  |__| |__| \__| |__|     |__|     | _| `._____/__/     \__\ \______||_______|

      omnitrace v1.13.0 (rev: 0a76f6f62fdb671e5ee5c2f1ca186d34117ef7f8, tag: v1.11.4-33-g0a76f6f, x86_64-linux-gnu, compiler: GNU v11.4.0, rocm: v6.3.x)
   [225.911]       perfetto.cc:47606 Configured tracing session 1, #sources:1, duration:0 ms, #buffers:1, total buffer size:1024000 KB, total sessions:1, uid:0 session name: ""
   Pushing custom region :: ./user-api.inst
   Pushing custom region :: initialization
   [./user-api.inst] Threads: 12
   [./user-api.inst] Iterations: 1000
   [./user-api.inst] fibonacci(10)...
   Pushing custom region :: thread_creation
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: run(10) x 1000
   Pushing custom region :: thread_wait
   Pushing custom region :: run(10) x 1000
   [./user-api.inst] fibonacci(10) x 12 = 715000

   [omnitrace][3099154][0][omnitrace_finalize] finalizing...
   [omnitrace][3099154][0][omnitrace_finalize]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154 : 0.969505 sec wall_clock,   25.856 MB peak_rss,   26.477 MB page_rss, 1.520000 sec cpu_clock,  156.8 % cpu_util [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/0 : 0.967630 sec wall_clock, 0.773075 sec thread_cpu_clock,   79.9 % thread_cpu_util,   25.216 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/1 : 0.027326 sec wall_clock, 0.027326 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.768 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/2 : 0.030120 sec wall_clock, 0.030097 sec thread_cpu_clock,   99.9 % thread_cpu_util,    3.968 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/3 : 0.034380 sec wall_clock, 0.034380 sec thread_cpu_clock,  100.0 % thread_cpu_util,    4.096 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/4 : 0.028657 sec wall_clock, 0.028651 sec thread_cpu_clock,  100.0 % thread_cpu_util,    3.968 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/5 : 0.032297 sec wall_clock, 0.032290 sec thread_cpu_clock,  100.0 % thread_cpu_util,    3.968 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/6 : 0.027747 sec wall_clock, 0.027735 sec thread_cpu_clock,  100.0 % thread_cpu_util,    3.840 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/7 : 0.034457 sec wall_clock, 0.034457 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.768 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/8 : 0.029121 sec wall_clock, 0.029121 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.768 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/9 : 0.032374 sec wall_clock, 0.032363 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.768 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/10 : 0.032084 sec wall_clock, 0.032084 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.512 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/11 : 0.029105 sec wall_clock, 0.029093 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.640 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize] omnitrace/process/3099154/thread/12 : 0.030908 sec wall_clock, 0.030901 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.384 MB peak_rss [laps: 1]
   [omnitrace][3099154][0][omnitrace_finalize]
   [omnitrace][3099154][0][omnitrace_finalize] Finalizing perfetto...
   [omnitrace][3099154][perfetto]> Outputting '/home/gliff/opt/user-api-omni-test/omnitrace-user-api.inst-output/2025-01-06_17.25/perfetto-trace-3099154.proto' (16728.65 KB / 16.73 MB / 0.02 GB)... Done
   [omnitrace][3099154][wall_clock]> Outputting 'omnitrace-user-api.inst-output/2025-01-06_17.25/wall_clock-3099154.json'
   [omnitrace][3099154][wall_clock]> Outputting 'omnitrace-user-api.inst-output/2025-01-06_17.25/wall_clock-3099154.txt'
   [omnitrace][3099154][roctracer]> Outputting 'omnitrace-user-api.inst-output/2025-01-06_17.25/roctracer-3099154.json'
   [omnitrace][3099154][roctracer]> Outputting 'omnitrace-user-api.inst-output/2025-01-06_17.25/roctracer-3099154.txt'
   [omnitrace][3099154][metadata]> Outputting 'omnitrace-user-api.inst-output/2025-01-06_17.25/metadata-3099154.json' and 'omnitrace-user-api.inst-output/2025-01-06_17.25/functions-3099154.json'
   [omnitrace][3099154][0][omnitrace_finalize] Finalized: 0.278750 sec wall_clock,   21.888 MB peak_rss,    6.250 MB page_rss, 0.250000 sec cpu_clock,   89.7 % cpu_util
   [227.163]       perfetto.cc:49204 Tracing session 1 ended, total sessions:0


Then review the output.

.. code-block:: shell

   $ cat omnitrace-example-output/wall_clock.txt
   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   |                                                                                          REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                                          |
   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   |                                                 LABEL                                                   | COUNT  | DEPTH  |   METRIC   | UNITS  |   SUM    |   MEAN   |   MIN    |   MAX    |   VAR    | STDDEV   | % SELF |
   |---------------------------------------------------------------------------------------------------------|--------|--------|------------|--------|----------|----------|----------|----------|----------|----------|--------|
   | |00>>> ./user-api.inst                                                                                  |      1 |      0 | wall_clock | sec    | 0.866498 | 0.866498 | 0.866498 | 0.866498 | 0.000000 | 0.000000 |    0.0 |
   | |00>>> |_initialization                                                                                 |      1 |      1 | wall_clock | sec    | 0.000014 | 0.000014 | 0.000014 | 0.000014 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_thread_creation                                                                                |      1 |      1 | wall_clock | sec    | 0.064493 | 0.064493 | 0.064493 | 0.064493 | 0.000000 | 0.000000 |    0.7 |
   | |00>>>   |_pthread_create                                                                               |     12 |      2 | wall_clock | sec    | 0.064047 | 0.005337 | 0.004141 | 0.007329 | 0.000001 | 0.001020 |    0.0 |
   | |01>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.027309 | 0.027309 | 0.027309 | 0.027309 | 0.000000 | 0.000000 |    0.1 |
   | |01>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.027270 | 0.027270 | 0.027270 | 0.027270 | 0.000000 | 0.000000 |  100.0 |
   | |02>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.030069 | 0.030069 | 0.030069 | 0.030069 | 0.000000 | 0.000000 |    0.3 |
   | |02>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.029977 | 0.029977 | 0.029977 | 0.029977 | 0.000000 | 0.000000 |  100.0 |
   | |04>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.028631 | 0.028631 | 0.028631 | 0.028631 | 0.000000 | 0.000000 |    0.2 |
   | |04>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.028582 | 0.028582 | 0.028582 | 0.028582 | 0.000000 | 0.000000 |  100.0 |
   | |03>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.034353 | 0.034353 | 0.034353 | 0.034353 | 0.000000 | 0.000000 |    0.1 |
   | |03>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.034305 | 0.034305 | 0.034305 | 0.034305 | 0.000000 | 0.000000 |  100.0 |
   | |06>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.027728 | 0.027728 | 0.027728 | 0.027728 | 0.000000 | 0.000000 |    0.1 |
   | |06>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.027688 | 0.027688 | 0.027688 | 0.027688 | 0.000000 | 0.000000 |  100.0 |
   | |05>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.032277 | 0.032277 | 0.032277 | 0.032277 | 0.000000 | 0.000000 |    0.1 |
   | |05>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.032232 | 0.032232 | 0.032232 | 0.032232 | 0.000000 | 0.000000 |  100.0 |
   | |08>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.029103 | 0.029103 | 0.029103 | 0.029103 | 0.000000 | 0.000000 |    0.2 |
   | |08>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.029056 | 0.029056 | 0.029056 | 0.029056 | 0.000000 | 0.000000 |  100.0 |
   | |07>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.034426 | 0.034426 | 0.034426 | 0.034426 | 0.000000 | 0.000000 |    0.2 |
   | |07>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.034361 | 0.034361 | 0.034361 | 0.034361 | 0.000000 | 0.000000 |  100.0 |
   | |09>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.032357 | 0.032357 | 0.032357 | 0.032357 | 0.000000 | 0.000000 |    0.1 |
   | |09>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.032313 | 0.032313 | 0.032313 | 0.032313 | 0.000000 | 0.000000 |  100.0 |
   | |10>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.032066 | 0.032066 | 0.032066 | 0.032066 | 0.000000 | 0.000000 |    0.1 |
   | |10>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.032027 | 0.032027 | 0.032027 | 0.032027 | 0.000000 | 0.000000 |  100.0 |
   | |11>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.029081 | 0.029081 | 0.029081 | 0.029081 | 0.000000 | 0.000000 |    0.2 |
   | |11>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.029023 | 0.029023 | 0.029023 | 0.029023 | 0.000000 | 0.000000 |  100.0 |
   | |12>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.030888 | 0.030888 | 0.030888 | 0.030888 | 0.000000 | 0.000000 |    0.1 |
   | |12>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.030849 | 0.030849 | 0.030849 | 0.030849 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_thread_wait                                                                                    |      1 |      1 | wall_clock | sec    | 0.031091 | 0.031091 | 0.031091 | 0.031091 | 0.000000 | 0.000000 |    0.7 |
   | |00>>>   |_std::vector<std::thread, std::allocator<std::thread> >::begin                                |      1 |      2 | wall_clock | sec    | 0.000003 | 0.000003 | 0.000003 | 0.000003 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |_std::vector<std::thread, std::allocator<std::thread> >::end                                  |      1 |      2 | wall_clock | sec    | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |___gnu_cxx::operator!=<std::thread*, std::vector<std::thread, std::allocator<std::thread> > > |     13 |      2 | wall_clock | sec    | 0.000017 | 0.000001 | 0.000001 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |_pthread_join                                                                                 |     12 |      2 | wall_clock | sec    | 0.030837 | 0.002570 | 0.000002 | 0.010114 | 0.000014 | 0.003801 |  100.0 |
   | |00>>>     |_std::distance<char const*>                                                                 |      4 |      3 | wall_clock | sec    | 0.000002 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_run                                                                                            |      1 |      1 | wall_clock | sec    | 0.770776 | 0.770776 | 0.770776 | 0.770776 | 0.000000 | 0.000000 |    0.0 |
   | |00>>>   |_std::char_traits<char>::length                                                               |      1 |      2 | wall_clock | sec    | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |_std::distance<char const*>                                                                   |      1 |      2 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |_std::operator+<char, std::char_traits<char>, std::allocator<char> >                          |      4 |      2 | wall_clock | sec    | 0.000005 | 0.000001 | 0.000001 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |_run(10) x 1000                                                                               |      1 |      2 | wall_clock | sec    | 0.770734 | 0.770734 | 0.770734 | 0.770734 | 0.000000 | 0.000000 |    0.0 |
   | |00>>>     |_run [{95,25}-{97,25}]                                                                      |      1 |      3 | wall_clock | sec    | 0.770693 | 0.770693 | 0.770693 | 0.770693 | 0.000000 | 0.000000 |    0.4 |
   | |00>>>       |_fib                                                                                      |   1000 |      4 | wall_clock | sec    | 0.767246 | 0.000767 | 0.000748 | 0.001192 | 0.000000 | 0.000018 |    1.0 |
   | |00>>>         |_fib                                                                                    |   2000 |      5 | wall_clock | sec    | 0.759546 | 0.000380 | 0.000278 | 0.000892 | 0.000000 | 0.000092 |    2.0 |
   | |00>>>           |_fib                                                                                  |   4000 |      6 | wall_clock | sec    | 0.744236 | 0.000186 | 0.000100 | 0.000716 | 0.000000 | 0.000066 |    4.0 |
   | |00>>>             |_fib                                                                                |   8000 |      7 | wall_clock | sec    | 0.714185 | 0.000089 | 0.000034 | 0.000537 | 0.000000 | 0.000041 |    8.6 |
   | |00>>>               |_fib                                                                              |  16000 |      8 | wall_clock | sec    | 0.652740 | 0.000041 | 0.000009 | 0.000464 | 0.000000 | 0.000024 |   18.7 |
   | |00>>>                 |_fib                                                                            |  32000 |      9 | wall_clock | sec    | 0.530611 | 0.000017 | 0.000001 | 0.000446 | 0.000000 | 0.000014 |   38.8 |
   | |00>>>                   |_fib                                                                          |  52000 |     10 | wall_clock | sec    | 0.324687 | 0.000006 | 0.000001 | 0.000428 | 0.000000 | 0.000008 |   61.7 |
   | |00>>>                     |_fib                                                                        |  44000 |     11 | wall_clock | sec    | 0.124277 | 0.000003 | 0.000001 | 0.000056 | 0.000000 | 0.000004 |   79.8 |
   | |00>>>                       |_fib                                                                      |  16000 |     12 | wall_clock | sec    | 0.025094 | 0.000002 | 0.000001 | 0.000029 | 0.000000 | 0.000002 |   91.8 |
   | |00>>>                         |_fib                                                                    |   2000 |     13 | wall_clock | sec    | 0.002057 | 0.000001 | 0.000001 | 0.000012 | 0.000000 | 0.000001 |  100.0 |
   | |00>>>     |_std::char_traits<char>::length                                                             |      1 |      3 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>     |_std::distance<char const*>                                                                 |      1 |      3 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>     |_std::operator+<char, std::char_traits<char>, std::allocator<char> >                        |      4 |      3 | wall_clock | sec    | 0.000005 | 0.000001 | 0.000001 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_std::operator&                                                                                 |      1 |      1 | wall_clock | sec    | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> std::vector<std::thread, std::allocator<std::thread> >::~vector                                  |      1 |      0 | wall_clock | sec    | 0.000239 | 0.000239 | 0.000239 | 0.000239 | 0.000000 | 0.000000 |   22.1 |
   | |00>>> |_std::thread::~thread                                                                           |     12 |      1 | wall_clock | sec    | 0.000176 | 0.000015 | 0.000013 | 0.000018 | 0.000000 | 0.000002 |   32.5 |
   | |00>>>   |_std::thread::joinable                                                                        |     12 |      2 | wall_clock | sec    | 0.000119 | 0.000010 | 0.000009 | 0.000012 | 0.000000 | 0.000001 |   79.1 |
   | |00>>>     |_std::thread::id::id                                                                        |     12 |      3 | wall_clock | sec    | 0.000013 | 0.000001 | 0.000001 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>     |_std::operator==                                                                            |     12 |      3 | wall_clock | sec    | 0.000012 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_std::allocator_traits<std::allocator<std::thread> >::deallocate                                |      1 |      1 | wall_clock | sec    | 0.000008 | 0.000008 | 0.000008 | 0.000008 | 0.000000 | 0.000000 |   75.6 |
   | |00>>>   |___gnu_cxx::new_allocator<std::thread>::deallocate                                            |      1 |      2 | wall_clock | sec    | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_std::allocator<std::thread>::~allocator                                                        |      1 |      1 | wall_clock | sec    | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
