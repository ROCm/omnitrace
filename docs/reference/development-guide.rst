.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Development guide
****************************************************

This guide discusses the `Omnitrace <https://github.com/ROCm/omnitrace>`_ design. 
It includes a list of the executables and libraries, along with a discussion of its 
memory, sampling, and time-window constraint models.

Executables
========================================

This section lists the Omnitrace executables.

``omnitrace-avail``: `source/bin/omnitrace-avail <https://github.com/ROCm/omnitrace/tree/main/source/bin/omnitrace-avail>`_
-------------------------------------------------------------------------------------------------------------------------------

The ``main`` routine of ``omnitrace-avail`` has three important sections:

* Printing components
* Printing options
* Printing hardware counters

``omnitrace-sample``: `source/bin/omnitrace-sample <https://github.com/ROCm/omnitrace/tree/main/source/bin/omnitrace-sample>`_
-------------------------------------------------------------------------------------------------------------------------------

General design:

* Requires a command-line format of ``omnitrace-sample <options> -- <command> <command-args>``
* Translates command line options into environment variables
* Adds ``libomnitrace-dl.so`` to ``LD_PRELOAD``
* Application is launched via ``execvpe`` with ``<command> <command-args>`` and modified environment

``omnitrace-casual``: `source/bin/omnitrace-causal <https://github.com/ROCm/omnitrace/tree/main/source/bin/omnitrace-causal>`_
-------------------------------------------------------------------------------------------------------------------------------

This has a nearly identical design to ``omnitrace-sample`` when
there is exactly one causal profiling configuration variant (this enables debugging).

When more than one causal profiling configuration variant it produced from command-line options,
for each variant:

* ``omnitrace-causal`` calls ``fork()``
* child process launches ``<command> <command-args>`` via ``execvpe`` which modified environment for variant
* parent process waits for child process to finish

``omnitrace-instrument``: `source/bin/omnitrace-instrument <https://github.com/ROCm/omnitrace/tree/main/source/bin/omnitrace-instrument>`_
-------------------------------------------------------------------------------------------------------------------------------------------

* Requires a command-line format of ``omnitrace-instrument <options> -- <command> <command-args>``
* User specifies in options whether they want to do runtime instrumentation, binary rewrite, or 
  attach to process
* Either opens the instrumentation target (binary rewrite), launches the target and stops it
  before it starts executing main (runtime), or
  attaches to running executable and pauses it
* Finds all functions in target(s)
* Finds ``libomnitrace-dl`` and finds the functions
* Iterates over all the functions and instruments them as long as they satisfy the 
  defined criteria (minimum number of instructions, etc.)

  * See the ``module_function`` class

* Most of the workflow has been the same at the point but once the instrumentation is complete, it diverges

  * For a binary rewrite: outputs new instrumented binary and exits
  * For runtime instrumentation or attaching to a process: instructs the application 
    to resume executing and then waits for the application to exit

Libraries
========================================

Common library: `source/lib/common <https://github.com/ROCm/omnitrace/tree/main/source/lib/common>`_
--------------------------------------------------------------------------------------------------------------------------------

General header-only functionality used in multiple executables and/or libraries. 
Not installed or exported outside of the build tree.

Core library: `source/lib/core <https://github.com/ROCm/omnitrace/tree/main/source/lib/core>`_
--------------------------------------------------------------------------------------------------------------------------------

Static PIC library with functionality that does not depend on any components. 
Not installed or exported outside of the build tree.

Binary library: `source/lib/binary <https://github.com/ROCm/omnitrace/tree/main/source/lib/binary>`_
--------------------------------------------------------------------------------------------------------------------------------

Static PIC library with functionality for reading/analyzing binary info. Mostly used by the 
causal profiling sections of ``libomnitrace``. Not installed or exported outside of the build tree.

``libomnitrace``: `source/lib/omnitrace <https://github.com/ROCm/omnitrace/tree/main/source/lib/omnitrace>`_
--------------------------------------------------------------------------------------------------------------------------------

This is the main library encapsulating all the capabilities.

``libomnitrace-dl``: `source/lib/omnitrace-dl <https://github.com/ROCm/omnitrace/tree/main/source/lib/omnitrace-dl>`_
--------------------------------------------------------------------------------------------------------------------------------

Lightweight, front-end library for ``libomnitrace`` which serves three primary purposes:

* Dramatically speeds up instrumentation time vs. using ``libomnitrace`` directly since 
  Dyninst must parse the entire library in order to find instrumentation functions 
  (a ``dlopen`` call is made on ``libomnitrace`` when the instrumentation functions get called)
* Prevents re-entry if ``libomnitrace`` calls an instrumented function internally
* Coordinates communication between ``libomnitrace-user`` and ``libomnitrace``

``libomnitrace-user``: `source/lib/omnitrace-user <https://github.com/ROCm/omnitrace/tree/main/source/lib/omnitrace-user>`_
--------------------------------------------------------------------------------------------------------------------------------

Provides a set of functions and types for the users to add to their code, 
e.g. disabling data collection globally or on a specific thread,
user-defined regions, etc. If ``libomnitrace-dl`` is not loaded, the user API is effectively 
no-op function calls.

Testing tools
========================================

* `CDash Testing Dashboard <https://my.cdash.org/index.php?project=Omnitrace>`_ (requires a login)

Components
========================================

Most measurements and capabilities are encapsulated into a "component" with the following definitions:

Measurement
   A recording of some data relevant to performance, e.g. current call-stack, 
   hardware counter values, current memory usage, timestamp

Capability
   Handles the implementation or orchestration of some feature which is used 
   to collect measurements, e.g. a component which handles setting up function wrappers 
   around various functions such as ``pthread_create``, ``MPI_Init``, etc.

Components are designed to hold no data at all or only the data for both an instantaneous 
measurement and a phase measurement.

Components which store data typically implement a static ``record()`` function 
(for getting a record of the measurement),
``start()`` + ``stop()`` member functions for calculating a phase measurement, 
and a ``sample()`` member function for storing an
instantaneous measurement. In reality, there are several more "standard" functions 
but these are the most often used ones.

Components which do not store data may also have ``start()``, ``stop()``, and ``sample()`` 
functions but for components which
implement function wrappers, they typically provide a call operator or ``audit(...)`` 
functions which are invoked with the
wrapped function's arguments before the wrapped function gets called and with the return value 
after the wrapped function gets called.

.. note::

   The goal of this design is to provide relatively small and resuable lightweight objects 
   for recording measurements and/or implementing capabilities.

Wall-clock component example
--------------------------------------

A component for computing the elapsed wall-clock time looks like this:

.. code-block:: cpp

   struct wall_clock
   {
      using value_type = int64_t;

      static value_type record() noexcept
      {
         return std::chrono::steady_clock::now().time_since_epoch().count();
      }

      void sample() noexcept
      {
         value = record();
      }

      void start() noexcept
      {
         value = record();
      }

      void stop() noexcept
      {
         auto _start_value = value;
         value = record();
         accum += (value - _start_value);
      }

   private:
      int64_t value = 0;
      int64_t accum = 0;
   };

Function wrapper component example
--------------------------------------

A component which implements wrappers around ``fork()`` and ``exit(int)`` (and stores no data) 
may look like this:

.. code-block:: cpp

   struct function_wrapper
   {
      pid_t operator()(const gotcha_data&, pid_t (*real_fork)())
      {
         // disable all collection before forking
         categories::disable_categories(config::get_enabled_categories());

         auto _pid_v = real_fork();

         // only re-enable collection on parent process
         if(_pid_v != 0)
               categories::enable_categories(config::get_enabled_categories());

         return _pid_v;
      }

      void operator()(const gotcha_data&, void (*real_exit)(int), int _exit_code)
      {
         // catch the call to exit and finalize before truly exiting
         omnitrace_finalize();

         real_exit(_exit_code);
      }
   };

Component member functions
--------------------------------------

There are no real restrictions or requirements on the member functions a component needs to provide.
Unless the component is being directly used, invocation of component member functions via a "component bundler"
(provided via timemory) makes extensive use of template metaprogramming concept to find the best match (if any)
for calling a components member function. This is a bit easier to demonstrate via example:

.. code-block:: cpp

   struct foo
   {
      void sample() { puts("foo::sample()"); }
   };

   struct bar
   {
      void sample(int) { puts("bar::sample(int)"); }
   };

   struct spam
   {
      void start(int) { puts("spam::start()"); }
      void stop()     { puts("spam::stop()"); }
   };

   int main()
   {
      auto _bundle = component_tuple<foo, bar, spam>{ "main" };

      puts("A");
      _bundle.start();

      puts("B");
      _bundle.sample(10);

      puts("C");
      _bundle.sample();

      puts("D");
      _bundle.stop();
   }

In the above, this would be the message printed:

.. code-block:: shell

   A
   bar::start()
   B
   foo::sample()
   bar::sample(int)
   C
   foo::sample()
   D
   spam::stop()

In section A, the bundle determined only the ``spam`` object had a ``start`` function. Since this is determined
via template metaprogramming instead of dynamic polymorphism, this effectively elides any code related to
the ``foo`` or ``bar`` objects. In section B, since an integer of ``10`` was passed to the bundle,
the bundle forwards that value onto ``spam::sample(int)`` after it invokes ``foo::sample()`` -- which
is invoked because it recognizes that the call is the ``sample`` member function is still possible without
the arguments.

Memory model
========================================

Collected data is generally stored in one of following three places:

* Perfetto (i.e. data is handed directly to Perfetto)
* Managed implicitly by timemory and accessed as needed
* Thread-local data

In general, only instrumentation for relatively simple data is directly passed to 
Perfetto and/or timemory during runtime.
For example, the callbacks from binary instrumentation, user API instrumentation, 
and roctracer directly invoke
calls to Perfetto and/or timemory's storage model. Otherwise, the data is stored 
by Omnitrace in the thread-data model
which is more persistent than simply using ``thread_local`` static data 
(which is problematic because the data gets deleted
when a thread terminates).

Thread identification
--------------------------------------

Each CPU thread is assigned two integral identifiers. One identifier is simply an 
atomic increment every time a new thread is created
(called ``internal_value``).
The other identifier tries to account for the fact that Omnitrace, Perfetto, ROCm, etc. 
start background threads and for these threads
(called ``sequent_value``). When a thread is created as a byproduct of Omnitrace, 
the index is offset by a large value. This serves
two purposes: (1) accessing the data for threads created by the user is closer in 
memory and (2) when log messages are printed,
the index more-or-less correlates to the order of thread creation to the user's knowledge.

The ``sequent_value`` is typically the one used to access the thread-data.

Thread-data class
--------------------------------------

Currently, most thread data is effectively stored in a static 
``std::array<std::unique_ptr<T>, OMNITRACE_MAX_THREADS>`` instance.
``OMNITRACE_MAX_THREADS`` is a value defined a compile-time and set to 2048 
for release builds. During finalization,
Omnitrace iterates over all the thread-data and then transforms that data 
into something that is passed to Perfetto and/or timemory.
The downside of the current model is that if the user exceeds ``OMNITRACE_MAX_THREADS``, 
a segmentation fault occurs. To fix this issue,
a new model is being adopted which has all the benefits of this model 
but permits dynamic expansion.

Sampling model
========================================

The general structure for the sampling is within timemory (``source/timemory/sampling``). 
Currently, all sampling is done per-thread
via POSIX timers. Omnitrace supports using a real-time timer and a CPU-time timer. 
Both have adjustable frequencies, delays, and durations.
By default, only CPU-time sampling is enabled. Initial settings are inherited from 
the settings starting with ``OMNITRACE_SAMPLING_``.
For each type of timer, there exists timer-specific settings that can be used to 
override the common/inherited settings for that timer
specifically. For the CPU-time sampler, these settings start with ``OMNITRACE_SAMPLING_CPUTIME`` 
and ``OMNITRACE_SAMPLING_REALTIME`` for
the real-time sampler. For example, ``OMNITRACE_SAMPLING_FREQ=500`` initially sets the 
sampling frequency to 500 interrupts per second
(based on their clock). Settings ``OMNITRACE_SAMPLING_REALTIME_FREQ=10`` will lower 
the sampling frequency for the real-time sampler
to 10 interrupts per second of real-time.

The Omnitrace-specific implementation can be found in 
`source/lib/omnitrace/library/sampling.cpp <https://github.com/ROCm/omnitrace/blob/main/source/lib/omnitrace/library/sampling.cpp>`_.
Within `sampling.cpp <https://github.com/ROCm/omnitrace/blob/main/source/lib/omnitrace/library/sampling.cpp>`_, 
you will a bundle of three sampling components:
``backtrace_timestamp``, ``backtrace``, and ``backtrace_metrics``.

* The first component `backtrace_timestamp <https://github.com/ROCm/omnitrace/blob/main/source/lib/omnitrace/library/components/backtrace_timestamp.hpp>`_ simply
  records the wall-clock time of the sample.
* The second component `backtrace <https://github.com/ROCm/omnitrace/blob/main/source/lib/omnitrace/library/components/backtrace.hpp>`_
  records the call-stack via libunwind.
* The last component `backtrace_metrics <https://github.com/ROCm/omnitrace/blob/main/source/lib/omnitrace/library/components/backtrace_metrics.hpp>`_
  is responsible for recording the metrics for that sample, e.g. peak RSS, hardware counters, etc.

These three components are bundled together in 
a tuple-like struct (e.g. ``tuple<backtrace_timestamp, backtrace, backtrace_metrics>``)
a buffer of at least 1024 instances of this tuple are mapped using ``mmap`` per-thread. When this buffer is full, 
before taking the next sample, the sampler will hand the buffer
off to it's allocator thread and mmap a new buffer. The allocator thread takes this data 
and either dynamically stores it in memory or writes it to a file depending on the 
value of ``OMNITRACE_USE_TEMPORARY_FILES``.
This schema avoids all allocations in the signal handler, allows the data to grow 
dynamically, avoid potentially slow I/O within the signal handler, and also enables 
the capability to avoid I/O altogether.
The maximum number of samplers handled by each allocator is governed by the 
``OMNITRACE_SAMPLING_ALLOCATOR_SIZE`` setting (the default is 8). Whenever an allocator has reached its limit,
a new internal thread is created to handle the new samplers.

Time-window constraint model
========================================

Recently with the introduction of tracing delay/duration/etc., the 
`constraint namespace <https://github.com/ROCm/omnitrace/blob/main/source/lib/core/constraint.hpp>`_
was introduced to improve the management of delays and/or duration limits of 
data collection. The ``spec`` class takes a clock identifier, a delay value, a duration value, and an
integer indicating how many times to repeat the delay + duration. Thus, it is 
possible to perform tasks such as periodically enabling tracing for brief periods
of time in between long periods without data collection during the application. 
For example, ``OMNITRACE_TRACE_PERIODS = realtime:10:1:5 process_cputime:10:2:20`` enables
five periods of no data collection for ten seconds of real-time, followed by one second of 
data collection, plus twenty periods of no data collection for ten seconds
of process CPU time, followed by two CPU-time seconds of data collection.

Eventually, the goal is have all subsets of data collection which currently support 
more rudimentary models of time window constraints, such as process sampling and causal profiling,
to be migrated to this model.