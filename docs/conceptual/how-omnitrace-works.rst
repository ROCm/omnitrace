.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

*******************
How Omnitrace works
*******************

This page explains the nomenclature necessary to use Omnitrace and provides 
some basic tips to help you get started. It also explains the main data
collection modes, including a comparison between binary instrumentation 
and statistical sampling.

Omnitrace nomenclature
========================================

The list provided below is intended to provide a basic glossary for those who 
are not familiar with binary instrumentation. It also clarifies ambiguities 
when certain terms have different 
contextual meanings, for example, the Omnitrace meaning of the term "module" 
when instrumenting Python.

**Binary**
  A file written in the Executable and Linkable Format (ELF). This is the standard file 
  format for executable files, shared libraries, etc.

**Binary instrumentation**
  Inserting callbacks to instrumentation into an existing binary. This can be performed 
  statically or dynamically.

**Static binary instrumentation**
  Loads an existing binary, determines instrumentation points, and generates a new binary 
  with instrumentation directly embedded. It is applicable to executables and libraries but 
  limited to only the functions defined in the binary. This is also known as **Binary rewrite**.

**Dynamic binary instrumentation**
  Loads an existing binary into memory, inserts instrumentation, and runs the binary. 
  It is limited to executables but is capable of instrumenting linked libraries. 
  This is also known as: **Runtime instrumentation**.

**Statistical sampling**  
  At periodic intervals, the application is paused and the current call-stack of the CPU 
  is recorded along with various other metrics. It uses timers that measure either 
  (A) real clock time or (B) the CPU time used by the current thread and the CPU time 
  expended on behalf of the thread by the system. This is also known as simply **sampling**.

  **Sampling rate**
    * The period at which (A) or (B) are triggered (in units of ``# interrupts / second``)
    * Higher values increase the number of samples

  **Sampling delay**
    * How long to wait before (A) and (B) begin triggering at their designated rate

  **Sampling duration**
    * The amount of time (in real-time) after the start of the application to record samples. 
    * After this time limit has been reached, no more samples are recorded.

**Process sampling**
  At periodic (real-time) intervals, a background thread records global metrics without 
  interrupting the current process. These metrics include, but are not limited to: 
  CPU frequency, CPU memory high-water mark (i.e. peak memory usage), GPU temperature,
  and GPU power usage.

  **Sampling rate**
    * The real-time period for recording metrics (in units of ``# measurements / second``)
    * Higher values increase the number of samples

  **Sampling delay**
    * How long to wait (in real-time) before recording samples

  **Sampling duration**
    * The amount of time (in real-time) after the start of the application to record samples. 
    * After this time limit has been reached, no more samples are recorded.

**Module**
  With respect to binary instrumentation, a module is defined as either the filename 
  (such as ``foo.c``) or library name (``libfoo.so``) which contains the definition 
  of one or more functions.

  With respect to Python instrumentation, a module is defined as the **file** which contains 
  the definition of one or more functions. The full path to this file typically contains the 
  name of the "Python module".

**Basic block**
  A straight-line code sequence with no branches in (except for the entry) and 
  no branches out (except for the exit).

**Address range**
  The instructions for a function in a binary start at certain address with the ELF file 
  and end at a certain address. The range is ``end - start``.

  The address range is a decent approximation for the "cost" of a function. 
  For example, a larger address range approximately equates to more instructions.

**Instrumentation traps**
  On the x86 architecture, because instructions are of variable size, an instruction 
  might be too small for Dyninst to replace it with the normal code sequence 
  used to call instrumentation. When instrumentation is placed at points other 
  than subroutine entry, exit, or call points, traps may be used to ensure 
  the instrumentation fits. (By default, ``omnitrace-instrument`` avoids instrumentation 
  which requires a trap.)

**Overlapping functions**
  Due to language constructs or compiler optimizations, it might be possible for 
  multiple functions to overlap (that is, share part of the same function body) 
  or for a single function to have multiple entry points. In practice, it's 
  impossible to determine the difference between multiple overlapping functions 
  and a single function with multiple entry points. (By default, ``omnitrace-instrument`` 
  avoids instrumenting overlapping functions.)

General tips for using Omnitrace
========================================

* Use ``omnitrace-avail`` to look up configuration settings, hardware counters, and data collection components

  * Use ``-d`` flag for descriptions

* Generate a default configuration with ``omnitrace-avail -G ${HOME}/.omnitrace.cfg`` and adjust it 
  to the desired default behavior
* **Decide whether binary instrumentation, statistical sampling, or both** provides the desired performance data (for non-Python applications)
* Compile code with optimization enabled (``-O2`` or higher), disable asserts (i.e. ``-DNDEBUG``), and include debug info (for instance, ``-g1`` at a minimum)

  * Compiling with debug info does not slow down the code, it only increases compile time and the size of the binary
  * In CMake, this is generally done with the settings ``CMAKE_BUILD_TYPE=RelWithDebInfo`` or ``CMAKE_BUILD_TYPE=Release`` and ``CMAKE_<LANG>_FLAGS=-g1``

* **Use binary instrumentation for characterizing the performance of every invocation of specific functions**
* **Use statistical sampling to characterize the performance of the entire application while minimizing overhead**
* Enable statistical sampling after binary instrumentation to help "fill in the gaps" between instrumented regions
* Use the user API to create custom regions and enable/disable Omnitrace for specific processes, threads, and regions
* Dynamic symbol interception, callback APIs, and the user API are always available with binary instrumentation and sampling

  * Dynamic symbol interception and callback APIs are (generally) controlled through ``OMNITRACE_USE_<API>`` 
    options, for example, ``OMNITRACE_USE_KOKKOSP`` and ``OMNITRACE_USE_OMPT`` enable Kokkos-Tools and OpenMP-Tools 
    callbacks, respectively

* When generically seeking regions for performance improvement:

  * **Start off by collecting a flat profile**
  * Look for functions with high call counts, large cumulative runtimes/values, or large standard deviations
  
    * When call counts are high, improving the performance of this function or "inlining" the function can result in quick and easy performance improvements
    * When the standard deviation is high, collect a hierarchical profile and see if the high variation can be attributable to the calling context. 
      In this scenario, consider creating a specialized version of the function for the longer-running contexts

  * **Collect a hierarchical profile** and verify the functions that are part of the "critical path" of your 
    application, as indicated in the flat profile

    * For example, functions with high call counts but which are part of a "setup" or "post-processing" 
      phase that does not consume much time relative to the overall time are generally a lower priority for optimization

* **Use the information from the profiles when analyzing detailed traces**
* When using binary instrumentation in "trace" mode, **binary rewrites are preferable to runtime instrumentation**.

  * Binary rewrites only instrument the functions defined in the target binary, whereas runtime instrumentation might instrument functions defined in the shared libraries which are linked into the target binary

* When using binary instrumentation with MPI, avoid runtime instrumentation

  * Runtime instrumentation requires a fork and a ``ptrace``, which is generally incompatible with how MPI applications spawn processes
  * Perform a binary rewrite of the executable (and optionally, libraries used by the executable) using MPI and run 
    the generated instrumented executable using ``omnitrace-run`` instead of the original. 
    For example, instead of ``mpirun -n 2 ./myexe``, use ``mpirun -n 2 omnitrace-run -- ./myexe.inst``, where 
    ``myexe.inst`` is the instrumented ``myexe`` executable that was generated.

Data collection modes
========================================

Omnitrace supports several modes of recording trace and profiling data for your application:

+-----------------------------+---------------------------------------------------------+
| Mode                        | Description                                             |
+=============================+=========================================================+
| Binary Instrumentation      | Locates functions (and loops, if desired) in the binary |
|                             | and inserts snippets at the entry and exit              |
+-----------------------------+---------------------------------------------------------+
| Statistical Sampling        | Periodically pauses application at specified intervals  |
|                             | and records various metrics for the given call stack    |
+-----------------------------+---------------------------------------------------------+
| Callback APIs               | Parallelism frameworks such as ROCm, OpenMP, and Kokkos |
|                             | make callbacks into Omnitrace to provide information    |
|                             | about the work the API is performing                    |
+-----------------------------+---------------------------------------------------------+
| Dynamic Symbol Interception | Wrap function symbols defined in a position independent |
|                             | dynamic library/executable, like ``pthread_mutex_lock`` |
|                             | in ``libpthread.so`` or ``MPI_Init`` in the MPI library |
+-----------------------------+---------------------------------------------------------+
| User API                    | User-defined regions and controls for Omnitrace         |
+-----------------------------+---------------------------------------------------------+

The two most generic and important modes are binary instrumentation and statistical sampling. 
It is important to understand their advantages and disadvantages.
Binary instrumentation and statistical sampling can be performed with the ``omnitrace`` 
executable but for statistical sampling, it's highly recommended to use the
``omnitrace-sample`` executable instead if no binary instrumentation is required/desired. 
Callback APIs and dynamic symbol interception can be utilized with either tool.

Binary instrumentation
-----------------------------------

Binary instrumentation allows you to record deterministic measurements for 
every single invocation of a given function.
Binary instrumentation effectively adds instructions to the target application to 
collect the required information. It therefore has the potential to cause performance 
changes which might, in some cases, lead to inaccurate results. The effect depends on 
the information being collected and which features are activated in Omnitrace. 
For example, collecting only the wall-clock timing data
has less of an effect than collecting the wall-clock timing, CPU-clock timing, 
memory usage, cache-misses, and number of instructions that were run. Similarly, 
collecting a flat profile has less overhead than a hierarchical profile 
and collecting a trace OR a profile has less overhead than collecting a 
trace AND a profile.

In Omnitrace, the primary heuristic for controlling the overhead with binary 
instrumentation is the minimum number of instructions for selecting functions 
for instrumentation.

Statistical sampling
-----------------------------------

Statistical call-stack sampling periodically interrupts the application at 
regular intervals using operating system interrupts.
Sampling is typically less numerically accurate and specific, but allows the 
target program to run nearly at full speed.
In contrast to the data derived from binary instrumentation, the resulting 
data is not exact but is instead a statistical approximation.
However, sampling often provides a more accurate picture of the application 
execution because it is less intrusive to the target application and has fewer
side effects on memory caches or instruction decoding pipelines. Furthermore, 
because sampling does not affect the execution speed as much, is it
relatively immune to over-evaluating the cost of small, frequently called 
functions or "tight" loops.

In Omnitrace, the overhead for statistical sampling depends on the 
sampling rate and whether the samples are taken with respect to the CPU time 
and/or real time.

Binary instrumentation vs. statistical sampling example
-------------------------------------------------------

Consider the following code:

.. code-block:: c++

   long fib(long n)
   {
        if(n < 2) return n;
        return fib(n - 1) + fib(n - 2);
   }

   void run(long n)
   {
        long result = fib(nfib);
        printf("[%li] fibonacci(%li) = %li\n", i, nfib, result);
   }

   int main(int argc, char** argv)
   {
        long nfib = 30;
        long nitr = 10;
        if(argc > 1) nfib = atol(argv[1]);
        if(argc > 2) nitr = atol(argv[2]);

        for(long i = 0; i < nitr; ++i)
            run(nfib);

        return 0;
   }

Binary instrumentation of the ``fib`` function will record **every single invocation** 
of the function. For a very small function
such as ``fib``, this results in **significant** overhead since this simple function 
takes about 20 instructions, whereas the entry and
exit snippets are ~1024 instructions. Therefore, you generally want to avoid 
instrumenting functions where the instrumented function has significantly fewer
instructions than entry and exit instrumentation. (Note that many of the 
instructions in entry and exit functions are either logging functions or
depend on the runtime settings and thus might never run). However, 
due to the number of potential instructions in the entry and exit snippets,
the default behavior of ``omnitrace-instrument`` is to only instrument functions 
which contain fewer than 1024 instructions.

However, recording every single invocation of the function can be extremely 
useful for detecting anomalies, such as profiles that show minimum or maximum values much smaller or larger
than the average or a high standard deviation. In this case, traces allow you to 
identify exactly when and where those instances deviated from the norm.
Compare the level of detail in the following traces. In the top image, 
every instance of the ``fib`` function is instrumented, while in the bottom image,
the ``fib`` call-stack is derived via sampling.

Binary instrumentation of the Fibonacci function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../data/fibonacci-instrumented.png
   :alt: Visualization of the output of a binary instrumentation of the Fibonacci function

Statistical sampling of the Fibonacci function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../data/fibonacci-sampling.png
   :alt: Visualization of the output of a statistical sample of the Fibonacci function