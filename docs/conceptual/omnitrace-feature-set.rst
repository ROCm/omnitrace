.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

***************************************
The Omnitrace feature set and use cases
***************************************

`Omnitrace <https://github.com/ROCm/omnitrace>`_ is designed to be highly extensible. 
Internally, it leverages the `timemory performance analysis toolkit <https://github.com/NERSC/timemory>`_ 
to manage extensions, resources, data, and other items. It supports the following features, 
modes, metrics, and APIs.

Data collection modes
========================================

* Dynamic instrumentation

  * Runtime instrumentation: Instrument executables and shared libraries at runtime
  * Binary rewriting: Generate a new executable and/or library with instrumentation built-in

* Statistical sampling: Periodic software interrupts per-thread
* Process-level sampling: Background thread records process-, system- and device-level metrics while the application runs
* Causal profiling: Quantifies the potential impact of optimizations in parallel code
  
.. note::

   Critical trace support was removed in Omnitrace v1.11.0. 
   It was replaced by the causal profiling feature.

Data analysis
========================================

* High-level summary profiles with mean, min, max, and standard deviation statistics

  * Low overhead, memory efficient
  * Ideal for running at scale

* Comprehensive traces for every individual event and measurement
* Application speed-up predictions resulting from potential optimizations in functions and lines of code based on causal profiling

Parallelism API support
========================================

* HIP
* HSA
* Pthreads
* MPI
* Kokkos-Tools (KokkosP)
* OpenMP-Tools (OMPT)

GPU metrics
========================================

* GPU hardware counters
* HIP API tracing
* HIP kernel tracing
* HSA API tracing
* HSA operation tracing
* System-level sampling (via rocm-smi)

  * Memory usage
  * Power usage
  * Temperature
  * Utilization

CPU metrics
========================================

* CPU hardware counters sampling and profiles
* CPU frequency sampling
* Various timing metrics

  * Wall time
  * CPU time (process and thread)
  * CPU utilization (process and thread)
  * User CPU time
  * Kernel CPU time

* Various memory metrics

  * High-water mark (sampling and profiles)
  * Memory page allocation
  * Virtual memory usage

* Network statistics
* I/O metrics
* Many others

Third-party API support
========================================

* TAU
* LIKWID
* Caliper
* CrayPAT
* VTune
* NVTX
* ROCTX

Omnitrace use cases
========================================

When analyzing the performance of an application, do NOT 
assume you know where the performance bottlenecks are
and why they are happening. Omnitrace is a tool for the analyzing the entire 
application and its performance. It is
ideal for characterizing where optimization would have the greatest impact 
on an end-to-end run of the application and for
viewing what else is happening on the system during a performance bottleneck.

When GPUs are involved, there is a tendency to assume that 
the quickest path to performance improvement is minimizing
the runtime of the GPU kernels. This is a highly flawed assumption. 
If you optimize the runtime of a kernel from one millisecond
to 1 microsecond (1000x speed-up) but the original application never 
spent time waiting for kernel(s) to complete,
there would be no statistically significant reduction in the end-to-end 
runtime of your application. In other words, it does not matter
how fast or slow the code on GPU is if the application has a  
bottleneck on waiting on the GPU.

Use Omnitrace to obtain a high-level view of the entire application. Use it 
to determine where the performance bottlenecks are and
obtain clues to why these bottlenecks are happening. Rather than worrying about kernel
performance, start your investigation with Omnitrace, which characterizes the
broad picture.

.. note::

   AMD is working on a separate tool that provides insight into the execution of individual kernels
   on the GPU.

In terms of CPU analysis, Omnitrace does not target any specific vendor. 
It works just as well on AMD and non-AMD CPUs.
With regard to the GPU, Omnitrace is currently restricted to HIP and HSA APIs 
and kernels running on AMD GPUs.