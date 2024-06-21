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

Data Collection Modes
========================================

* Dynamic instrumentation

  * Runtime instrumentation: Instrument executables and shared libraries at runtime
  * Binary rewriting: Generate a new executable and/or library with instrumentation built-in

* Statistical sampling: Periodic software interrupts per-thread
* Process-level sampling: Background thread records process-, system- and device-level metrics while the application executes
* Causal profiling: Quantifies the potential impact of optimizations in parallel codes

Data Analysis
========================================

* High-level summary profiles with mean/min/max/stddev statistics

  * Low overhead, memory efficient
  * Ideal for running at scale

* Comprehensive traces for every individual event/measurement
* Application speedup predictions resulting from potential optimizations in functions and lines of code (causal profiling)

Parallelism API Support
========================================

* HIP
* HSA
* Pthreads
* MPI
* Kokkos-Tools (KokkosP)
* OpenMP-Tools (OMPT)

GPU Metrics
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

CPU Metrics
========================================

* CPU hardware counters sampling and profiles
* CPU frequency sampling
* Various timing metrics

  * Wall time
  * CPU time (process and/or thread)
  * CPU utilization (process and/or thread)
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

When analyzing the performance of an application, it is always best to NOT 
assume you know where the performance bottlenecks are
and why they are happening. OmniTrace is a tool for the entire execution 
of application. It is the sort of tool which is
ideal for characterizing where optimization would have the greatest impact 
on the end-to-end execution of the application and/or
viewing what else is happening on the system during a performance bottleneck.

Especially when GPUs are involved, there is a tendency to assume that 
the quickest path to performance improvement is minimizing
the runtime of the GPU kernels. This is a highly flawed assumption. 
If you optimize the runtime of a kernel from one millisecond
to 1 microsecond (1000x speed-up) but the original application never 
spent time waiting for kernel(s) to complete,
you will see zero statistically significant speed-up in end-to-end 
runtime of your application. In other words, it does not matter
how fast or slow the code on GPU is if the application is not 
bottlenecked waiting on the GPU.

Use OmniTrace to obtain a high-level view of the entire application. Use it 
to determine where the performance bottlenecks are and
obtain clues to why these bottlenecks are happening. If you want extensive 
insight into the execution of individual kernels
on the GPU, AMD Research is working on another tool for this but you should 
start with the tool which characterizes the
broad picture: OmniTrace.

With regard to the CPU, OmniTrace does not target any specific vendor, 
it works just as well with non-AMD CPUs as with AMD CPUs.
With regard to the GPU, OmniTrace is currently restricted to the HIP and HSA APIs 
and kernels executing on AMD GPUs.