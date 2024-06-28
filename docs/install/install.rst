.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

*************************************
Omnitrace installation
*************************************

The following information builds on the guidelines in the :doc:`Quick start <./quick-start>` guide.
It covers how to install `Omnitrace <https://github.com/ROCm/omnitrace>`_ from source or a binary distribution,
as well as post-installation steps.

Operating system support
========================================

Omnitrace is only supported on Linux. The following distributions are tested:

* Ubuntu 18.04
* Ubuntu 20.04
* Ubuntu 22.04
* OpenSUSE 15.2
* OpenSUSE 15.3
* OpenSUSE 15.4
* Red Hat 8.7
* Red Hat 9.0
* Red Hat 9.1

Other OS distributions might be supported but are not tested.

Identifying the operating system
-----------------------------------

If you are unsure of the operating system and version, the ``/etc/os-release`` and 
``/usr/lib/os-release`` files contain operating system identification data for Linux systems.

.. code-block:: shell

   $ cat /etc/os-release

.. code-block:: shell

   NAME="Ubuntu"
   VERSION="20.04.4 LTS (Focal Fossa)"
   ID=ubuntu
   ...
   VERSION_ID="20.04"
   ...

The relevant fields are ``ID`` and the ``VERSION_ID``.

Architecture
========================================

With regards to instrumentation, at present only AMD64 (x86_64) architectures are tested. However,
Dyninst supports several more architectures and Omnitrace instrumentation may support other
CPU architectures such as aarch64 and ppc64.
Other modes of use, such as sampling and causal profiling, are not dependent on Dyninst and therefore
might be more portable.

Installing Omnitrace from binary distributions
================================================

Every Omnitrace release provides binary installer scripts of the form:

.. code-block:: shell

   omnitrace-{VERSION}-{OS_DISTRIB}-{OS_VERSION}[-ROCm-{ROCM_VERSION}[-{EXTRA}]].sh

For example,

.. code-block:: shell

   omnitrace-1.0.0-ubuntu-18.04-OMPT-PAPI-Python3.sh
   omnitrace-1.0.0-ubuntu-18.04-ROCm-405000-OMPT-PAPI-Python3.sh
   ...
   omnitrace-1.0.0-ubuntu-20.04-ROCm-50000-OMPT-PAPI-Python3.sh

Any of the ``EXTRA`` fields with a CMake build option 
(for example, PAPI, as referenced in a following section) or 
with no link requirements (such as OMPT) have
self-contained support for these packages.

To install Omnitrace using a binary installer script, follow these steps:

#. Download the appropriate binary distribution

   .. code-block:: shell

      wget https://github.com/ROCm/omnitrace/releases/download/v<VERSION>/<SCRIPT>

#. Create the target installation directory

   .. code-block:: shell

      mkdir /opt/omnitrace

#. Run the installer script

   .. code-block:: shell

      ./omnitrace-1.0.0-ubuntu-18.04-ROCm-405000-OMPT-PAPI.sh --prefix=/opt/omnitrace --exclude-subdir

Installing Omnitrace from source
========================================

Omnitrace needs a GCC compiler with full support for C++17 and CMake v3.16 or higher.
The Clang compiler may be used in lieu of the GCC compiler if `Dyninst <https://github.com/dyninst/dyninst>`_  
is already installed.

Build requirements
-----------------------------------

* GCC compiler v7+
  
  * Older GCC compilers may be supported but are not tested
  * Clang compilers are generally supported for Omnitrace but not Dyninst
  
* `CMake <https://cmake.org/>`_ v3.16+

  .. note::

     * If the installed version of CMake is too old, installing a new version of CMake can be done through several methods
     * One of the easiest options is to use the python ``pip`` utility, as follows:

     .. code-block:: shell

        pip install --user 'cmake==3.18.4'
        export PATH=${HOME}/.local/bin:${PATH}

Required third-party packages
-----------------------------------

* `Dyninst <https://github.com/dyninst/dyninst>`_ for dynamic or static instrumentation. 
  Dyninst uses the following required and optional components.

  * `TBB <https://github.com/oneapi-src/oneTBB>`_ (required)
  * `Elfutils <https://sourceware.org/elfutils/>`_ (required)
  * `Libiberty <https://github.com/gcc-mirror/gcc/tree/master/libiberty>`_ (required)
  * `Boost <https://www.boost.org/>`_ (required)
  * `OpenMP <https://www.openmp.org/>`_ (optional)

* `libunwind <https://www.nongnu.org/libunwind/>`_ for call-stack sampling

Any of the third-party packages required by Dyninst, along with Dyninst itself, can be built and installed
during the Omnitrace build. The following list indicates the package, the version,
the application that requires the package (for example, Omnitrace requires Dyninst
while Dyninst requires TBB), and the CMake option to build the package alongside Omnitrace:

.. csv-table:: 
   :header: "Third-Party Library", "Minimum Version", "Required By", "CMake Option"
   :widths: 15, 10, 12, 40

   "Dyninst", "12.0", "Omnitrace", "``OMNITRACE_BUILD_DYNINST`` (default: OFF)"
   "Libunwind", "", "Omnitrace", "``OMNITRACE_BUILD_LIBUNWIND`` (default: ON)"
   "TBB", "2018.6", "Dyninst", "``DYNINST_BUILD_TBB`` (default: OFF)"
   "ElfUtils", "0.178", "Dyninst", "``DYNINST_BUILD_ELFUTILS`` (default: OFF)"
   "LibIberty",  "", "Dyninst", "``DYNINST_BUILD_LIBIBERTY`` (default: OFF)"
   "Boost",  "1.67.0", "Dyninst", "``DYNINST_BUILD_BOOST`` (default: OFF)"
   "OpenMP", "4.x", "Dyninst", ""

Optional third-party packages
-----------------------------------

* `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest>`_

  * HIP
  * Roctracer for HIP API and kernel tracing
  * ROCM-SMI for GPU monitoring
  * Rocprofiler for GPU hardware counters

* `PAPI <https://icl.utk.edu/papi/>`_
* MPI

  * ``OMNITRACE_USE_MPI`` enables full MPI support
  * ``OMNITRACE_USE_MPI_HEADERS`` enables wrapping of the dynamically-linked MPI C function calls.
    (By default, if Omnitrace cannot find an OpenMPI MPI distribution, it uses a local copy 
    of the OpenMPI ``mpi.h``.)

* Several optional third-party profiling tools supported by timemory 
  (e.g. `Caliper <https://github.com/LLNL/Caliper>`_, `TAU <https://www.cs.uoregon.edu/research/tau/home.php>`_, CrayPAT, etc.)

.. csv-table:: 
   :header: "Third-Party Library", "CMake Enable Option", "CMake Build Option"
   :widths: 15, 45, 40

   "PAPI", "``OMNITRACE_USE_PAPI`` (default: ON)", "``OMNITRACE_BUILD_PAPI`` (default: ON)"
   "MPI", "``OMNITRACE_USE_MPI`` (default: OFF)", ""
   "MPI (header-only)", "``OMNITRACE_USE_MPI_HEADERS`` (default: ON)", ""

Installing Dyninst
-----------------------------------

The easiest way to install Dyninst is alongside Omnitrace, but it can also be installed using Spack.

Building Dyninst alongside Omnitrace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install Dyninst is to configure Omnitrace with ``OMNITRACE_BUILD_DYNINST=ON``. 
Depending on the version of Ubuntu, the ``apt`` package manager may have current enough
versions of the Dyninst Boost, TBB, and LibIberty dependencies 
(i.e. ``apt-get install libtbb-dev libiberty-dev libboost-dev``). 
However, it is possible to request Dyninst to install
its dependencies via ``DYNINST_BUILD_<DEP>=ON``, e.g.:

.. code-block:: shell

   git clone https://github.com/ROCm/omnitrace.git omnitrace-source
   cmake -B omnitrace-build -DOMNITRACE_BUILD_DYNINST=ON -DDYNINST_BUILD_{TBB,ELFUTILS,BOOST,LIBIBERTY}=ON omnitrace-source

where ``-DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON`` is expanded by 
the shell to ``-DDYNINST_BUILD_TBB=ON -DDYNINST_BUILD_BOOST=ON ...``

Installing Dyninst via Spack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Spack <https://github.com/spack/spack>`_ is another option to install Dyninst and its dependencies:

.. code-block:: shell

   git clone https://github.com/spack/spack.git
   source ./spack/share/spack/setup-env.sh
   spack compiler find
   spack external find --all --not-buildable
   spack spec -I --reuse dyninst
   spack install --reuse dyninst
   spack load -r dyninst

Installing Omnitrace
-----------------------------------

Omnitrace has CMake configuration options for supporting MPI (``OMNITRACE_USE_MPI`` or 
``OMNITRACE_USE_MPI_HEADERS``), HIP kernel tracing (``OMNITRACE_USE_ROCTRACER``),
sampling ROCm devices (``OMNITRACE_USE_ROCM_SMI``), OpenMP-Tools (``OMNITRACE_USE_OMPT``), 
and hardware counters via PAPI (``OMNITRACE_USE_PAPI``), among others.
Various additional features can be enabled via the 
``TIMEMORY_USE_*`` `CMake options <https://timemory.readthedocs.io/en/develop/installation.html#cmake-options>`_.
Any ``OMNITRACE_USE_<VAL>`` option which has a corresponding ``TIMEMORY_USE_<VAL>`` 
option means that the support within timemory for this feature has been integrated
into Perfetto support for Omnitrace, e.g. ``OMNITRACE_USE_PAPI=<VAL>`` forces 
``TIMEMORY_USE_PAPI=<VAL>``. This means the data that timemory is able to collect via this package
is passed along to Perfetto and will be displayed when the `.proto` file is visualized in `the Perfetto UI <https://ui.perfetto.dev>`_.

.. code-block:: shell

   git clone https://github.com/ROCm/omnitrace.git omnitrace-source
   cmake                                       \
       -B omnitrace-build                      \
       -D CMAKE_INSTALL_PREFIX=/opt/omnitrace  \
       -D OMNITRACE_USE_HIP=ON                 \
       -D OMNITRACE_USE_ROCM_SMI=ON            \
       -D OMNITRACE_USE_ROCTRACER=ON           \
       -D OMNITRACE_USE_PYTHON=ON              \
       -D OMNITRACE_USE_OMPT=ON                \
       -D OMNITRACE_USE_MPI_HEADERS=ON         \
       -D OMNITRACE_BUILD_PAPI=ON              \
       -D OMNITRACE_BUILD_LIBUNWIND=ON         \
       -D OMNITRACE_BUILD_DYNINST=ON           \
       -D DYNINST_BUILD_TBB=ON                 \
       -D DYNINST_BUILD_BOOST=ON               \
       -D DYNINST_BUILD_ELFUTILS=ON            \
       -D DYNINST_BUILD_LIBIBERTY=ON           \
       omnitrace-source
   cmake --build omnitrace-build --target all --parallel 8
   cmake --build omnitrace-build --target install
   source /opt/omnitrace/share/omnitrace/setup-env.sh

MPI support within Omnitrace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Omnitrace can have full (``OMNITRACE_USE_MPI=ON``) or partial (``OMNITRACE_USE_MPI_HEADERS=ON``) MPI support.
The only difference between these two modes is whether or not the results collected 
via timemory and/or Perfetto can be aggregated into a single
output file during finalization. When full MPI support is enabled, combining the 
timemory results always occurs whereas combining the Perfetto
results is configurable via the ``OMNITRACE_PERFETTO_COMBINE_TRACES`` setting.

The primary benefits of partial or full MPI support are the automatic wrapping 
of MPI functions and the ability
to label output with suffixes which correspond to the ``MPI_COMM_WORLD`` rank ID 
instead of using the system process identifier (i.e. `PID`).
In general, it is recommended to use partial MPI support with the OpenMPI 
headers as this is the most portable configuration.
If full MPI support is selected, make sure your target application is built 
against the same MPI distribution as Omnitrace,
i.e. do not build Omnitrace with MPICH and use it on a target application built against OpenMPI.
If partial support is selected, the reason the OpenMPI headers are recommended instead of the MPICH headers is
because the ``MPI_COMM_WORLD`` in OpenMPI is a pointer to ``ompi_communicator_t`` (8 bytes), 
whereas ```MPI_COMM_WORLD``` in MPICH is an ``int`` (4 bytes). Building Omnitrace with partial MPI support 
and the MPICH headers and then using
Omnitrace on an application built against OpenMPI will cause a segmentation fault 
due to the value of the ``MPI_COMM_WORLD`` being narrowed
during the function wrapping before being passed along to the underlying MPI function.

Post-installation steps
========================================

After installation, you can optionally configure the Omnitrace environment.
It is recommended you test the executables to confirm Omnitrace is correctly installed.

Configure the environment
-----------------------------------

If environment modules are available and preferred, add them using these commands:

.. code-block:: shell

   module use /opt/omnitrace/share/modulefiles
   module load omnitrace/1.0.0

Alternatively, you can directly source the ``setup-env.sh`` script:

.. code-block:: shell

   source /opt/omnitrace/share/omnitrace/setup-env.sh

Test the executables
-----------------------------------

Successful execution of these commands indicates that the installation does not have any issues locating the installed libraries:

.. code-block:: shell

   omnitrace-instrument --help
   omnitrace-avail --help

.. note::

   If ROCm support was enabled, you may have to add the path to the ROCm libraries to ``LD_LIBRARY_PATH``,
   for example, ``export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}``.