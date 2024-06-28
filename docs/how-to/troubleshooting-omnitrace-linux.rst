.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Troubleshooting Omnitrace on Linux
****************************************************

RHEL (Red Hat Enterprise Linux) distributions of Linux automatically enable a security feature 
called SELinux that prevents `Omnitrace <https://github.com/ROCm/omnitrace>`_ from operating successfully.
This issue applies to systems running the CentOS 9 operating system using
AMD ROCm Software version 6.1.x alongside an MI300 GPU.

Steps to reproduce the issue
========================================

The problem occurs after the following operations:

#. Pull the Omnitrace ``main`` branch

#. Build and install Omnitrace using the following instructions:

   .. code-block:: cpp

      cmake -B build -DOMNITRACE_BUILD_DYNINST=ON -DDYNINST_BUILD_{TBB,ELFUTILS,BOOST,LIBIBERTY}=ON ./
      cmake --build build --target all --parallel 8
      sudo cmake --build build --target install

#. Instrument a program, such as ``hello world`` in C++:

   .. code-block:: shell

      g++ hello.cpp -o hello
      omniperf-instrument -M sampling -o hello.instr -- ./hello

#. Run ``omnitrace-run`` with the instrumented program:

   .. code-block:: shell

      omnitrace-run -- ./hello.instr

Instead of successfully running the binary with call-stack sampling, 
Omnitrace crashes with a segmentation fault.

Temporary and permanent workarounds
========================================

A workaround for this problem can be applied permanently or to the current session:

* To avoid this problem for the duration of the current session, run the command 
  ``sudo setenforce 0`` from the shell

* For a permanent worakound, edit the configuration file using the command
  ``sudo vim /etc/sysconfig/selinux`` and change the ``SELINUX`` setting to 
  either ``Permissive`` or ``Disabled``

.. note::

   Permanently changing the SELinux settings can have security implications. 
   Ensure you review your system security settings before making any changes.
