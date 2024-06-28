.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

***********************
Omnitrace documentation
***********************

Omnitrace is a multi-purpose analysis tool for profiling and tracing applications
running on the CPU and GPU. It supports dynamic binary instrumentation,
call-stack sampling, and causal profiling, along with a full set of configuration
options. Omnitrace is now part of the AMD ROCm (TM) Software stack. To learn more, see :doc:`what-is-omnitrace`

The code is open and hosted at `<https://github.com/ROCm/omnitrace>`_.


.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Quick start <./install/quick-start>`
    * :doc:`Omnitrace installation <./install/install>`


The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Tutorials

    * `GitHub examples <https://github.com/ROCm/omnitrace/tree/main/examples>`_
    * :doc:`YouTube tutorials <./tutorials/you-tube>`

  .. grid-item-card:: How to

    * :doc:`Configuring and validating the Omnitrace environment <./how-to/configuring-validating-environment>`
    * :doc:`Configuring runtime options <./how-to/configuring-runtime-options>`
    * :doc:`Sampling the call stack <./how-to/sampling-call-stack>`
    * :doc:`Instrumenting and rewriting a binary application <./how-to/instrumenting-rewriting-binary-application>`
    * :doc:`Performing causal profiling <./how-to/performing-causal-profiling>`
    * :doc:`Understanding the Omnitrace output <./how-to/understanding-omnitrace-output>`
    * :doc:`Profiling Python scripts <./how-to/profiling-python-scripts>`
    * :doc:`Troubleshooting Omnitrace on Linux <./how-to/troubleshooting-omnitrace-linux>`
    * :doc:`Using the Omnitrace API <./how-to/using-omnitrace-api>`

  .. grid-item-card:: Conceptual

    * :doc:`How Omnitrace works <./conceptual/how-omnitrace-works>`
    * :doc:`The Omnitrace feature set <./conceptual/omnitrace-feature-set>`
  
  .. grid-item-card:: Reference

    * :doc:`API library <../doxygen/html/files>`
    * :doc:`Functions <../doxygen/html/globals>`
    * :doc:`Data structures <../doxygen/html/annotated>`
    * :doc:`Compiling Omnitrace with CMake <./reference/compiling-cmake>`
    * :doc:`Development guide <./reference/development-guide>`


To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.