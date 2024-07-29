.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

***********************
Omnitrace documentation
***********************

Omnitrace is designed for the high-level profiling and comprehensive tracing
of applications running on the CPU or the CPU and GPU. It supports dynamic binary
instrumentation, call-stack sampling, and various other features for determining
which function and line number are currently executing. To learn more, see :doc:`what-is-omnitrace`

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
    * :doc:`Video tutorials <./tutorials/video-tutorials>`

  .. grid-item-card:: How to

    * :doc:`Configuring and validating the Omnitrace environment <./how-to/configuring-validating-environment>`
    * :doc:`Configuring runtime options <./how-to/configuring-runtime-options>`
    * :doc:`Sampling the call stack <./how-to/sampling-call-stack>`
    * :doc:`Instrumenting and rewriting a binary application <./how-to/instrumenting-rewriting-binary-application>`
    * :doc:`Performing causal profiling <./how-to/performing-causal-profiling>`
    * :doc:`Understanding the Omnitrace output <./how-to/understanding-omnitrace-output>`
    * :doc:`Profiling Python scripts <./how-to/profiling-python-scripts>`
    * :doc:`Using the Omnitrace API <./how-to/using-omnitrace-api>`
    * :doc:`General tips for using Omnitrace <./how-to/general-tips-using-omnitrace>`


  .. grid-item-card:: Conceptual

    * :doc:`Data collection modes <./conceptual/data-collection-modes>`
    * :doc:`The Omnitrace feature set <./conceptual/omnitrace-feature-set>`
  
  .. grid-item-card:: Reference

    * :doc:`Development guide <./reference/development-guide>`
    * :doc:`Omnitrace glossary <./reference/omnitrace-glossary>`
    * :doc:`API library <./doxygen/html/files>`
    * :doc:`Class member functions <./doxygen/html/functions>`
    * :doc:`Globals <./doxygen/html/globals>`
    * :doc:`Classes, structures, and interfaces <./doxygen/html/annotated>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.