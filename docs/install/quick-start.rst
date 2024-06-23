.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

*************************************
Omnitrace quick start installation
*************************************

To install Omnitrace, download the `Omnitrace installer <https://github.com/ROCm/omnitrace/releases/latest/download/omnitrace-install.py>`_ 
and specify ``--prefix <install-directory>``. This script attempts to auto-detect 
the appropriate OS distribution and version. To include AMD ROCm (TM) Software support, 
specify ``--rocm X.Y``, where ``X`` is the ROCm major
version and ``Y`` is the ROCm minor version, for example, ``--rocm 6.2``.

.. code:: shell

   wget https://github.com/ROCm/omnitrace/releases/latest/download/omnitrace-install.py
   python3 ./omnitrace-install.py --prefix /opt/omnitrace --rocm 6.2

This script supports installation on Ubuntu, OpenSUSE, Red Hat, Debian, CentOS, and Fedora.
If the target OS is compatible with one of the operating system versions listed here,
specify ``-d <DISTRO> -v <VERSION>``. For example, if the OS is compatible with Ubuntu 18.04, pass
``-d ubuntu -v 18.04`` to the script.

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

Release links
========================================

To review and install either the current Omnitrace release or earlier releases, use these links:

* `Latest Omnitrace Release <https://github.com/ROCm/omnitrace/releases/latest>`_ 
* `All Omnitrace Releases <https://github.com/ROCm/omnitrace/releases>`_ 