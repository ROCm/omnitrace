.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Configuring and validating the environment
****************************************************

After installing the `Omnitrace <https://github.com/ROCm/omnitrace>`_ application, some additional steps are required to set up
and validate the environment.

Configuring the environment
========================================

After Omnitrace is installed, source the ``setup-env.sh`` script to prefix the 
``PATH``, ``LD_LIBRARY_PATH``, and other environment variables:

.. code-block:: shell

   source /opt/omnitrace/share/omnitrace/setup-env.sh

Alternatively, if environment modules are supported, add the ``<prefix>/share/modulefiles`` directory
to ``MODULEPATH``:

.. code-block:: shell

   module use /opt/omnitrace/share/modulefiles

.. note::
    
   As an alternative, the above line can be added to the ``${HOME}/.modulerc`` file.

After Omnitrace has been added to the ``MODULEPATH``, it can be loaded 
via ``module load omnitrace/<VERSION>`` and unloaded via ``module unload omnitrace/<VERSION>``.

.. code-block:: shell

   module load omnitrace/1.0.0
   module unload omnitrace/1.0.0

.. note::

   You may need to also add the path to the ROCm libraries to ``LD_LIBRARY_PATH``,
   for example, ``export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}``

Validating the environment configuration
========================================

If all the following commands execute successfully with the expected output, 
then you are ready to use Omnitrace:

.. code-block:: shell

   which omnitrace
   which omnitrace-avail
   which omnitrace-sample
   omnitrace-instrument --help
   omnitrace-avail --all
   omnitrace-sample --help

If Omnitrace was built with Python support, validate these additional commands:

.. code-block:: shell

   which omnitrace-python
   omnitrace-python --help