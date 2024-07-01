.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Compiling Omnitrace with CMake
****************************************************

To compile and build Omnitrace, use the following CMake and ``g++`` directives.

CMake
========================================

.. code-block:: cmake

   find_package(omnitrace REQUIRED COMPONENTS user)
   add_executable(foo foo.cpp)
   target_link_libraries(foo PRIVATE omnitrace::omnitrace-user-library)

g++ compilation
========================================

Assuming Omnitrace is installed in ``/opt/omnitrace``, use the ``g++`` compiler 
to build the application.

.. code-block:: shell

   g++ -I/opt/omnitrace foo.cpp -o foo -lomnitrace-user
