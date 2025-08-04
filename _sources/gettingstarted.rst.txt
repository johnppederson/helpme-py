.. # BEGINLICENSE
.. #
.. # This file is part of helPME, which is distributed under the BSD 3-clause license,
.. # as described in the LICENSE file in the top level directory of this project.
.. #
.. # Author: Andrew C. Simmonett
.. #
.. # ENDLICENSE

.. _`sec:gettingstarted`:

===============
Getting Started
===============

Installation
============

Python installation
-------------------

The easiest way to install helPME-py is using ``pip``:

.. code-block:: bash
    
    $ pip install helpme-py

Alternatively, you can clone the repository and build/install using ``pip``:

.. code-block:: bash
    
    $ git clone https://github.com/johnppederson/helpme-py
    $ cd helpme-py
    $ pip install . -v

The Python API is demonstrated in :testcase:`fullexample.py`.

Compiling from source
---------------------

|helPME| uses CMake for configuring and building.  It's generally a good idea
to build in a directory other than the source directory, *e.g.*, to build using
4 cores:

.. code-block:: bash

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make -j4
    $ ctest

On some systems, the default setup detected by CMake may not be what's
appropriate; always check the information provided by CMake to ensure that the
desired compilers / tools are detected.  Here's a much more complete example
that specifies MPI compiler wrappers to use, as well as the installation
directory:

.. code-block:: bash

    $ CC=mpicc CXX=mpicxx FFTWROOT=/path/to/fftw/installation cmake .. -DPYTHON_EXECUTABLE=/path/to/python -DCMAKE_INSTALL_PREFIX=/path/to/install/helpme/into
    $ make -j4
    $ ctest
    $ make docs
    $ make install

Installation may not be necessary, depending on your choice of language and
build setup.  Here's a quick overview of what is needed for each language
choice.  Examples of the library's usage can be found in the :repo:`test`
directory.

C++
---

Because the library is written in C++ this is straightforward.  After building
and installing, simply adding the include directory to the compiler include
path using the ``-I`` flag will allow the library to be used as demonstrated in
:testcase:`fullexample.cpp` for OpenMP parallel or
:testcase:`fullexample_parallel.cpp` for hybrid OpenMP/MPI parallel.

C
-

To use the library from C, follow the `Compiling from source`_ instructions and
ensure that the include directory is in the C compiler's include path and that
the |helPME| library from `lib` is linked.  From there, the library can be used
by including the ``helpme.h`` header and calling the API, as demonstrated in
:testcase:`fullexample.c` for OpenMP parallel or
:testcase:`fullexample_parallel.c` for hybrid OpenMP / MPI parallel.
