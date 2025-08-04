<p align="center">
<img width=75% src="https://github.com/johnppederson/helpme-py/blob/master/docs/sphinx/_media/helpme-py.svg">
</p>

# helPME-py: A Python Utility for Particle Mesh Ewald Based on helPME

[![C](https://img.shields.io/badge/C-17+-00599C?logo=c&logoColor=white)](#)
[![C++](https://img.shields.io/badge/C++-17+-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=fff)](#)
[![License](https://img.shields.io/badge/license-BDS--3--clause-blue.svg)](https://opensource.org/license/bsd-3-clause)

[![Build](https://github.com/johnppederson/helpme-py/actions/workflows/test_and_coverage.yml/badge.svg)](#)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/johnppederson/4c2e0c473552016c70f6cc0f0a44a9c1/raw/coverage.json)](#)
[![Deployment](https://github.com/johnppederson/helpme-py/actions/workflows/build_and_deploy.yml/badge.svg)](#)
[![Docs](https://github.com/johnppederson/helpme-py/actions/workflows/docs.yml/badge.svg)](#)

## About

**h**elPME: an **e**fficient **l**ibrary for **p**article **m**esh **E**wald.
The recursive acronym is a tip of the hat to early open source software tools
and reflects the recursive algorithms that are key to helPME's support for
arbitrary operators. The library is freely available and is designed to be easy
to use, with minimal setup code required.

helPME-py is a refactorization of the helPME repository aimed to provide an
efficient and modular Python utility for performing Particle Mesh Ewald
calculations.  The build system and documentation have been revamped, and
functionality that was originally only available in the C++ layer has been
exposed with Python bindings.  The easiest way to install helPME-py is
using ``pip``:

    pip install helpme-py

Because helPME-py focuses on adapting helPME for distribution as a Python
package, several features of the original package are not available
in helPME-py, including:

* Availability as a single C++ header.
* Support for Fortran bindings.

Additionally, error function evaluations above a certain threshold value
are screened out and assumed to be equal to one, which presents a speed 
gain for some real-space calculations at the expense of some amount of
error.  The default threshold value is 6, which introduces a maximum
error on the order of 10<sup>-17</sup>, which is just at double
precision.  This value may be changed upon compilation by specifying
a new value with the ``ERF_SCREEN_VAL`` CMake option.

## Features

* Support for C++/C/Python bindings.
* Arbitrary operators including *r*<sup>-1</sup> (Coulomb) and *r*<sup>-6</sup>
  (dispersion).
* Ability to use any floating point precision mode, selectable at run time.
* Three dimensional parallel decomposition with MPI.
* OpenMP parallel threading within each MPI instance (still a work in
  progress).
* Support for arbitrary triclinic lattices and orientations thereof.
* Arbitrary order multipoles supported (still a work in progress).
* Memory for coordinates and forces is taken directly from the caller's pool,
  avoiding copies.

## License

helPME-py and helPME are distributed under the
[BSD-3-clause](https://opensource.org/licenses/BSD-3-Clause) open source
license, as described in the LICENSE file in the top level of the repository.
Some external dependencies are used that are licensed under different terms, as
enumerated below.

## Dependencies
#### Required for Building
* [CMake](https://cmake.org) required if building the code
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).
* [FFTW](http://www.fftw.org/) required to carry out fast Fourier transforms
  [(GPL license)](https://opensource.org/licenses/gpl-license).
* [pybind11](https://github.com/pybind/pybind11) required to build Python bindings
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).

#### Optional for Building
* [OpenMP](https://www.openmp.org/) for threading (licenses vary).
* [MPI](https://github.com/open-mpi/ompi) for multiprocessing
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).
* [BLAS](https://www.netlib.org/blas/) for using optimized linear
  algebra routines (licenses vary).
* [MKL](https://software.intel.com/en-us/mkl) for using optimized linear 
  algebra routines
  [(ISSL license)](https://software.intel.com/en-us/license/intel-simplified-software-license).

#### Required for Testing
* [Catch2](https://github.com/catchorg/Catch2) required for unit testing
  [(BSL license)](https://opensource.org/licenses/BSL-1.0).

#### Optional for Testing
* [gcovr](https://github.com/gcovr/gcovr) for generating coverage reports
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).

#### Required for Documentation
* [Sphinx](https://github.com/sphinx-doc/sphinx) required for generating documentation
  [(BSD-2-clause license)](https://opensource.org/licenses/BSD-2-Clause).
* [Doxygen](https://github.com/doxygen/doxygen) required for generating C++ documentation
  [(GPL-2.0 license)](https://opensource.org/license/gpl-2-0).

#### Optional for Documentation
* [LaTeX](https://www.latex-project.org/) for generating PDF documentation
  [(LPPL license)](https://www.latex-project.org/lppl/).
* [Breathe](https://github.com/breathe-doc/breathe) for rendering Doxygen output with Sphinx
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).
* [Exhale](https://github.com/svenevs/exhale) for summarizing Doxygen output rendered with Breathe
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).
* [Furo](https://github.com/pradyunsg/furo) theme for Sphinx documentation
  [(MIT license)](https://opensource.org/license/mit).

## Requirements
helPME-py is written in C++17, and should work with any modern 
C++ compiler.  There are no other requirements once the Python bindings
are built, though NumPy is recommended.

## helPME Authors
Andrew C. Simmonett (NIH)
Lori A. Burns (GA Tech)
Daniel R. Roe (NIH)
Bernard R. Brooks (NIH)

## helPME-py Authors
John P. Pederson (GA Tech)
