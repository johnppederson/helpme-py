// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE
#include "helpme.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <cstdint>

namespace py = pybind11;

namespace {
template <typename Real>
void declarePMEInstance(py::module& mod, std::string const& suffix) {
    using PME = helpme::PMEInstance<Real>;
    using Matrix = helpme::Matrix<Real>;

    py::class_<Matrix> mat(mod, ("Matrix" + suffix).c_str(), py::buffer_protocol(),
                           R"pbdoc(
                           A matrix object that accepts data from NumPy arrays.

                           The input should be NxN.

                           Args:
                               array: The Python-side 2D matrix object.
                           )pbdoc");
    mat.def(py::init([](py::array_t<Real, py::array::forcecast> b) {
                /* Request a buffer descriptor from Python to construct a matrix from numpy arrays directly */
                py::buffer_info info = b.request();
                if (info.format != py::format_descriptor<Real>::format())
                    throw std::runtime_error("Incompatible format used to create Matrix py-side.");
                if (info.ndim != 2) throw std::runtime_error("Matrix object should have 2 dimensions.");
                return Matrix(static_cast<Real*>(info.ptr), info.shape[0], info.shape[1]);
            }),
            py::arg("array"), py::keep_alive<1, 2>());
    mat.def_buffer([](Matrix& m) -> py::buffer_info {
        return py::buffer_info(m[0],                                    /* Pointer to buffer */
                               sizeof(Real),                            /* Size of one scalar */
                               py::format_descriptor<Real>::format(),   /* Python struct-style format descriptor */
                               2,                                       /* Number of dimensions */
                               {m.nRows(), m.nCols()},                  /* Buffer dimensions */
                               {sizeof(Real) * m.nCols(), sizeof(Real)} /* Strides (in bytes) for each index */
        );
    });

    py::class_<PME> pme(mod, ("PMEInstance" + suffix).c_str(),
                        R"pbdoc(An object that performs various calculations.)pbdoc");
    pme.def(py::init<>());
    pme.def("setup", &PME::setup, py::arg("r_power"), py::arg("kappa"), py::arg("spline_order"), py::arg("a_dim"),
            py::arg("b_dim"), py::arg("c_dim"), py::arg("scale_factor"), py::arg("n_threads"),
            R"pbdoc(
            Initializes this object for a PME calculation.
     
            This method may be called repeatedly without compromising
            performance.
     
            Args:
                r_power: The exponent of the (inverse) distance kernel
                    (e.g. 1 for Coulomb, 6 for attractive dispersion).
                kappa: The attenuation parameter in units inverse of those
                    used to specify coordinates.
                spline_order: The order of B-spline; must be at least
                    (2 + max. multipole order + deriv. level needed).
                a_dim: The dimension of the FFT grid along the A axis.
                b_dim: The dimension of the FFT grid along the B axis.
                c_dim: The dimension of the FFT grid along the C axis.
                scale_factor: A scale factor to be applied to all
                    computed energies and derivatives thereof (e.g. the
                    1 / [4 pi epslion0] for Coulomb calculations).
                n_threads: The maximum number of threads to use for each
                    MPI instance; if set to 0 all available threads are
                    used.
            )pbdoc");
    pme.def("setup_compressed", &PME::setupCompressed, py::arg("r_power"), py::arg("kappa"), py::arg("spline_order"),
            py::arg("a_dim"), py::arg("b_dim"), py::arg("c_dim"), py::arg("max_ka"), py::arg("max_kb"),
            py::arg("max_kc"), py::arg("scale_factor"), py::arg("n_threads"),
            R"pbdoc(
            Initializes this object for a compressed PME calculation.
            
            This may be called repeatedly without compromising
            performance.

            Args:
                r_power: The exponent of the (inverse) distance kernel
                    (e.g. 1 for Coulomb, 6 for attractive dispersion).
                kappa: The attenuation parameter in units inverse of those
                    used to specify coordinates.
                spline_order: The order of B-spline; must be at least
                    (2 + max. multipole order + deriv. level needed).
                a_dim: The dimension of the FFT grid along the A axis.
                b_dim: The dimension of the FFT grid along the B axis.
                c_dim: The dimension of the FFT grid along the C axis.
                max_ka: The maximum K value in the reciprocal sum along
                    the A axis.
                max_kb: The maximum K value in the reciprocal sum along
                    the B axis.
                max_kc: The maximum K value in the reciprocal sum along
                    the C axis.
                scale_factor: A scale factor to be applied to all
                    computed energies and derivatives thereof (e.g. the
                    1 / [4 pi epslion0] for Coulomb calculations).
                n_threads: The maximum number of threads to use for each
                    MPI instance; if set to 0 all available threads are
                    used.
            )pbdoc");
    pme.def("set_lattice_vectors", &PME::setLatticeVectors, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("alpha"),
            py::arg("beta"), py::arg("gamma"), py::arg("orientation"),
            R"pbdoc(
            Sets the unit cell lattice vectors.

            The units of the length parameters should be consistent with
            those used to specify coordinates.

            Args:
                a: The first length lattice parameter.
                b: The second length lattice parameter.
                c: The third length lattice parameter.
                alpha: The first angle lattice parameter, in degrees.
                beta: The second angle lattice parameter, in degrees.
                gamma: The third angle lattice parameter, in degrees.
                orientation: The orientation of the lattice vectors.
            )pbdoc");
    pme.def("compute_E_self", &PME::computeESlf, py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the Ewald self interaction energy.

            Args:
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python

                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz

            Returns:
                The self energy.
            )pbdoc");
    pme.def("compute_E_dir", &PME::computeEDir, py::arg("pair_list").noconvert(), py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the direct real-space energy.

            This is provided mostly for debugging and testing purposes;
            generally, the host program should provide the pairwise
            interactions.

            Args:
                pair_list: A dense list of atom pairs, ordered like i1,
                    j1, i2, j2, i3, j3, ... iN, jN.
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.

            Returns:
                The direct real-space energy.
            )pbdoc");
    pme.def("compute_EF_dir", &PME::computeEFDir, py::arg("pair_list").noconvert(), py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("forces").noconvert(), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the direct real-space energy and forces.

            This is provided mostly for debugging and testing purposes;
            generally, the host program should provide the pairwise
            interactions.

            Args:
                pair_list: A dense list of atom pairs, ordered like i1,
                    j1, i2, j2, i3, j3, ... iN, jN.
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                forces: An Nx3 matrix of the forces.  This matrix is
                    incremented, not assigned.

            Returns:
                The direct real-space energy.
            )pbdoc");
    pme.def("compute_EFV_dir", &PME::computeEFVDir, py::arg("pair_list").noconvert(), py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("forces").noconvert(), py::arg("virial").noconvert(),
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the direct real-space energy, forces, and virial.

            This is provided mostly for debugging and testing purposes;
            generally, the host program should provide the pairwise
            interactions.

            Args:
                pair_list: A dense list of atom pairs, ordered like i1,
                    j1, i2, j2, i3, j3, ... iN, jN.
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python

                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                forces: An Nx3 matrix of the forces.  This matrix is
                    incremented, not assigned.
                virial: A vector of length 6 containing the unique virial
                    elements, in the order XX XY YY XZ YZ ZZ.  This
                    vector is incremented, not assigned.

            Returns:
                The direct real-space energy.
            )pbdoc");
    pme.def("compute_E_adj", &PME::computeEAdj, py::arg("pair_list").noconvert(), py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the adjusted real-space energy.

            This method extracts the energy for excluded pairs that is
            present in reciprocal space.  This is provided mostly for
            debugging and testing purposes; generally, the host program
            should provide the pairwise interactions.

            Args:
                pair_list: A dense list of atom pairs, ordered like i1,
                    j1, i2, j2, i3, j3, ... iN, jN.
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.

            Returns:
                The adjusted real-space energy.
            )pbdoc");
    pme.def("compute_EF_adj", &PME::computeEFAdj, py::arg("pair_list").noconvert(), py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("forces").noconvert(), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the adjusted real-space energy and forces.

            This method extracts the energy and forces for excluded
            pairs that is present in reciprocal space.  This is provided
            mostly for debugging and testing purposes; generally, the
            host program should provide the pairwise interactions.

            Args:
                pair_list: A dense list of atom pairs, ordered like i1,
                    j1, i2, j2, i3, j3, ... iN, jN.
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                forces: An Nx3 matrix of the forces.  This matrix is
                    incremented, not assigned.

            Returns:
                The adjusted real-space energy.
            )pbdoc");
    pme.def("compute_EFV_adj", &PME::computeEFVAdj, py::arg("pair_list").noconvert(), py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("forces").noconvert(), py::arg("virial").noconvert(),
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the adjusted real-space energy, forces, and virial.

            This method extracts the energy, forces, and virial for
            excluded pairs that is present in reciprocal space.  This is
            provided mostly for debugging and testing purposes;
            generally, the host program should provide the pairwise
            interactions.

            Args:
                pair_list: A dense list of atom pairs, ordered like i1,
                    j1, i2, j2, i3, j3, ... iN, jN.
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                forces: An Nx3 matrix of the forces.  This matrix is
                    incremented, not assigned.
                virial: A vector of length 6 containing the unique virial
                    elements, in the order XX XY YY XZ YZ ZZ.  This
                    vector is incremented, not assigned.

            Returns:
                The adjusted real-space energy.
            )pbdoc");
    pme.def("compute_E_rec", &PME::computeERec, py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the reciprocal-space energy using PME.

            Args:
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.

            Returns:
                The reciprocal-space energy.
            )pbdoc");
    pme.def("compute_EF_rec", &PME::computeEFRec, py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("forces").noconvert(), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the reciprocal-space energy and forces using PME.

            Args:
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                forces: An Nx3 matrix of the forces.  This matrix is
                    incremented, not assigned.

            Returns:
                The reciprocal-space energy.
            )pbdoc");
    pme.def("compute_EFV_rec", &PME::computeEFVRec, py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("forces").noconvert(), py::arg("virial").noconvert(),
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the reciprocal-space energy, forces, and virial using PME.

            Args:
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                forces: An Nx3 matrix of the forces.  This matrix is
                    incremented, not assigned.
                virial: A vector of length 6 containing the unique virial
                    elements, in the order XX XY YY XZ YZ ZZ.  This
                    vector is incremented, not assigned.

            Returns:
                The reciprocal-space energy.
            )pbdoc");
    pme.def("compute_E_all", &PME::computeEAll, py::arg("included_list").noconvert(),
            py::arg("excluded_list").noconvert(), py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the full (real- and reciprocal-space) energy.

            This is provided mostly for debugging and testing purposes;
            generally, the host program should provide the pairwise
            interactions.
            
            Args:
                included_list: A dense list of real-space included atom
                    pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN,
                    jN.
                excluded_list: A dense list of real-space excluded atom
                    pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN,
                    jN.
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.

            Returns:
                The full energy.
            )pbdoc");
    pme.def("compute_EF_all", &PME::computeEFAll, py::arg("included_list").noconvert(),
            py::arg("excluded_list").noconvert(), py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("forces").noconvert(), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the full (real- and reciprocal-space) energy and forces.

            This is provided mostly for debugging and testing purposes;
            generally, the host program should provide the pairwise
            interactions.
            
            Args:
                included_list: A dense list of real-space included atom
                    pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN,
                    jN.
                excluded_list: A dense list of real-space excluded atom
                    pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN,
                    jN.
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                forces: An Nx3 matrix of the forces.  This matrix is
                    incremented, not assigned.

            Returns:
                The full energy.
            )pbdoc");
    pme.def("compute_EFV_all", &PME::computeEFVAll, py::arg("included_list").noconvert(),
            py::arg("excluded_list").noconvert(), py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("forces").noconvert(), py::arg("virial").noconvert(),
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the full (real- and reciprocal-space) energy, forces, and virial.

            This is provided mostly for debugging and testing purposes;
            generally, the host program should provide the pairwise
            interactions.
            
            Args:
                included_list: A dense list of real-space included atom
                    pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN,
                    jN.
                excluded_list: A dense list of real-space excluded atom
                    pairs, ordered like i1, j1, i2, j2, i3, j3, ... iN,
                    jN.
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                forces: An Nx3 matrix of the forces.  This matrix is
                    incremented, not assigned.
                virial: A vector of length 6 containing the unique virial
                    elements, in the order XX XY YY XZ YZ ZZ.  This
                    vector is incremented, not assigned.

            Returns:
                The full energy.
            )pbdoc");
    pme.def("compute_P_rec", &PME::computePRec, py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("grid_points").noconvert(), py::arg("derivative_level"),
            py::arg("potential").noconvert(), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the reciprocal-space potential and derivatives at a set of points.

            Args:
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).  A negative value indicates that only the
                    shell with \|parameter_ang_mom\| is to be considered,
                    e.g. a value of -2 specifies that only quadrupoles
                    (and not dipoles or charges) will be provided; the
                    input matrix should have dimensions corresponding
                    only to the number of terms in this shell.
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                grid_points: An Mx3 matrix of cartesian coordinates at
                    which the potential is needed; this can be the same
                    as the coordinates.
                derivative_level: The order of the potential derivatives
                    required; 0 is the potential, 1 is (minus) the field,
                    etc.  A negative value indicates that only the
                    derivative with order \|derivative_level\| is to be
                    generated, e.g. -2 specifies that only the second
                    derivative (not the potential or its gradient) will
                    be returned as output.  The output matrix should
                    have space for only these terms, accordingly.
                potential: The array holding the potential.  This is a
                    matrix of dimensions nGridPoints x nD, where nD is
                    the derivative level requested.  See the details for
                    the parameters argument for information about
                    ordering of derivative components.  This matrix is
                    incremented, not assigned.
            )pbdoc");
    pme.def("compute_P_adj", &PME::computePAdj, py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("grid_points").noconvert(), py::arg("potential").noconvert(),
            py::arg("minimum_image"), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the adjusted real-space potential at a set of points.

            Args:
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).  A negative value indicates that only the
                    shell with \|parameter_ang_mom\| is to be considered,
                    e.g. a value of -2 specifies that only quadrupoles
                    (and not dipoles or charges) will be provided; the
                    input matrix should have dimensions corresponding
                    only to the number of terms in this shell.
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                grid_points: An Mx3 matrix of cartesian coordinates at
                    which the potential is needed; this can be the same
                    as the coordinates.
                potential: The array holding the potential.  This is a
                    matrix of dimensions nGridPoints x 1.  This matrix is
                    incremented, not assigned.
                minimum_image: Whether or not to use the minimum image
                    convection when calculating the potential.
            )pbdoc");
    pme.def("compute_PDP_adj", &PME::computePDPAdj, py::arg("parameter_ang_mom"),
            py::arg("parameters").noconvert(),  // Force pb11 not to make a copy of the incoming numpy data.
            py::arg("coordinates").noconvert(), py::arg("grid_points").noconvert(), py::arg("potential").noconvert(),
            py::arg("minimum_image"), py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Computes the adjusted real-space potential and its first-derivative at a set of points.

            Args:
                parameter_ang_mom: The angular momentum of the parameters
                    (0 for charges, C6 coefficients, 2 for quadrupoles,
                    etc.).  A negative value indicates that only the
                    shell with \|parameter_ang_mom\| is to be considered,
                    e.g. a value of -2 specifies that only quadrupoles
                    (and not dipoles or charges) will be provided; the
                    input matrix should have dimensions corresponding
                    only to the number of terms in this shell.
                parameters: The list of parameters associated with each
                    atom (charges, C6 coefficients, multipoles, etc...).
                    For a parameter with angular momentum L, a matrix of
                    dimension nAtoms x nL is expected, where nL =
                    (L+1)*(L+2)*(L+3)/6 and the fast running index nL
                    has the ordering:

                    0 X Y Z XX XY YY XZ YZ ZZ XXX XXY XYY YYY XXZ XYZ YYZ
                    XZZ YZZ ZZZ ...

                    i.e. generated by the python loops:

                    .. code-block:: python
                      
                        for L in range(maxAM+1):
                            for Lz in range(0,L+1):
                                for Ly in range(0, L - Lz + 1):
                                    Lx  = L - Ly - Lz
                coordinates: An Nx3 matrix of cartesian coordinates.
                grid_points: An Mx3 matrix of cartesian coordinates at
                    which the potential is needed; this can be the same
                    as the coordinates.
                potential: The array holding the potential.  This is a
                    matrix of dimensions nGridPoints x 4.  This matrix is
                    incremented, not assigned.
                minimum_image: Whether or not to use the minimum image
                    convection when calculating the potential.
            )pbdoc");
}
}  // namespace

PYBIND11_MODULE(helpme_py, m, py::mod_gil_not_used()) {
    py::options options;
    options.disable_enum_members_docstring();
    m.doc() = R"pbdoc(helPME-py: A Python Utility for Particle Mesh Ewald Based on helPME)pbdoc";

    using PairList = helpme::Matrix<int>;

    py::class_<PairList> plist(m, "PairList", py::buffer_protocol(),
                               R"pbdoc(
                               A matrix object for dense pair lists of atom.
                               
                               The input should be ordered like i1, j1, i2,
                               j2, i3, j3, ... iN, jN.
              
                               Args:
                                   array: The Python-side pair matrix object.
                               )pbdoc");
    plist.def(py::init([](py::array_t<int, py::array::forcecast> b) {
                  /* Request a buffer descriptor from Python to construct a matrix from numpy arrays directly */
                  py::buffer_info info = b.request();
                  if (info.itemsize != sizeof(int))
                      throw std::runtime_error("Incompatible format used to create PairList py-side.");
                  if (info.ndim != 2) throw std::runtime_error("PairList object should have 2 dimensions.");
                  return PairList(static_cast<int*>(info.ptr), info.shape[0], info.shape[1]);
              }),
              py::arg("array"), py::keep_alive<1, 2>());
    plist.def_buffer([](PairList& pl) -> py::buffer_info {
        return py::buffer_info(pl[0],                                  /* Pointer to buffer */
                               sizeof(int),                            /* Size of one scalar */
                               py::format_descriptor<int>::format(),   /* Python struct-style format descriptor */
                               2,                                      /* Number of dimensions */
                               {pl.nRows(), pl.nCols()},               /* Buffer dimensions */
                               {sizeof(int) * pl.nCols(), sizeof(int)} /* Strides (in bytes) for each index */
        );
    });

    py::enum_<helpme::LatticeType>(m, "LatticeType",
                                   R"pbdoc(
            The representation of the Lattice vectors.

            Args:
                value: Selects which representation to take, where 1
                    corresponds to an XAligned representation and 2
                    corresponds to a ShapeMatrix representation.

            Attributes:
                XAligned: Makes the A vector coincide with the X axis, the B
                    vector fall in the XY plane, and the C vector take the
                    appropriate alignment to completely define the system.
                ShapeMatrix: Enforces a symmetric representation of the lattice
                    vectors [c.f. S. Nosé and M. L. Klein, Mol. Phys. 50 1055
                    (1983)] particularly appendix C.
            )pbdoc")
        .value("ShapeMatrix", helpme::LatticeType::ShapeMatrix)
        .value("XAligned", helpme::LatticeType::XAligned);

    declarePMEInstance<double>(m, "D");
    declarePMEInstance<float>(m, "F");

    m.attr("__version__") = py::str(VERSION);
}
