import os
import sys
import unittest

import helpme_py as pme
import numpy as np

def print_results(label, e, f=None, v=None):
    np.set_printoptions(precision=10, linewidth=100)
    print(label)
    print("Energy = {:16.10f}".format(e))
    if f is not None:
        print("Forces:")
        print(f)
    if v is not None:
        print("Virial:")
        print(v)
    print()


class TestHelpme(unittest.TestCase):
    def setUp(self):
        self.numThreads = os.getenv("HELPME_TESTS_NTHREADS", default=1)
        print(f"Num Threads: {self.numThreads}")
        self.toleranceD = 1e-8
        self.toleranceF = 1e-4
        self.expectedSelfEnergy = -117.2820123482703
        self.expectedDirectEnergy = -186.46739220834684
        self.expectedDirectForces = np.array(
            [[ -4.80050413, -6.08572681, 160.84170309],
             [ 17.74389872,  1.2403298 , -79.64693998],
             [-14.76898896,  2.62295853, -78.65843964],
             [ -3.62886129, -3.43179709, 158.73687086],
             [ 19.97029815,  4.31212038, -80.43681725],
             [-14.51584248,  1.34211519, -80.83637708]],
        )
        self.expectedDirectVirial = np.array(
            [[29.84832541, -3.65118874,   -4.44487697,  
               4.77817668,  5.07264694, -314.50592701]],
        )
        self.expectedAdjustedEnergy = 111.17508883
        self.expectedAdjustedForces = np.array(
            [[ 0.61218116,  0.75465473, -6.56630909],
             [-0.5129453 , -0.45074263,  2.66527422],
             [-0.35134478, -0.55407743,  2.73966459],
             [ 1.15661774,  1.06775516, -5.34200372],
             [-0.41247541, -0.46434949,  3.33860599],
             [-0.49203341, -0.35324034,  3.164768  ]],
        )
        self.expectedAdjustedVirial = np.array(
            [[-0.54523911, -0.50421785, -0.50033067,
              -2.27301675, -2.32274056,  9.58557225]],
        )
        self.expectedAdjustedPotential= np.array(
            [[-1.27424325,  0.73403016,  0.90486179, -7.87327229],
             [-8.06710665,  1.23008466,  1.08091758, -6.39154489],
             [-9.12628039,  0.84255342,  1.32872286, -6.56993907],
             [10.44084805,  1.38683183,  1.28028196, -6.40528024],
             [ 3.67569982,  0.98914967,  1.11354793, -8.00624939],
             [ 2.56109575,  1.17993623,  0.84709913, -7.58937171]],
        )
        self.expectedReciprocalEnergy = 5.864957414
        self.expectedReciprocalForces = np.array(
            [[-1.20630693, -1.49522843, 12.65589187],
             [ 1.00695882,  0.88956328, -5.08428301],
             [ 0.69297661,  1.09547848, -5.22771480],
             [-2.28988057, -2.10832506, 10.18914165],
             [ 0.81915340,  0.92013663, -6.43738026],
             [ 0.97696467,  0.69833887, -6.09492437]]
        )
        self.expectedReciprocalVirial = np.array(
            [[0.65613058, 0.49091167, 0.61109732,
              2.26906257, 2.31925449, -10.04901641]],
        )
        self.expectedReciprocalPotential = np.array(
            [[ 1.18119329, -0.72320559, -0.89641992, 7.58746515],
             [ 7.69247982, -1.20738468, -1.06662264, 6.09626260],
             [ 8.73449635, -0.83090721, -1.31352336, 6.26824317],
             [-9.98483179, -1.37283008, -1.26398385, 6.10859811],
             [-3.50591589, -0.98219832, -1.10328133, 7.71868137],
             [-2.39904512, -1.17142047, -0.83733677, 7.30806279]]
        )
        self.expectedTotalEnergy = (self.expectedSelfEnergy
                                    + self.expectedDirectEnergy
                                    + self.expectedAdjustedEnergy
                                    + self.expectedReciprocalEnergy)
        self.expectedTotalForces = (self.expectedDirectForces
                                    + self.expectedAdjustedForces
                                    + self.expectedReciprocalForces/2)
        self.expectedTotalVirial = (self.expectedDirectVirial
                                    + self.expectedAdjustedVirial
                                    + self.expectedReciprocalVirial)
        self.coordsD = np.array(
            [[ 2.00000,  2.00000, 2.00000],
             [ 2.50000,  2.00000, 3.00000],
             [ 1.50000,  2.00000, 3.00000],
             [ 0.00000,  0.00000, 0.00000],
             [ 0.50000,  0.00000, 1.00000],
             [-0.50000,  0.00000, 1.00000]],
             dtype=np.float64,
        )
        self.chargesD = np.array(
            [[-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]],
            dtype=np.float64,
        ).T
        self.coordsF = np.array(
            [[ 2.00000,  2.00000, 2.00000],
             [ 2.50000,  2.00000, 3.00000],
             [ 1.50000,  2.00000, 3.00000],
             [ 0.00000,  0.00000, 0.00000],
             [ 0.50000,  0.00000, 1.00000],
             [-0.50000,  0.00000, 1.00000]],
             dtype=np.float32,
        )
        self.chargesF = np.array(
            [[-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]],
            dtype=np.float32,
        ).T
        self.pairList = np.array(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
             [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]],
            dtype=np.int32,
        ).T
        self.pairList = np.array(
            [[0, 1],
             [0, 2],
             [0, 3],
             [0, 4],
             [0, 5],
             [1, 2],
             [1, 3],
             [1, 4],
             [1, 5],
             [2, 3],
             [2, 4],
             [2, 5],
             [3, 4],
             [3, 5],
             [4, 5]],
            dtype=np.int32,
        )

    def test_matrix(self):
        # Instantiate a matrix with types of invalid dimensions.
        with self.assertRaises(RuntimeError):
            pme.PairList(1)
        with self.assertRaises(RuntimeError):
            pme.PairList([[[1,2]]])
        with self.assertRaises(RuntimeError):
            pme.MatrixD(1)
        with self.assertRaises(RuntimeError):
            pme.MatrixD([[[1,2]]])
        with self.assertRaises(RuntimeError):
            pme.MatrixF(1)
        with self.assertRaises(RuntimeError):
            pme.MatrixF([[[1,2]]])
        if sys.version_info.minor > 11:
            matD = pme.MatrixD(np.array([[0., 1.]]))
            self.assertEqual(matD.__buffer__(0).format, "B")
            matF = pme.MatrixF(np.array([[0., 1.]]))
            self.assertEqual(matF.__buffer__(0).format, "B")
            pl = pme.PairList(np.array([[0., 1.]]))
            self.assertEqual(pl.__buffer__(0).format, "B")

    def test_double(self):
        # Instantiate double precision PME object.
        pmeD = pme.PMEInstanceD()
        pmeD.setup(1, 0.3, 5, 32, 32, 32, 332.0716, self.numThreads)
        mat = pme.MatrixD
        pair = pme.PairList
        pmeD.set_lattice_vectors(20, 20, 20, 90, 90, 90, pme.LatticeType.XAligned)

        # Perform tests for self energy.
        energy = 0
        # Compute just the energy.
        print_results("Before pmeD.compute_E_self", energy)
        energy = pmeD.compute_E_self(0, mat(self.chargesD))
        print_results("After pmeD.compute_E_self", energy)
        print("\n")

        self.assertTrue(np.allclose([self.expectedSelfEnergy], [energy], atol=self.toleranceD))

        # Perform tests for direct real-space interaction.
        energy = 0
        forces = np.zeros((6,3),dtype=np.float64)
        virial = np.zeros((1,6),dtype=np.float64)
        # Compute just the energy.
        print_results("Before pmeD.compute_E_dir", energy, forces, virial)
        energy = pmeD.compute_E_dir(pair(self.pairList), 0, mat(self.chargesD), mat(self.coordsD))
        print_results("After pmeD.compute_E_dir", energy, forces, virial)
        # Compute the energy and forces.
        energy = pmeD.compute_EF_dir(pair(self.pairList), 0, mat(self.chargesD), mat(self.coordsD), mat(forces))
        print_results("After pmeD.compute_EF_dir", energy, forces, virial)
        # Compute the energy, forces and virial.
        energy = pmeD.compute_EFV_dir(pair(self.pairList), 0, mat(self.chargesD), mat(self.coordsD), mat(forces), mat(virial))
        print_results("After pmeD.compute_EFV_dir", energy, forces, virial)
        print("\n")

        self.assertTrue(np.allclose([self.expectedDirectEnergy], [energy], atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedDirectForces, forces/2, atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedDirectVirial, virial, atol=self.toleranceD))

        # Perform tests for adjusted real-space interaction.
        energy = 0
        forces = np.zeros((6,3),dtype=np.float64)
        virial = np.zeros((1,6),dtype=np.float64)
        potentialAndGradient = np.zeros((6,4),dtype=np.float64)
        # Compute just the energy.
        print_results("Before pmeD.compute_E_adj", energy, forces, virial)
        energy = pmeD.compute_E_adj(pair(self.pairList), 0, mat(self.chargesD), mat(self.coordsD))
        print_results("After pmeD.compute_E_adj", energy, forces, virial)
        # Compute the energy and forces.
        energy = pmeD.compute_EF_adj(pair(self.pairList), 0, mat(self.chargesD), mat(self.coordsD), mat(forces))
        print_results("After pmeD.compute_EF_adj", energy, forces, virial)
        # Compute the energy, forces and virial.
        energy = pmeD.compute_EFV_adj(pair(self.pairList), 0, mat(self.chargesD), mat(self.coordsD), mat(forces), mat(virial))
        print_results("After pmeD.compute_EFV_adj", energy, forces, virial)
        # Compute the reciprocal space potential and its gradient.
        pmeD.compute_PDP_adj(0, mat(self.chargesD), mat(self.coordsD), mat(self.coordsD), mat(potentialAndGradient), False)
        print("Adjusted real-space potential and its gradient:")
        print(potentialAndGradient, "\n")

        self.assertTrue(np.allclose([self.expectedAdjustedEnergy], [energy], atol=self.toleranceD))
        print(forces/2)
        print(self.expectedAdjustedForces)
        self.assertTrue(np.allclose(self.expectedAdjustedForces, forces/2, atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedAdjustedVirial, virial, atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedAdjustedPotential, potentialAndGradient, atol=self.toleranceD))

        # Perform tests for reciprocal-space interaction.
        energy = 0
        forces = np.zeros((6,3),dtype=np.float64)
        virial = np.zeros((1,6),dtype=np.float64)
        potentialAndGradient = np.zeros((6,4),dtype=np.float64)
        # Compute just the energy.
        print_results("Before pmeD.compute_E_rec", energy, forces, virial)
        energy = pmeD.compute_E_rec(0, mat(self.chargesD), mat(self.coordsD))
        print_results("After pmeD.compute_E_rec", energy, forces, virial)
        # Compute the energy and forces.
        energy = pmeD.compute_EF_rec(0, mat(self.chargesD), mat(self.coordsD), mat(forces))
        print_results("After pmeD.compute_EF_rec", energy, forces, virial)
        # Compute the energy, forces and virial.
        energy = pmeD.compute_EFV_rec(0, mat(self.chargesD), mat(self.coordsD), mat(forces), mat(virial))
        print_results("After pmeD.compute_EFV_rec", energy, forces, virial)
        # Compute the reciprocal space potential and its gradient.
        pmeD.compute_P_rec(0, mat(self.chargesD), mat(self.coordsD), mat(self.coordsD), 1, mat(potentialAndGradient))
        print("Reciprocal-space potential and its gradient:")
        print(potentialAndGradient, "\n")

        self.assertTrue(np.allclose([self.expectedReciprocalEnergy], [energy], atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedReciprocalForces, forces, atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedReciprocalVirial, virial, atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedReciprocalPotential, potentialAndGradient, atol=self.toleranceD))

        # Perform tests for the total interaction.
        energy = 0
        forces = np.zeros((6,3),dtype=np.float64)
        virial = np.zeros((1,6),dtype=np.float64)
        # Compute just the energy.
        print_results("Before pmeD.compute_E_all", energy, forces, virial)
        energy = pmeD.compute_E_all(pair(self.pairList), pair(self.pairList), 0, mat(self.chargesD), mat(self.coordsD))
        print_results("After pmeD.compute_E_all", energy, forces, virial)
        # Compute the energy and forces.
        energy = pmeD.compute_EF_all(pair(self.pairList), pair(self.pairList), 0, mat(self.chargesD), mat(self.coordsD), mat(forces))
        print_results("After pmeD.compute_EF_all", energy, forces, virial)
        # Compute the energy, forces and virial.
        energy = pmeD.compute_EFV_all(pair(self.pairList), pair(self.pairList), 0, mat(self.chargesD), mat(self.coordsD), mat(forces), mat(virial))
        print_results("After pmeD.compute_EFV_all", energy, forces, virial)
        print("\n")

        self.assertTrue(np.allclose([self.expectedTotalEnergy], [energy], atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedTotalForces, forces/2, atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedTotalVirial, virial, atol=self.toleranceD))

    def test_double_compressed(self):
        # Instatiate double precision PME object
        energy = 0
        forces = np.zeros((6,3),dtype=np.float64)
        virial = np.zeros((1,6),dtype=np.float64)
        potentialAndGradient = np.zeros((6,4),dtype=np.float64)

        pmeD = pme.PMEInstanceD()
        pmeD.setup_compressed(1, 0.3, 5, 32, 32, 32, 9, 9, 9, 332.0716, self.numThreads)
        mat = pme.MatrixD
        pmeD.set_lattice_vectors(20, 20, 20, 90, 90, 90, pme.LatticeType.XAligned)
        # Compute just the energy
        print_results("Before pmeD.compute_E_rec", energy, forces, virial)
        energy = pmeD.compute_E_rec(0, mat(self.chargesD), mat(self.coordsD))
        print_results("After pmeD.compute_E_rec", energy, forces, virial)
        # Compute the energy and forces
        energy = pmeD.compute_EF_rec(0, mat(self.chargesD), mat(self.coordsD), mat(forces))
        print_results("After pmeD.compute_EF_rec", energy, forces, virial)
        # Compute the energy, forces and virial
        energy = pmeD.compute_EFV_rec(0, mat(self.chargesD), mat(self.coordsD), mat(forces), mat(virial))
        print_results("After pmeD.compute_EFV_rec", energy, forces, virial)
        # Compute the reciprocal space potential and its gradient
        pmeD.compute_P_rec(0, mat(self.chargesD), mat(self.coordsD), mat(self.coordsD), 1, mat(potentialAndGradient))
        print("Reciprocal-space potential and its gradient:")
        print(potentialAndGradient, "\n")

        self.assertTrue(np.allclose([self.expectedReciprocalEnergy], [energy], atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedReciprocalForces, forces, atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedReciprocalVirial, virial, atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedReciprocalPotential, potentialAndGradient, atol=self.toleranceD))


    def test_float(self):
        # Instatiate single precision PME object.
        pmeF = pme.PMEInstanceF()
        pmeF.setup(1, 0.3, 5, 32, 32, 32, 332.0716, self.numThreads)
        mat = pme.MatrixF
        pair = pme.PairList
        pmeF.set_lattice_vectors(20, 20, 20, 90, 90, 90, pme.LatticeType.XAligned)

        # Perform tests for self energy.
        energy = 0
        # Compute just the energy.
        print_results("Before pmeF.compute_E_self", energy)
        energy = pmeF.compute_E_self(0, mat(self.chargesD))
        print_results("After pmeF.compute_E_self", energy)
        print("\n")

        self.assertTrue(np.allclose([self.expectedSelfEnergy], [energy], atol=self.toleranceF))

        # Perform tests for direct real-space interaction.
        energy = 0
        forces = np.zeros((6,3),dtype=np.float32)
        virial = np.zeros((1,6),dtype=np.float32)
        # Compute just the energy.
        print_results("Before pmeF.compute_E_dir", energy, forces, virial)
        energy = pmeF.compute_E_dir(pair(self.pairList), 0, mat(self.chargesF), mat(self.coordsF))
        print_results("After pmeF.compute_E_dir", energy, forces, virial)
        # Compute the energy and forces.
        energy = pmeF.compute_EF_dir(pair(self.pairList), 0, mat(self.chargesF), mat(self.coordsF), mat(forces))
        print_results("After pmeF.compute_EF_dir", energy, forces, virial)
        # Compute the energy, forces and virial.
        energy = pmeF.compute_EFV_dir(pair(self.pairList), 0, mat(self.chargesF), mat(self.coordsF), mat(forces), mat(virial))
        print_results("After pmeF.compute_EFV_dir", energy, forces, virial)
        print("\n")

        self.assertTrue(np.allclose([self.expectedDirectEnergy], [energy], atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedDirectForces, forces/2, atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedDirectVirial, virial, atol=self.toleranceF))

        # Perform tests for adjusted real-space interaction.
        energy = 0
        forces = np.zeros((6,3),dtype=np.float32)
        virial = np.zeros((1,6),dtype=np.float32)
        potentialAndGradient = np.zeros((6,4),dtype=np.float32)
        # Compute just the energy.
        print_results("Before pmeF.compute_E_adj", energy, forces, virial)
        energy = pmeF.compute_E_adj(pair(self.pairList), 0, mat(self.chargesF), mat(self.coordsF))
        print_results("After pmeF.compute_E_adj", energy, forces, virial)
        # Compute the energy and forces.
        energy = pmeF.compute_EF_adj(pair(self.pairList), 0, mat(self.chargesF), mat(self.coordsF), mat(forces))
        print_results("After pmeF.compute_EF_adj", energy, forces, virial)
        # Compute the energy, forces and virial.
        energy = pmeF.compute_EFV_adj(pair(self.pairList), 0, mat(self.chargesF), mat(self.coordsF), mat(forces), mat(virial))
        print_results("After pmeF.compute_EFV_adj", energy, forces, virial)
        # Compute the reciprocal space potential and its gradient.
        pmeF.compute_PDP_adj(0, mat(self.chargesF), mat(self.coordsF), mat(self.coordsF), mat(potentialAndGradient), False)
        print("Adjusted real-space potential and its gradient:")
        print(potentialAndGradient, "\n")

        self.assertTrue(np.allclose([self.expectedAdjustedEnergy], [energy], atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedAdjustedForces, forces/2, atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedAdjustedVirial, virial, atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedAdjustedPotential, potentialAndGradient, atol=self.toleranceF))

        # Perform tests for reciprocal-space interaction.
        energy = 0
        forces = np.zeros((6,3),dtype=np.float32)
        virial = np.zeros((1,6),dtype=np.float32)
        potentialAndGradient = np.zeros((6,4),dtype=np.float32)
        # Compute just the energy
        print_results("Before pmeF.compute_E_rec", energy, forces, virial)
        energy = pmeF.compute_E_rec(0, mat(self.chargesF), mat(self.coordsF))
        print_results("After pmeF.compute_E_rec", energy, forces, virial)
        # Compute the energy and forces
        energy = pmeF.compute_EF_rec(0, mat(self.chargesF), mat(self.coordsF), mat(forces))
        print_results("After pmeF.compute_EF_rec", energy, forces, virial)
        # Compute the energy, forces and virial
        energy = pmeF.compute_EFV_rec(0, mat(self.chargesF), mat(self.coordsF), mat(forces), mat(virial))
        print_results("After pmeF.compute_EFV_rec", energy, forces, virial)
        # Compute the reciprocal space potential and its gradient
        pmeF.compute_P_rec(0, mat(self.chargesF), mat(self.coordsF), mat(self.coordsF), 1, mat(potentialAndGradient))
        print("Reciprocal-space potential and its gradient:")
        print(potentialAndGradient, "\n")

        self.assertTrue(np.allclose([self.expectedReciprocalEnergy], [energy], atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedReciprocalForces, forces, atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedReciprocalVirial, virial, atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedReciprocalPotential, potentialAndGradient, atol=self.toleranceF))

        # Perform tests for the total interaction.
        energy = 0
        forces = np.zeros((6,3),dtype=np.float32)
        virial = np.zeros((1,6),dtype=np.float32)
        # Compute just the energy.
        print_results("Before pmeF.compute_E_all", energy, forces, virial)
        energy = pmeF.compute_E_all(pair(self.pairList), pair(self.pairList), 0, mat(self.chargesF), mat(self.coordsF))
        print_results("After pmeF.compute_E_all", energy, forces, virial)
        # Compute the energy and forces.
        energy = pmeF.compute_EF_all(pair(self.pairList), pair(self.pairList), 0, mat(self.chargesF), mat(self.coordsF), mat(forces))
        print_results("After pmeF.compute_EF_all", energy, forces, virial)
        # Compute the energy, forces and virial.
        energy = pmeF.compute_EFV_all(pair(self.pairList), pair(self.pairList), 0, mat(self.chargesF), mat(self.coordsF), mat(forces), mat(virial))
        print_results("After pmeF.compute_EFV_all", energy, forces, virial)
        print("\n")

        self.assertTrue(np.allclose([self.expectedTotalEnergy], [energy], atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedTotalForces, forces/2, atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedTotalVirial, virial, atol=self.toleranceF))

    def test_float_compressed(self):
        # Instatiate single precision PME object
        energy = 0
        forces = np.zeros((6,3),dtype=np.float32)
        virial = np.zeros((1,6),dtype=np.float32)
        potentialAndGradient = np.zeros((6,4),dtype=np.float32)

        pmeF = pme.PMEInstanceF()
        pmeF.setup_compressed(1, 0.3, 5, 32, 32, 32, 9, 9, 9, 332.0716, self.numThreads)
        mat = pme.MatrixF
        pmeF.set_lattice_vectors(20, 20, 20, 90, 90, 90, pme.LatticeType.XAligned)
        # Compute just the energy
        print_results("Before pmeF.compute_E_rec", energy, forces, virial)
        energy = pmeF.compute_E_rec(0, mat(self.chargesF), mat(self.coordsF))
        print_results("After pmeF.compute_E_rec", energy, forces, virial)
        # Compute the energy and forces
        energy = pmeF.compute_EF_rec(0, mat(self.chargesF), mat(self.coordsF), mat(forces))
        print_results("After pmeF.compute_EF_rec", energy, forces, virial)
        # Compute the energy, forces and virial
        energy = pmeF.compute_EFV_rec(0, mat(self.chargesF), mat(self.coordsF), mat(forces), mat(virial))
        print_results("After pmeF.compute_EFV_rec", energy, forces, virial)
        # Compute the reciprocal space potential and its gradient
        pmeF.compute_P_rec(0, mat(self.chargesF), mat(self.coordsF), mat(self.coordsF), 1, mat(potentialAndGradient))
        print("Reciprocal-space potential and its gradient:")
        print(potentialAndGradient, "\n")

        self.assertTrue(np.allclose([self.expectedReciprocalEnergy], [energy], atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedReciprocalForces, forces, atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedReciprocalVirial, virial, atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedReciprocalPotential, potentialAndGradient, atol=self.toleranceF))

if __name__ == '__main__':
    unittest.main()
