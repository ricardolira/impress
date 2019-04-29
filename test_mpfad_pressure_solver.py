"""This is the mpfad-3D preprocessor Tests."""
import unittest
import numpy as np
from mesh_preprocessor import Mesh
from mpfad import MpfaD

class PressureSolverTest(unittest.TestCase):

    def setUp(self):
        K_1 = [1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0]
        self.mesh1 = Mesh('mesh/mpfad_pressure_test.h5m', dim=3)
        self.mesh1.set_boundary_conditions('Dirichlet', {101: 0.0})
        self.mesh1.set_boundary_conditions('Neumann', {201: 0.0})
        self.mesh1.set_material_prop('permeability', {1: K_1})
        self.mpfad1 = MpfaD(self.mesh1)

    def tearDown(self):
        self.mpfad = None

    def test_assemble_problem_on_triangular_faces_only(self):
        b_verts = self.mesh1.b_verts




if __name__== "__main__":
    unittest.main()
