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
        self.mesh1 = Mesh('mesh/mesh_slanted_mesh.h5m', dim=3)
        self.mesh1.set_boundary_conditions('Dirichlet', {101: 0.0})
        self.mesh1.set_boundary_conditions('Neumann', {201: 0.0})
        self.mesh1.set_material_prop('permeability', {1: K_1})
        self.mesh1.set_material_prop('source', {1: 1.0})
        self.mpfad1 = MpfaD(self.mesh1)

    def tearDown(self):
        self.mpfad = None

    def test_set_boundary_conditions_on_vertex(self):
        b_verts = self.mesh1.b_verts
        b_verts_coords = self.mesh1.M.nodes.coords(b_verts)
        print(b_verts)
        b_verts_pressure = {}
        # b_verts_pressure[b_verts] = b_verts_coords[:, 0]
        # print(b_verts, b_verts_coords[:, 0])

    def test_assemble_problem_on_triangular_faces_only(self):
        b_verts = self.mesh1.b_verts
        b_verts_coords = self.mesh1.M.nodes.coords(b_verts)
        b_verts_pressure = {}
        # b_verts_pressure[b_verts] = b_verts_coords[:, 0]
        # print(b_verts, b_verts_coords[:, 0])



if __name__== "__main__":
    unittest.main()
