"""This is the mpfad-3D preprocessor Tests."""
import unittest
import numpy as np
from mesh_preprocessor import Mesh
from mpfad import MpfaD


class MpfaDTest(unittest.TestCase):

    def setUp(self):
        K_1 = [1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0]
        self.mesh = Mesh('mesh/mpfad.h5m', dim=3)
        self.mesh.set_boundary_conditions('Dirichlet', {101: 0.0})
        self.mesh.set_material_prop('permeability', {1: K_1})
        self.mpfad = MpfaD(self.mesh)

    def tearDown(self):
        self.mpfad = None

    def test_preprocessor_class_should_be_none(self):
        """Test class initiatilization."""
        mpfad = self.mpfad
        self.assertIsNotNone(mpfad)

    def test_multiply_is_working(self):
        b_faces = self.mesh.b_faces
        tri_faces, tetra_faces = self.mesh.screen_faces_by_verts(b_faces)
        volumes = self.mesh.get_left_and_right_volumes(tri_faces,
                                                       boundary=True)
        N_IJK, tan_JI, tan_JK = self.mesh.construct_face_vectors(tri_faces)
        perm = self.mesh.permeability[volumes][0][0]
        K_n_L = self.mpfad.multiply(N_IJK, perm, N_IJK)
        self.assertTrue(all(K_n_L == 1))

    def test_cross_diffusion_term_for_boundary_elems(self):
        b_faces = self.mesh.b_faces
        tri_faces, tetra_faces = self.mesh.screen_faces_by_verts(b_faces)
        volumes = self.mesh.get_left_and_right_volumes(tetra_faces,
                                                       boundary=True)
        N_IJK, tan_JI, tan_JK = self.mesh.construct_face_vectors(tetra_faces)
        perm = self.mesh.permeability[volumes][0][0]
        K_n_L = self.mpfad.multiply(N_IJK, perm, N_IJK)
        K_L_JI = self.mpfad.multiply(N_IJK, perm, tan_JI)
        K_L_JK = self.mpfad.multiply(N_IJK, perm, tan_JK)
        D_JK = self.get_cross_diffusion_term(tan_JK, LJ, face_area,
                                             h_L, K_n_L, K_L_JK,
                                             boundary=True)
        D_JI = self.get_cross_diffusion_term(tan_JI, LJ, face_area,
                                             h_L, K_n_L, K_L_JI,
                                             boundary=True)




if __name__== "__main__":
    unittest.main()
