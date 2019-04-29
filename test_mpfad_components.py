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

    def test_cross_diffusion_term_for_boundary_quad_elems(self):
        b_faces = self.mesh.b_faces
        tri_faces, tetra_faces = self.mesh.screen_faces_by_verts(b_faces)
        is_boundary = True
        volumes = self.mesh.get_left_and_right_volumes(tetra_faces,
                                                       boundary=is_boundary)
        N_IJK, tan_JI, tan_JK, tan_J2I, tan_J2K = \
            self.mesh.construct_face_vectors(tetra_faces)
        face_area = self.mesh.get_area(tetra_faces)
        LJ, h_L = self.mesh.get_additional_vectors_and_height(tetra_faces,
                                                              boundary=is_boundary)
        perm = self.mesh.permeability[volumes][0][0]
        K_L_n = self.mpfad.multiply(N_IJK, perm, N_IJK)
        K_L_JI = self.mpfad.multiply(N_IJK, perm, tan_JI)
        K_L_JK = self.mpfad.multiply(N_IJK, perm, tan_JK)
        K_L_J2I = self.mpfad.multiply(N_IJK, perm, tan_J2I)
        K_L_J2K = self.mpfad.multiply(N_IJK, perm, tan_J2K)
        D_JK = self.mpfad.d_flux_term(tan_JK, LJ, face_area / 2, h_L, K_L_n,
                                      K_L_JK, boundary=is_boundary)
        D_JI = self.mpfad.d_flux_term(tan_JI, LJ, face_area / 2, h_L, K_L_n,
                                      K_L_JI, boundary=is_boundary)
        D_J2K = self.mpfad.d_flux_term(tan_J2K, LJ, face_area / 2, h_L, K_L_n,
                                       K_L_J2K, boundary=is_boundary)
        D_J2I = self.mpfad.d_flux_term(tan_J2I, LJ, face_area / 2, h_L, K_L_n,
                                       K_L_J2I, boundary=is_boundary)
        K_eq = self.mpfad.n_flux_term(K_L_n, h_L, face_area,
                                      boundary=is_boundary)
        self.assertListEqual(list(D_JI), list(D_J2K))
        self.assertListEqual(list(D_JK), list(D_J2I))

    def test_cross_diffusion_term_for_boundary_tri_elems(self):
        b_faces = self.mesh.b_faces
        tri_faces, tetra_faces = self.mesh.screen_faces_by_verts(b_faces)
        is_boundary = True
        volumes = self.mesh.get_left_and_right_volumes(tri_faces,
                                                       boundary=is_boundary)
        N_IJK, tan_JI, tan_JK = self.mesh.construct_face_vectors(tri_faces)
        face_area = self.mesh.get_area(tri_faces)
        LJ, h_L = self.mesh.get_additional_vectors_and_height(tri_faces,
                                                              boundary=is_boundary)
        perm = self.mesh.permeability[volumes][0][0]
        K_L_n = self.mpfad.multiply(N_IJK, perm, N_IJK)
        K_L_JI = self.mpfad.multiply(N_IJK, perm, tan_JI)
        K_L_JK = self.mpfad.multiply(N_IJK, perm, tan_JK)
        D_JK = self.mpfad.d_flux_term(tan_JK, LJ, face_area, h_L, K_L_n,
                                      K_L_JK, boundary=is_boundary)
        D_JI = self.mpfad.d_flux_term(tan_JI, LJ, face_area, h_L, K_L_n,
                                      K_L_JI, boundary=is_boundary)
        K_eq = self.mpfad.n_flux_term(K_L_n, h_L, face_area,
                                          boundary=is_boundary)

        self.assertIsNotNone([D_JK, D_JI, K_eq])

    def test_cross_diffusion_term_for_itnern_tri_elems(self):
        in_faces = self.mesh.in_faces
        tri_faces, tetra_faces = self.mesh.screen_faces_by_verts(in_faces)
        left_volumes, right_volumes = self.mesh.get_left_and_right_volumes(tetra_faces)
        N_IJK, tan_JI, tan_JK, tan_J2I, tan_J2K = \
            self.mesh.construct_face_vectors(tetra_faces)
        face_area = self.mesh.get_area(tetra_faces)
        LR, h_L, h_R = self.mesh.get_additional_vectors_and_height(tetra_faces)
        left_perm = self.mesh.permeability[left_volumes][0][0]
        right_perm = self.mesh.permeability[right_volumes][0][0]
        K_L_n = self.mpfad.multiply(N_IJK, left_perm, N_IJK)
        K_L_JI = self.mpfad.multiply(N_IJK, left_perm, tan_JI)
        K_L_JK = self.mpfad.multiply(N_IJK, left_perm, tan_JK)
        K_L_J2I = self.mpfad.multiply(N_IJK, left_perm, tan_J2I)
        K_L_J2K = self.mpfad.multiply(N_IJK, left_perm, tan_J2K)
        K_R_n = self.mpfad.multiply(N_IJK, right_perm, N_IJK)
        K_R_JI = self.mpfad.multiply(N_IJK, right_perm, tan_JI)
        K_R_JK = self.mpfad.multiply(N_IJK, right_perm, tan_JK)
        K_R_J2I = self.mpfad.multiply(N_IJK, right_perm, tan_J2I)
        K_R_J2K = self.mpfad.multiply(N_IJK, right_perm, tan_J2K)
        D_JK = self.mpfad.d_flux_term(tan_JK, LR, face_area / 2, h_L, K_L_n,
                                      K_L_JK, h_R, K_R_JK, K_R_n)
        D_JI = self.mpfad.d_flux_term(tan_JI, LR, face_area / 2, h_L, K_L_n,
                                      K_L_JI, h_R, K_R_JI, K_R_n)
        D_J2K = self.mpfad.d_flux_term(tan_J2K, LR, face_area / 2, h_L, K_L_n,
                                       K_L_J2K, h_R, K_R_J2K, K_R_n)
        D_J2I = self.mpfad.d_flux_term(tan_J2I, LR, face_area / 2, h_L, K_L_n,
                                       K_L_J2I, h_R, K_R_J2I, K_R_n)
        K_eq = self.mpfad.n_flux_term(K_L_n, h_L, face_area, K_R_n, h_R)
        self.assertListEqual(list(D_JI), list(D_J2K))
        self.assertListEqual(list(D_JK), list(D_J2I))

if __name__== "__main__":
    unittest.main()
