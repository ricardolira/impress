"""This is the mpfad-3D preprocessor Tests."""
import unittest
# import numpy as np
from preMpfaD import MpfaD


class PreMpfaDTest(unittest.TestCase):

    def setUp(self):
        K_1 = [1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0]
        self.mpfad = MpfaD('mesh/mpfad.h5m', dim=3)
        self.mpfad.set_boundary_conditions('Dirichlet', {101: 0.0})
        self.mpfad.set_material_prop('permeability', {1: K_1})

    def tearDown(self):
        self.mpfad = None

    def test_preprocessor_class_should_be_none(self):
        """Test class initiatilization."""
        mpfad = self.mpfad
        self.assertIsNotNone(mpfad)

    def test_get_all_volumes(self):
        """Test class get all volumes."""
        volumes = self.mpfad.all_volumes
        self.assertEqual(len(volumes), 9)

    def test_get_all_internal_faces(self):
        """Test class get all internal faces."""
        in_faces = self.mpfad.in_faces
        self.assertEqual(len(in_faces), 12)

    def test_get_all_boundary_faces(self):
        """Test class get all boundary faces."""
        b_faces = self.mpfad.b_faces
        self.assertEqual(len(b_faces), 16)

    def test_get_all_internal_verts(self):
        """Test class get all internal verts."""
        in_verts = self.mpfad.in_verts
        self.assertEqual(len(in_verts), 0)

    def test_get_all_boundary_verts(self):
        """Test class get all boundary verts."""
        b_verts = self.mpfad.b_verts
        self.assertEqual(len(b_verts), 13)

    def test_screening_faces_by_verts_gets_only_quadrilateral_faces(self):
        """Test class to get only quadrilateral faces."""
        _, quad_faces = self.mpfad.screen_faces_by_verts()
        self.assertEqual(len(quad_faces),  7)

    def test_screening_faces_by_verts_gets_only_triangular_faces(self):
        """Test class to get only quadrilateral faces."""
        tri_faces, _ = self.mpfad.screen_faces_by_verts()
        self.assertEqual(len(tri_faces),  21)

    def test_if_permeability_tensor_is_assigned(self):
        """Test if permeability tensor is being assigned for all volumes."""
        all_volumes = self.mpfad.all_volumes
        K_1 = [1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0]
        for volume in all_volumes:
            perm = self.mpfad.M.permeability[[volume]][0]
            for value, i in zip(perm, range(len(perm))):
                self.assertEqual(value, K_1[i])

    def test_get_dirichlet_faces(self):
        """Test if Dirichlet BC is implemented."""
        d_faces = self.mpfad.M.faces.flag[101]
        dirichlet_faces_pressure = self.mpfad.M.dirichlet_faces
        for face in d_faces:
            b_pressure = dirichlet_faces_pressure[[face]]
            self.assertEqual(b_pressure, 0.)


if __name__ == "__main__":
    unittest.main()
