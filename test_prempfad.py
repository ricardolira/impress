"""This is the mpfad-3D preprocessor Tests."""
import unittest
import numpy as np
from preMpfaD import preMpfaD


class PreMpfaDTest(unittest.TestCase):

    def setUp(self):
        preSolver = preMpfaD('mesh/mpfad.h5m', dim=3)
