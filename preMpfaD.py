import numpy as np
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh as mesh


class PreMpfaD:

    def __init__(self, mesh_file):
        M = mesh(mesh_file, dim=3)
