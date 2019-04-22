"""Universidade Federal de Pernambuco."""
import numpy as np
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh as mesh
# from PyTrilinos import Epetra, AztecOO

"""PRESTO - Python REservoir Simulation TOolbox
   Author: Ricardo Lira (April/2019)
   Description: Implementation of a Multipoint Flux Approximation scheme with
   Diamond Stencil (MPFA-D) for Diffusion problems using generalized
   polehedral mesh."""


class MpfaD:
    """Implements the MPFA-D method."""

    def __init__(self, mesh_file, dim, x=None):
        """Initialize the mesh.

        Arguments
            - mesh_file: .h5m data file or .msh
            - dim: mesh dimension (ex: 3 = tridimensional mesh file)
        Returns None
        """
        self.M = mesh(mesh_file, dim)
        # self.comm = Epetra.PyComm()

        # Get all geometric entities, separating from bondaries and internal.
        self.all_volumes = self.M.volumes.all

        self.in_faces = self.M.faces.internal
        self.b_faces = self.M.faces.boundary

        self.in_verts = self.M.nodes.internal
        self.b_verts = self.M.nodes.boundary

        # Get all coordinates
        self.centroid = self.M.volumes.center


        # std_map = Epetra.Map(len(self.volumes), 0, self.comm)
        # self.T = Epetra.CrsMatrix(Epetra.Copy, std_map, 0)
        # self.Q = Epetra.Vector(std_map)
        # if x is None:
        #     self.x = Epetra.Vector(std_map)
        # else:
        #     self.x = x

    def set_boundary_conditions(self, boundary_type, boundary_flag):
        for _flag, value in boundary_flag.items():
            faces = self.M.faces.flag[_flag]
            if boundary_type == 'Dirichlet':
                self.M.dirichlet_faces[faces] = value
            if boundary_type == 'Neumann':
                self.M.neumann_faces[faces] = value

    def set_material_prop(self, material_property, material_flag):
        for _flag, value in material_flag.items():
            volumes = self.M.volumes.flag[_flag]
            if material_property == 'permeability':
                self.M.permeability[volumes] = value
            if material_property == 'saturation':
                self.M.saturation[volumes] = value
            if material_property == 'porosity':
                self.M.porosity[volumes] = value

    def screen_faces_by_verts(self):
        all_faces = self.M.faces.all
        tri_faces = []
        quad_faces = []
        for face in all_faces:
            verts_in_face = self.M.faces.bridge_adjacencies([face], 2, 0)[0]
            if len(verts_in_face) == 3:
                tri_faces.append(face)
            else:
                quad_faces.append(face)
        return tri_faces, quad_faces
