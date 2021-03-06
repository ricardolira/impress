"""Universidade Federal de Pernambuco."""
import numpy as np
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh as load_mesh
from scipy.spatial import ConvexHull
# import preprocessor.geoUtil.cgeom as geo

"""PRESTO - Python REservoir Simulation TOolbox
   ***************************************************************************
   Author: Ricardo Lira (April/2019)
   Project: Multipoint FLux Approximation with Diamond Stencil (MPFA-D) Solver
   Description: Development of a mesh preprocessor for the implementation of a
   numerical solver using a Multipoint Flux approximatino scheme with diamond
   stencil.

   Depencencies: Build on top of the Intuitive Multilevel Preprocessor for
   Smart Simulation - IMPRESS"""


class Mesh:
    """Implements the MPFA-D method."""

    def __init__(self, mesh_file, dim):
        """Initialize the mesh.

        Arguments
            - mesh_file: .h5m data file or .msh
            - dim: mesh dimension (ex: 3 = tridimensional mesh file)
        Returns None
        """
        self.M = load_mesh(mesh_file, dim)
        # Get all geometric entities, separating from bondaries and internal.
        self.all_volumes = self.M.volumes.all

        self.b_volumes = self.M.volumes.boundary

        self.in_faces = self.M.faces.internal
        self.b_faces = self.M.faces.boundary

        self.in_verts = self.M.nodes.internal
        self.b_verts = self.M.nodes.boundary

        # Get all coordinates
        self.centroid = self.M.volumes.center

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
                shape = (1, len(volumes), 3, 3)
                self.permeability = self.M.permeability[volumes
                                                        ].reshape(shape)[0]
            if material_property == 'saturation':
                self.M.saturation[volumes] = value
            if material_property == 'porosity':
                self.M.porosity[volumes] = value
            if material_property == 'source':
                self.M.source[volumes] = value

    def screen_faces_by_verts(self, faces):
        all_faces = faces
        tri_faces = []
        quad_faces = []
        for face in all_faces:
            verts_in_face = self.M.faces.bridge_adjacencies([face], 2, 0)[0]
            if len(verts_in_face) == 3:
                tri_faces.append(face)
            else:
                quad_faces.append(face)
        return tri_faces, quad_faces

    def get_left_and_right_volumes(self, faces, boundary=False):
        if boundary:
            adj_vol = self.M.faces.bridge_adjacencies(faces, 2, 3)
            return adj_vol
        N_IJK = self.construct_face_vectors(faces)[0]
        adj_vols = self.M.faces.bridge_adjacencies(faces, 2, 3)
        left_volumes, right_volumes = adj_vols[:, 0], adj_vols[:, 1]
        vector_left_to_right = (self.M.volumes.center[left_volumes]
                                - self.M.volumes.center[right_volumes])
        is_positive = np.sum(vector_left_to_right * N_IJK, axis=1)
        ids = np.flatnonzero(is_positive < 0)
        right_volumes[ids], left_volumes[ids] = (left_volumes[ids],
                                                 right_volumes[ids])
        return left_volumes, right_volumes

    def construct_face_vectors(self, faces, boundary=False):
        try:  # To calculate a quadrilateral face area vector
            i, j, k, j_2 = self.get_position_IJK_verts(faces)
            JI = self.M.nodes.coords[i] - self.M.nodes.coords[j]
            JK = self.M.nodes.coords[k] - self.M.nodes.coords[j]
            N_IJK = np.cross(JI, JK)
            J2I = self.M.nodes.coords[i] - self.M.nodes.coords[j_2]
            J2K = self.M.nodes.coords[k] - self.M.nodes.coords[j_2]
            if boundary:
                adj_vol = self.M.faces.bridge_adjacencies(faces, 2, 3)
                centroid = self.M.volumes.center[adj_vol]
                outward_vector = self.M.nodes.coords[i] - centroid
                is_positive = np.sum(outward_vector * N_IJK, axis=1)
                ids = np.flatnonzero(is_positive < 0)
                N_IJK[ids] = -N_IJK[ids]
            tan_JI = np.cross(N_IJK, JI)
            tan_JK = np.cross(N_IJK, JK)
            tan_J2I = - np.cross(N_IJK, J2I)
            tan_J2K = - np.cross(N_IJK, J2K)
            return N_IJK, tan_JI, tan_JK, tan_J2I, tan_J2K
        except ValueError:   # To calculate a triangular face area vector
            i, j, k = self.get_position_IJK_verts(faces)
            JI = self.M.nodes.coords[i] - self.M.nodes.coords[j]
            JK = self.M.nodes.coords[k] - self.M.nodes.coords[j]
            N_IJK = np.cross(JI, JK) / 2
        if boundary:
            adj_vol = self.M.faces.bridge_adjacencies(faces, 2, 3)
            centroid = self.M.volumes.center[adj_vol]
            outward_vector = self.M.nodes.coords[i] - centroid
            is_positive = np.sum(outward_vector * N_IJK, axis=1)
            ids = np.flatnonzero(is_positive < 0)
            N_IJK[ids] = -N_IJK[ids]
        tan_JI = np.cross(N_IJK, JI)
        tan_JK = np.cross(N_IJK, JK)
        return N_IJK, tan_JI, tan_JK

    def get_additional_vectors_and_height(self, faces, boundary=False):
        N_IJK = self.construct_face_vectors(faces)[0]
        area = self.get_area(faces)
        j = self.get_position_IJK_verts(faces)[1]
        j_coords = self.M.nodes.coords[j]
        if boundary:
            left_volumes = self.get_left_and_right_volumes(faces, boundary)
            L_center = self.M.volumes.center[left_volumes]
            LJ = j_coords - L_center
            h_L = np.absolute(np.sum(N_IJK * LJ, axis=1) / area)
            return LJ, h_L
        left_volumes, right_volumes = self.get_left_and_right_volumes(faces)
        L_center = self.M.volumes.center[left_volumes]
        R_center = self.M.volumes.center[right_volumes]
        LR = L_center - R_center
        LJ = j_coords - L_center
        h_L = np.absolute(np.sum(N_IJK * LJ, axis=1) / area)
        RJ = j_coords - R_center
        h_R = np.absolute(np.sum(N_IJK * RJ, axis=1) / area)
        return LR, h_L, h_R

    def get_area(self, faces):
        face_vectors = self.construct_face_vectors(faces)[0]
        return np.sqrt(np.sum(face_vectors * face_vectors, axis=1))

    # TODO: Go on a Cythonized func
    def get_volume(self, volumes):
        def _get_volume(volume):
            volume_verts = self.M.volumes.connectivities(volumes)[volume]
            return ConvexHull(volume_verts).volume
        n_verts = [_get_volume(volume) for volume in volumes]
        print(n_verts)

    def get_position_IJK_verts(self, faces):
        verts = self.M.faces.connectivities(faces)
        try:  # To get all vertices in a quadrangular face
            i, j, k, j_2 = verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3]
            return i, j, k, j_2
        except IndexError:  # To get all vertices in a triangular face
            i, j, k = verts[:, 0], verts[:, 1], verts[:, 2]
        return i, j, k

    def record_data(self, args):
        pass
