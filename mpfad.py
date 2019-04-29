"""Universidade Federal de Pernambuco."""
import numpy as np
from mesh_preprocessor import Mesh
from PyTrilinos import Epetra, AztecOO
import time

"""PRESTO - Python REservoir Simulation TOolbox
   ***************************************************************************
   Author: Ricardo Lira (April/2019)
   Project: Multipoint FLux Approximation with Diamond Stencil (MPFA-D) Solver
   Description: This class is designed for assembling the MPFA-D problem

   Depencencies: Build on top of the Intuitive Multilevel Preprocessor for
   Smart Simulation - IMPRESS"""


class MpfaD:
    def __init__(self, mesh, interpolation_method=None, x=None):
        self.comm = Epetra.PyComm()
        std_map = Epetra.Map(len(mesh.all_volumes), 0, self.comm)
        self.T = Epetra.CrsMatrix(Epetra.Copy, std_map, 0)
        self.Q = Epetra.Vector(std_map)
        if x is None:
            self.x = Epetra.Vector(std_map)
        else:
            self.x = x

    def multiply(self, normal_vector, tensor, CD):
        vmv = np.sum(np.dot(normal_vector,
                            tensor) * CD, axis=1)
        area = np.sum(normal_vector * normal_vector, axis=1)
        return vmv / area

    def n_flux_term(self, K_L_n, h_L, face_area, K_R_n=0, h_R=0,
                    boundary=False):
        if boundary:
            K_eq = (1 / h_L)*(face_area * K_L_n)
            return K_eq
        K_eq = (K_R_n * K_L_n)/(K_R_n * h_L + K_L_n * h_R) * face_area
        return K_eq

    def d_flux_term(self, tan, vec, S, h1, Kn1, Kt1, h2=0, Kt2=0, Kn2=0,
                    boundary=False):
        if not boundary:
            mesh_anisotropy_term = (np.sum(tan * vec, axis=1)/(S ** 2))
            physical_anisotropy_term = -((1 / S) * (h1 * (Kt1 / Kn1)
                                         + h2 * (Kt2 / Kn2)))
            cross_diffusion_term = (mesh_anisotropy_term +
                                    physical_anisotropy_term)
            return cross_diffusion_term
        if boundary:
            dot_term = np.sum(-tan * vec, axis=1) * Kn1
            cdf_term = h1 * S * Kt1
            b_cross_difusion_term = (dot_term + cdf_term) / (2 * h1 * S)
            return b_cross_difusion_term

    def _node_treatment(self, args):
        pass

    def linear_problem(self, args):
        pass
        # go through volumes and fill source_term
        # go through boundary faces:
        #   triangular b_faces
        #   quadriangular b_faces

        # go through intern faces:
        #   triangular in_faces
        #   quadriangular in_faces

        # go through vertex

        # self.T.InsertGlobalValues(self.ids, self.v_ids, self.ivalues)
        # self.T.InsertGlobalValues(id_volumes, id_volumes, all_LHS)
        # self.T.InsertGlobalValues(all_cols, all_rows, all_values)
        # self.T.FillComplete()

    def solve_linear_problem(self):
        linearProblem = Epetra.LinearProblem(self.T, self.x, self.Q)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_none)
        solver.Iterate(2000, 1e-16)
        # return self.x
