import numpy as np
from scipy import sparse
import einops

from .dec_operators import (
    get_star1_product,
    get_slice_product_laplacian,
    get_connection_product,
    get_curvature_product,
    get_d1_product,
    get_d2_product,
)

from .singularity_solver import find_edge_edge_zeros


from .surface_mesh import SurfaceMesh


class ProductMesh:
    def __init__(self, mesh1: SurfaceMesh, mesh2: SurfaceMesh):
        self.mesh1 = mesh1
        self.mesh2 = mesh2

        self._curvature_spin = None

    @property
    def n1(self):
        return self.mesh1.n_vertices

    @property
    def n2(self):
        return self.mesh2.n_vertices

    @property
    def e1(self):
        return self.mesh1.n_edges

    @property
    def e2(self):
        return self.mesh2.n_edges

    @property
    def f1(self):
        return self.mesh1.n_faces

    @property
    def f2(self):
        return self.mesh2.n_faces

    @property
    def n_faces(self):
        """
        Number of vertices in the product mesh.

        Returns
        -------
        n_vertices : int
            Number of vertices in the product mesh
        """
        return self.f1 * self.n2 + self.e1 * self.e2 + self.n1 * self.f2

    @property
    def n_edges(self):
        """
        Number of undirected edges in the product mesh.

        Returns
        -------
        n_edges : int
            Number of undirected edges in the product mesh
        """
        return self.e1 * self.n2 + self.n1 * self.e2

    @property
    def n_vertices(self):
        """
        Number of vertices in the product mesh.

        Returns
        -------
        n_vertices : int
            Number of vertices in the product mesh
        """
        return self.n1 * self.n2

    def get_product_connection(self, conn1, conn2):
        assert conn1.shape[0] == self.e1
        assert conn2.shape[0] == self.e2
        return get_connection_product(
            conn1,
            conn2,
            self.mesh1.n_vertices,
            self.mesh2.n_vertices,
        )

    def get_product_curvature(self, curv1, curv2):
        return get_curvature_product(
            curv1,
            curv2,
            self.mesh1.n_vertices,
            self.mesh2.n_vertices,
            self.mesh1.n_edges,
            self.mesh2.n_edges,
        )

    @property
    def connection_angle_spin(self):
        return self.get_product_connection(
            self.mesh1.connection_angle_spin, self.mesh2.connection_angle_spin
        )

    @property
    def curvature_spin(self):
        """
        Half of the curvature 2-form of the product mesh.

        Returns
        -------
        curvature_spin : np.array
            (f_A * n_B + e_A*e_B + f_B * n_A,) Array of half curvature on each face of the product mesh (in radians, any value)
        """
        if not hasattr(self, "_curvature_spin"):
            self._curvature_spin = None
        if self._curvature_spin is None:
            self._curvature_spin = self.get_product_curvature(
                self.mesh1.curvature_spin,
                self.mesh2.curvature_spin,
            )
        return self._curvature_spin

    @property
    def connection_angle_LC(self):
        return self.get_product_connection(
            self.mesh1.connection_angle_LC,
            self.mesh2.connection_angle_LC,
        )

    @property
    def curvature_LC(self):
        """
        Half of the curvature 2-form of the product mesh.

        Returns
        -------
        curvature_LC : np.array
            (f_A * n_B + e_A*e_B + f_B * n_A,) Array of half curvature on each face of the product mesh (in radians, any value)
        """
        return self.get_product_curvature(
            self.mesh1.curvature_LC, self.mesh2.curvature_LC
        )

    @property
    def d1(self):
        return get_d1_product(
            self.mesh1.d0, self.mesh2.d0, self.mesh1.d1, self.mesh2.d1
        )

    @property
    def d2(self):
        return get_d2_product(
            self.mesh1.d0, self.mesh2.d0, self.mesh1.d1, self.mesh2.d1
        )

    @property
    def star1(self):
        return get_star1_product(
            self.mesh1.star0, self.mesh2.star0, self.mesh1.star1, self.mesh2.star1
        )

    @property
    def star1_inv(self):
        return get_star1_product(
            self.mesh1.star0_inv,
            self.mesh2.star0_inv,
            self.mesh1.star1_inv,
            self.mesh2.star1_inv,
        )

    def product_view_0form(self, vec):
        """
        View a 0-form on the product mesh as a 2D array.

        Parameters
        ----------
        vec : np.array
            (n_A*n_B,) Array of 0-form on the product mesh

        Returns
        -------
        vert_verts_vals : np.array
            (n_A,n_B) Array of values on vertices of the first mesh for each vertex of the second mesh
        """
        n1 = self.mesh1.n_vertices
        n2 = self.mesh2.n_vertices

        assert vec.shape[0] == n1 * n2

        vert_verts_vals = einops.rearrange(vec, "(n1 n2) -> n1 n2", n1=n1, n2=n2)

        return vert_verts_vals

    def product_view_1form(self, vec):
        """
        View a 1-form on the product mesh as a 2D array.

        Parameters
        ----------
        vec : np.array
            (e_A*n_B + n_A*e_B,) Array of 1-form on the product mesh

        Returns
        -------
        edge_verts_vals : np.array
            (e_A,n_B) Array of values on edges of the first mesh for each vertex of the second mesh
        vert_edge_vals : np.array
            (n_A,e_B) Array of values on edges of the second mesh for each vertex of the first mesh
        """
        n1 = self.mesh1.n_vertices
        n2 = self.mesh2.n_vertices

        e1 = self.mesh1.n_edges
        e2 = self.mesh2.n_edges

        assert vec.shape[0] == e1 * n2 + n1 * e2

        n_half = e1 * n2

        edge_verts_vals = einops.rearrange(
            vec[:n_half], "(e1 n2) -> e1 n2", e1=e1, n2=n2
        )
        vert_edge_vals = einops.rearrange(
            vec[n_half:], "(n1 e2) -> n1 e2", n1=n1, e2=e2
        )

        return edge_verts_vals, vert_edge_vals

    def product_view_2form(self, vec):
        """
        View a 2-form on the product mesh as a 3D array.

        Parameters
        ----------
        vec : np.array
            (f_A * n_B + e_A*e_B + f_B * n_A,) Array of 2-form on the product mesh

        Returns
        -------
        face_verts_vals : np.array
            (f_A,n_B) Array of values on faces of the first mesh for each vertex of the second mesh
        edge_edge_vals : np.array
            (e_A,e_B) Array of values on edges of the first mesh for each edge of the second mesh
        vert_face_vals : np.array
            (n_A,f_B) Array of values on faces of the second mesh for each vertex of the first mesh
        """
        n1 = self.mesh1.n_vertices
        n2 = self.mesh2.n_vertices

        e1 = self.mesh1.n_edges
        e2 = self.mesh2.n_edges

        f1 = self.mesh1.n_faces
        f2 = self.mesh2.n_faces

        assert vec.shape[0] == f1 * n2 + e1 * e2 + f2 * n1

        n_half1 = f1 * n2
        n_half2 = n_half1 + e1 * e2

        face_verts_vals = einops.rearrange(
            vec[:n_half1], "(f1 n2) -> f1 n2", f1=f1, n2=n2
        )
        edge_edge_vals = einops.rearrange(
            vec[n_half1:n_half2], "(e1 e2) -> e1 e2", e1=e1, e2=e2
        )
        vert_face_vals = einops.rearrange(
            vec[n_half2:], "(n1 f2) -> n1 f2", n1=n1, f2=f2
        )

        return face_verts_vals, edge_edge_vals, vert_face_vals

    def evaluate_section_at_barycentric(
        self,
        section,
        face_inds1,
        face_inds2,
        bary_coords1,
        bary_coords2,
        connection1,
        connection2,
    ):
        """
        Evaluate a section on the product mesh at given barycentric coordinates.

        Parameters
        ----------
        section : np.array
            (n1*n2,) or (n1,n2) Array of section values (complex numbers) on the vertices of the product mesh
        face_inds1 : np.array
            (p,) Array of face indices on the first mesh
        face_inds2 : np.array
            (p,) Array of face indices on the second mesh
        bary_coords1 : np.array
            (p, 3) Array of barycentric coordinates for each face in face_inds1
        bary_coords2 : np.array
            (p, 3) Array of barycentric coordinates for each face in face_inds2
        connection1 : np.array
            (e1,) Array of connection angles on each undirected edge of the first mesh (in radians between -pi and pi)
        connection2 : np.array
            (e2,) Array of connection angles on each undirected edge of the second mesh (
        """
        # (p,3)
        phi_1 = self.mesh1.get_basis_functions_at_barycentric(
            face_inds1, bary_coords1, connection1
        )
        phi_2 = self.mesh2.get_basis_functions_at_barycentric(
            face_inds2, bary_coords2, connection2
        )

        faces1_verts = self.mesh1.faces[face_inds1]  # (p, 3)
        faces2_verts = self.mesh2.faces[face_inds2]  # (p, 3)

        if section.ndim == 1:
            section = self.product_view_0form(section)  # (n1,n2)

        # Extract the 3x3 local section values for each point pair
        # We need Z values for every pair of vertices in the two triangles.
        # rows: (N, 3, 1) indices from mesh1
        # cols: (N, 1, 3) indices from mesh2
        # Z_local: (N, 3, 3)
        rows = faces1_verts[:, :, None]  # (p, 3, 1)
        cols = faces2_verts[:, None, :]  # (p, 1, 3)
        Z_local = section[rows, cols]

        result = np.einsum("ni, nij, nj -> n", phi_1, Z_local, phi_2)

        return result

    def evaluate_section_at_vertices_of_mesh2(
        self, section, face_inds1, bary_coords1, connection1
    ):
        """For each point p_n on mesh1, return z(p_n, v) for every v in V_2.

        Returns: (p, n2) complex array.
        """
        phi_1 = self.mesh1.get_basis_functions_at_barycentric(
            face_inds1, bary_coords1, connection1
        )  # (p, 3)
        faces1_verts = self.mesh1.faces[face_inds1]  # (p, 3)

        if section.ndim == 1:
            section = self.product_view_0form(section)  # (n1, n2)

        Z_rows = section[faces1_verts]  # (p, 3, n2)
        return np.einsum("pi, pin -> pn", phi_1, Z_rows)

    def compute_rotation_form(self, section, connection_angle):
        """
        Compute the rotation 1-form alpha for a given section on the vertices of the product mesh.

        The rotation form is defined such that it represents the change in angle of the section along each edge of the product mesh.

        Parameters
        ----------
        section : np.array
            (n_A*n_B,) Array of complex values on the vertices of the product mesh
        connection : np.array or (np.array, np.array)
            (e_A*n_B + n_A*e_B,) Array of connection angles on each undirected edge of the product mesh (in radians between -pi and pi)

        Returns
        -------
        alpha : np.array
            (e_A*n_B + n_A*e_B,) Array of rotation angles on each undirected edge of the product mesh (in radians between -pi and pi)
        """
        n1 = self.mesh1.n_vertices
        n2 = self.mesh2.n_vertices

        e1 = self.mesh1.n_edges
        e2 = self.mesh2.n_edges

        assert section.shape[0] == n1 * n2

        section_prod = self.product_view_0form(section)  # (n1,n2)

        if isinstance(connection_angle, np.ndarray) and connection_angle.ndim == 1:
            assert connection_angle.shape[0] == e1 * n2 + n1 * e2
            # (e1,n2), (n1,e2)
            rho_hor, rho_vert = self.product_view_1form(connection_angle)
            rho_vert = rho_vert.T  # (e2,n1)
        else:
            # Factorized connection
            rho_hor, rho_vert = connection_angle
            assert rho_hor.shape[0] == e1
            assert rho_vert.shape[0] == e2

        # (e1, n2)
        rot_form_hor = self.mesh1.compute_rotation_form(section_prod, rho_hor)

        # (e2,n1)
        rot_form_vert = self.mesh2.compute_rotation_form(section_prod.T, rho_vert)

        rot_form = np.concatenate(
            [
                einops.rearrange(rot_form_hor, "e1 n2 -> (e1 n2)", e1=e1).flatten(),
                einops.rearrange(rot_form_vert, "e2 n1 -> (n1 e2)", e2=e2).flatten(),
            ],
            axis=0,
        )  # (e1*n2 + n1*e2,)

        return rot_form

    def compute_index_form(
        self, curvature_form, rotform=None, section=None, connection_angle=None
    ):
        """
        Compute the index 2-form I for a given curvature 2-form and rotation 1-form on the product mesh.

        The index form is defined such that it represents the sum of indices of singularities within each face

        Parameters
        ----------
        curvature_form : np.array
            (f_A * n_B + e_A*e_B + f_B * n_A,) Array of curvature on each face of the product mesh (in radians, any value)
        rotform : np.array, optional
            (e_A*n_B + n_A*e_B,) rotation form of the section on the product mesh (in radians between -pi and pi)
        section : np.array, optional
            (n_A*n_B,) Array of complex values on the vertices of the product mesh
            Required if rotform is not provided.
        connection : np.array, optional
            (e_A*n_B + n_A*e_B,) Array of connection angles on each
            undirected edge of the product mesh (in radians between -pi and pi)
            Required if rotform is not provided.

        Returns
        -------
        index : np.array
            (f_A * n_B + e_A*e_B + f_B * n_A,) Array of index values on each face of the product mesh (integer values)
        """

        if rotform is None:
            assert section is not None
            assert connection_angle is not None
            rotform = self.compute_rotation_form(section, connection_angle)

        n1 = self.mesh1.n_vertices
        n2 = self.mesh2.n_vertices

        e1 = self.mesh1.n_edges
        e2 = self.mesh2.n_edges

        f1 = self.mesh1.n_faces
        f2 = self.mesh2.n_faces

        assert rotform.shape[0] == self.n_edges

        if isinstance(curvature_form, np.ndarray) and curvature_form.ndim == 1:
            assert curvature_form.shape[0] == f1 * n2 + e1 * e2 + f2 * n1
            index_form = (self.d1 @ rotform + curvature_form) / (
                2 * np.pi
            )  # (f1*n2 + e1*e2 + f2*n1)
        else:
            # (f1,), (f2,)
            curv_hor, curv_vert = curvature_form
            assert curv_hor.shape[0] == f1
            assert curv_vert.shape[0] == f2
            # (e1,n2), (n1,e2)
            rotform_hor, rotform_vert = self.product_view_1form(rotform)

            # (f1,n2)
            ind_hor = self.mesh1.compute_index_form(curv_hor, rotform=rotform_hor)
            # (f2,n1)
            ind_vert = self.mesh2.compute_index_form(curv_vert, rotform=rotform_vert.T)

            # (e1, e2)
            ind_mix = self.mesh1.d0 @ rotform_vert - (self.mesh2.d0 @ rotform_hor.T).T

            index_form = np.concatenate(
                [
                    einops.rearrange(ind_hor, "f1 n2 -> (f1 n2)", f1=f1).flatten(),
                    einops.rearrange(ind_mix, "e1 e2 -> (e1 e2)", e1=e1).flatten(),
                    einops.rearrange(ind_vert, "f2 n1 -> (n1 f2)", f2=f2).flatten(),
                ],
                axis=0,
            )  # (f1*n2 + e1*e2 + f2*n1,)
        return index_form

    def find_map_21(self, section, connection_angle, curvature_form, verbose=False):
        """
        Find the singularities in the index form on the product mesh.

        Parameters
        ----------
        index_form : np.array
            (f_A * n_B + e_A*e_B + f_B * n_A,) Array of index values on each face of the product mesh (integer values)

        Returns
        -------
        singularities : list of tuples
            List of (vertex_index, index_value) tuples representing the singularities in the product mesh
        """

        section_v1_v2 = self.product_view_0form(section)  # (n1, n2)

        if isinstance(connection_angle, np.ndarray) and connection_angle.ndim == 1:
            assert connection_angle.shape[0] == self.n_edges
            # (e1,n2), (n1,e2)
            rho_hor, _ = self.product_view_1form(connection_angle)
        else:
            # Factorized connection
            # (e1,), (e2,)
            rho_hor, _ = connection_angle
            assert rho_hor.shape[0] == self.e1
            # assert rho_vert.shape[0] == self.e2

        if isinstance(curvature_form, np.ndarray) and curvature_form.ndim == 1:
            assert curvature_form.shape[0] == self.n_faces
            curv_hor, _, _ = self.product_view_2form(curvature_form)  # (f1, v2)
        else:
            # (f1,), (f2,)
            curv_hor, _ = curvature_form
            assert curv_hor.shape[0] == self.f1
            # assert curv_vert.shape[0] == self.f2

        v2f_21, bary_coods_21 = self.mesh1.find_single_zero_on_sections(
            section_v1_v2, rho_hor, curv_hor, verbose=verbose
        )

        return v2f_21, bary_coods_21

    def find_map_12(self, section, connection_angle, curvature_form, verbose=False):
        """
        Find the singularities in the index form on the product mesh.

        Parameters
        ----------
        index_form : np.array
            (f_A * n_B + e_A*e_B + f_B * n_A,) Array of index values on each face of the product mesh (integer values)

        Returns
        -------
        singularities : list of tuples
            List of (vertex_index, index_value) tuples representing the singularities in the product mesh
        """

        section_v1_v2 = self.product_view_0form(section)  # (n1, n2)

        if isinstance(connection_angle, np.ndarray) and connection_angle.ndim == 1:
            assert connection_angle.shape[0] == self.n_edges
            # (e1,n2), (n1,e2)
            _, rho_vert = self.product_view_1form(connection_angle)
            rho_vert = rho_vert.T  # (e2,n1)
        else:
            # Factorized connection
            # (e1,), (e2,)
            _, rho_vert = connection_angle
            # assert rho_hor.shape[0] == self.e1
        assert rho_vert.shape[0] == self.e2

        if isinstance(curvature_form, np.ndarray) and curvature_form.ndim == 1:
            assert curvature_form.shape[0] == self.n_faces
            _, _, curv_vert = self.product_view_2form(curvature_form)  # (v1, f2)
            curv_vert = curv_vert.T  # (f2, v1)
        else:
            # (f1,), (f2,)
            _, curv_vert = curvature_form
            # assert curv_hor.shape[0] == self.f1
            assert curv_vert.shape[0] == self.f2

        v2f_12, bary_coods_12 = self.mesh2.find_single_zero_on_sections(
            section_v1_v2.T, rho_vert, curv_vert, verbose=verbose
        )

        return v2f_12, bary_coods_12

    def find_edge_edge_singularities(
        self, section, connection_angle, curvature_form, verbose=False
    ):
        """
        Find zeros of the section on edge-edge (quad) faces of the product mesh.

        Computes the index form on edge-edge faces, identifies singular quads
        (non-zero index), and solves for the (s, t) intersection parameters.

        Parameters
        ----------
        section : np.ndarray, shape (n1*n2,)
            Complex section on product mesh vertices.
        connection_angle : np.ndarray or tuple
            Connection 1-form. Either packed (e1*n2 + n1*e2,) or factorized
            tuple ((e1,), (e2,)).
        curvature_form : np.ndarray or tuple
            Curvature 2-form. Either packed (f1*n2 + e1*e2 + n1*f2,) or
            factorized tuple ((f1,), (f2,)).
        verbose : bool
            Print diagnostic information.

        Returns
        -------
        edge_pairs : np.ndarray, shape (K, 2)
            Pairs (edge_A_idx, edge_B_idx) of singular edge-edge faces.
        s_vals : np.ndarray, shape (K,)
            Parameter along edge A (NaN if no valid zero found).
        t_vals : np.ndarray, shape (K,)
            Parameter along edge B (NaN if no valid zero found).
        valid : np.ndarray, shape (K,) bool
            Whether a valid zero was found in [0,1]x[0,1].
        """

        section_v1_v2 = self.product_view_0form(section)  # (n1, n2)

        if isinstance(connection_angle, np.ndarray) and connection_angle.ndim == 1:
            assert connection_angle.shape[0] == self.n_edges
            # (e1,n2), (n1,e2)
            rho_hor, rho_vert = self.product_view_1form(connection_angle)
            rho_vert = rho_vert.T  # (e2,n1)
        else:
            # Factorized connection
            # (e1,), (e2,)
            rho_hor, rho_vert = connection_angle
            assert rho_hor.shape[0] == self.e1
            assert rho_vert.shape[0] == self.e2

        if isinstance(curvature_form, np.ndarray) and curvature_form.ndim == 1:
            assert curvature_form.shape[0] == self.n_faces
            _, curv_edge_edge, _ = self.product_view_2form(curvature_form)  # (f1, v2)
        else:
            curv_edge_edge = np.zeros((self.e1, self.e2))

        rot_form = self.compute_rotation_form(
            section, connection_angle
        )  # (e1*n2 + n1*e2,)
        rot_hor, rot_vert = self.product_view_1form(rot_form)  # (e1,n2), (n1,e2)

        index_edge_edge = (
            curv_edge_edge + self.mesh1.d0 @ rot_vert - (self.mesh2.d0 @ rot_hor.T).T
        ) / (
            2 * np.pi
        )  # (e1,e2)

        # --- Identify singular edge-edge faces ---
        sing_mask = np.abs(index_edge_edge) > 0.5  # (e1, e2)
        eA_indices, eB_indices = np.where(sing_mask)
        n_sing = len(eA_indices)

        if verbose:
            print(f"Edge-edge singularities: {n_sing} / {self.e1 * self.e2} quads")

        if n_sing == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.empty(0),
                np.empty(0),
                np.empty(0, dtype=bool),
            )

        # Warnings (same pattern as find_single_zero_on_sections)
        sing_indices = index_edge_edge[sing_mask]
        if not np.allclose(np.abs(sing_indices), 1):
            unique, counts = np.unique(
                np.round(sing_indices).astype(int), return_counts=True
            )
            count_dict = dict(zip(unique, counts))
            print(f"Warning: Index != +-1 found in edge-edge faces: {count_dict}")

        # --- Gather corner values ---
        edges1 = self.mesh1.edges  # (e1, 2)
        edges2 = self.mesh2.edges  # (e2, 2)

        i_verts = edges1[eA_indices, 0]  # (K,)
        j_verts = edges1[eA_indices, 1]  # (K,)
        k_verts = edges2[eB_indices, 0]  # (K,)
        l_verts = edges2[eB_indices, 1]  # (K,)

        z_ik = section_v1_v2[i_verts, k_verts]  # (K,)
        z_jk = section_v1_v2[j_verts, k_verts]  # (K,)
        z_il = section_v1_v2[i_verts, l_verts]  # (K,)
        z_jl = section_v1_v2[j_verts, l_verts]  # (K,)

        # --- Gather connection angles ---
        if rho_hor.ndim == 1:
            rho_ij = rho_hor[eA_indices]  # (K,)
        else:
            rho_ij = rho_hor[eA_indices, k_verts]  # (K,)

        if rho_vert.ndim == 1:
            rho_kl = rho_vert[eB_indices]  # (K,)
        else:
            rho_kl = rho_vert[eB_indices, i_verts]  # (K,)

        # --- Solve for zeros ---
        s_vals, t_vals, valid = find_edge_edge_zeros(
            z_ik, z_jk, z_il, z_jl, rho_ij, rho_kl
        )

        if verbose:
            n_valid = valid.sum()
            print(f"  Valid zeros found: {n_valid} / {n_sing}")
            if n_valid < n_sing:
                print(f"  Warning: {n_sing - n_valid} singular quads had no valid zero")

        edge_pairs = np.stack([eA_indices, eB_indices], axis=1)  # (K, 2)

        return edge_pairs, s_vals, t_vals, index_edge_edge, valid

    def compute_initial_section(
        self,
        v2f_21,
        v2f_12,
        con_angle_init_1,
        con_angle_init_2,
        curv_init_1,
        curv_init_2,
        initial_guess=None,
        trivial_con_method="hodge",
        method="linop",
        use_preconditioner=True,
        tol=1e-4,
        verbose=False,
    ):

        if v2f_21.shape[0] != self.n2:
            raise ValueError("v2f_21 does not match number of vertices in mesh2.")

        if v2f_12.shape[0] != self.n1:
            raise ValueError("v2f_12 does not match number of vertices in mesh1.")

        # (f1, n2)
        target_curv1 = np.zeros((self.f1, self.n2))
        target_curv1[v2f_21, np.arange(self.n2)] = curv_init_1.sum()

        # (f2, n1)
        target_curv2 = np.zeros((self.f2, self.n1))
        target_curv2[v2f_12, np.arange(self.n1)] = curv_init_2.sum()

        # Modify connections to match target curvatures
        # (e1, n2)
        trivial_con1 = self.mesh1.modify_connection_to_curvature(
            initial_connection_angle=con_angle_init_1,
            initial_curvature=curv_init_1,
            target_curvature=target_curv1,
            method=trivial_con_method,
        )

        # (e2, n1)
        trivial_con2 = self.mesh2.modify_connection_to_curvature(
            initial_connection_angle=con_angle_init_2,
            initial_curvature=curv_init_2,
            target_curvature=target_curv2,
            method=trivial_con_method,
        )

        connection = np.concatenate(
            [
                einops.rearrange(trivial_con1, "e1 n2 -> (e1 n2)"),
                einops.rearrange(trivial_con2, "e2 n1 -> (n1 e2)"),
            ]
        )  # (e1*n2 + e2*n1,)

        Lap_connection, Lap_mass, D_inv = get_slice_product_laplacian(
            connection,
            self.mesh1.edges,
            self.mesh2.edges,
            self.n1,
            self.n2,
            self.mesh1.star0,
            self.mesh2.star0,
            self.mesh1.star1,
            self.mesh2.star1,
            return_precondition=True,
        )

        if initial_guess is not None:
            if initial_guess.ndim == 1:
                assert initial_guess.shape[0] == self.n_vertices
                X = initial_guess[:, None]
            elif initial_guess.ndim == 2:
                if initial_guess.shape[1] == 1:
                    assert initial_guess.shape[0] == self.n_vertices
                    X = initial_guess.copy()
                else:
                    assert initial_guess.shape == (self.n1, self.n2)
                    X = einops.rearrange(initial_guess, "n1 n2 -> (n1 n2) 1")
            else:
                raise ValueError(
                    "initial_guess must be a 1D or 2D array with compatible shape."
                )
        else:
            # Random initial guess
            X = np.random.rand(self.n_vertices, 1) + 1j * np.random.rand(
                self.n_vertices, 1
            )

        evals, evecs = sparse.linalg.lobpcg(
            Lap_connection,
            X,
            B=Lap_mass,
            M=D_inv if use_preconditioner else None,
            largest=False,
            tol=tol,
            maxiter=1000,
            verbosityLevel=int(verbose),
        )
        print("Lobpcg converged:", evals)
        min_section = evecs[:, 0]

        return min_section, connection
