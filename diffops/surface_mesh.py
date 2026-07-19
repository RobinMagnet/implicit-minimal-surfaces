import numpy as np
from scipy import sparse

from .complex_intrinsic import ComplexIntrinsicTriangulation
from .geom_utils import compute_face_areas_from_len
from . import trivial_connection
from .sqroot_solver import sqrt_lc_undirected
from . import fem_lap
from .dec_operators import get_dec_laplacian
from .singularity_solver import find_zeros_in_triangle


class SurfaceMesh(ComplexIntrinsicTriangulation):
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        tufted_border: bool = False,
        compute_cotan_with_len: bool = True,
        allow_self_loops: bool = True,
    ):
        super().__init__(
            vertices,
            faces,
            tufted_border=tufted_border,
            compute_cotan_with_len=compute_cotan_with_len,
            allow_self_loops=allow_self_loops,
        )

        self.vertices = np.array(vertices).copy()  # (N, 3)

        # self._rho_LC = None
        self._rho_spin = None
        self._d0 = None
        self._d1 = None

    @property
    def faces(self):
        return self.mesh.faces

    @property
    def midedge(self):
        """(p, 3) Midpoint of each edge."""
        edges = self.edges  # (n_ue, 2)
        v0 = self.vertices[edges[:, 0]]  # (n_ue, 3)
        v1 = self.vertices[edges[:, 1]]  # (n_ue, 3)

        midpoints = (v0 + v1) / 2.0  # (n_ue, 3)

        return midpoints

    def flip_edge(self, he_id):
        res = super().flip_edge(he_id)

        if res:
            # self._rho_LC = None
            self._rho_spin = None
            self._d0 = None
            self._d1 = None
        return res

    @property
    def connection_angle_LC(self):
        """
        Connection angle on halfedges.

        Returns
        -------
        rho_LC : np.array
            (e,) Array of connection angles on each undirected edge (in radians between -pi and pi)
        """
        return self.get_uedge_connection_angle_LC()

    @property
    def curvature_LC(self):
        """
        Curvature on vertices.

        Returns
        -------
        curvature_LC : np.array
            (n,) Array of curvature values at each vertex
        """
        return self.get_curvature_LC()

    @property
    def curvature_spin(self):
        """
        Spin curvature on vertices.

        Returns
        -------
        omega_spin : np.array
            (n,) Array of spin curvature values at each vertex
        """
        return self.curvature_LC / 2

    @property
    def connection_angle_spin(self):
        """
        Half of the Levi-Civita connection 1-form.

        Returns
        -------
        rho_sqrtLC : np.array
            (e,) Array of half connection angles on each undirected edge (in radians between -pi and pi)
        """
        if self._rho_spin is None:
            # self._rho_sqrtLC = self.rho_LC / 2
            self._rho_spin = sqrt_lc_undirected(
                self.d1, self.connection_angle_LC, self.curvature_LC
            )
        return self._rho_spin

    def get_connection_from_skyscraper(
        self, target_curvature, faceid=None, method="hodge"
    ):
        """
        Compute connection 1-form from skyscraper curvature.

        Parameters
        ----------
        target_curvature : np.array
            (n,) Array of target curvature values at each vertex
        faceid : int, optional
            Face index to place the skyscraper curvature, by default None

        Returns
        -------
        rho_skyscraper : np.array
            (e,) Array of connection angles on each undirected edge (in radians between -pi and pi)
        """

        if faceid is None:
            faceid = 0  # Default to first face

        curv_index = target_curvature.sum() / (2 * np.pi)
        assert np.isclose(
            curv_index, round(curv_index)
        ), "Total curvature must be multiple of 2pi for skyscraper."

        curvature_skyscraper = np.zeros(self.mesh.n_faces)
        curvature_skyscraper[faceid] = target_curvature.sum()

        connection_skyscraper = np.zeros(self.mesh.n_edges)
        # print(connection_skyscraper.shape, self.star1_inv.shape, self.star1.shape)$

        rho_skyscraper = self.modify_connection_to_curvature(
            initial_connection_angle=connection_skyscraper,
            initial_curvature=curvature_skyscraper,
            target_curvature=target_curvature,
            method=method,
        )

        return rho_skyscraper

    def modify_connection_to_curvature(
        self,
        initial_connection_angle,
        initial_curvature,
        target_curvature,
        method="hodge",
    ):
        """
        Modify a given connection 1-form to achieve a target curvature.

        Parameters
        ----------
        initial_connection_angle : np.array
            (e,) Array of initial connection angles on each undirected edge (in radians between -pi and pi)
        initial_curvature : np.array
            (n,) Array of initial curvature values at each vertex
        target_curvature : np.array
            (n,) Array of target curvature values at each vertex
        method : str, optional
            Method to use for modification ('hodge' or 'l2'), by default 'hodge'

        Returns
        -------
        rho_modified : np.array
            (e,) Array of modified connection angles on each undirected edge (in radians between -pi and pi)
        """

        if method.lower() == "hodge":
            rho_modified = trivial_connection.modify_connection_to_curvature_hodge(
                d1=self.d1,
                star1_inv=self.star1_inv,
                initial_connection_angle=initial_connection_angle,
                initial_curvature=initial_curvature,
                target_curvature=target_curvature,
            )
        elif method.lower() == "l2":
            rho_modified = trivial_connection.modify_connection_to_curvature_l2(
                d1=self.d1,
                initial_connection_angle=initial_connection_angle,
                initial_curvature=initial_curvature,
                target_curvature=target_curvature,
                star1=self.star1 if self.mesh.has_flipped else None,
            )
        else:
            raise ValueError(f"Unknown method '{method}' for modifying connection.")

        return rho_modified

    @property
    def d0(self):
        """
        Discrete gradient operator.

        Returns
        -------
        d0 : sparse.coo_matrix
            (e,n) Sparse matrix of the discrete gradient operator
        """

        if self._d0 is None:
            v0, v1 = self.mesh.get_vertices(self.mesh.canonical_hedges)

            n_edges = len(v0)

            I = np.tile(np.arange(n_edges), (2,))
            J = np.concatenate((v0, v1))
            V = np.concatenate((-np.ones(n_edges), np.ones(n_edges)))

            self._d0 = sparse.csr_matrix((V, (I, J)), shape=(n_edges, self.n_vertices))
        return self._d0

    @property
    def d1(self):
        """
        Discrete curl operator.

        Returns
        -------
        d1 : sparse.coo_matrix
            (f,e) Sparse matrix of the discrete curl operator
        """

        if self._d1 is None:

            face_hedges = self.mesh.face_hedges

            is_canonical_mask = self.mesh.is_canonical_mask
            is_canonical = is_canonical_mask[face_hedges]
            n_edges = is_canonical_mask.sum()

            face_hedge_can_id = self.mesh.hedge_to_edge[face_hedges]
            # face_hedges_can = np.where(
            #     is_canonical, face_hedges, self.mesh.opposites[face_hedges]
            # )

            values = np.where(is_canonical, 1.0, -1.0)

            I = np.repeat(np.arange(self.mesh.n_faces), 3)
            J = face_hedge_can_id.flatten()
            V = values.flatten()

            self._d1 = sparse.csr_matrix(
                (V, (I, J)), shape=(self.mesh.n_faces, n_edges)
            )

        return self._d1

    @property
    def star0(self):
        """
        Hodge star operator on 0-forms (vertices).

        Returns
        -------
        star0 : sparse.coo_matrix
            (n,n) Sparse diagonal matrix of the Hodge star operator on 0-forms
        """
        vertex_areas = self.vertex_areas
        return sparse.diags(vertex_areas)

    @property
    def star0_inv(self):
        """
        Inverse Hodge star operator on 0-forms (vertices).

        Returns
        -------
        star0_inv : sparse.coo_matrix
            (n,n) Sparse diagonal matrix of the inverse Hodge star operator on 0-forms
        """
        vertex_areas = self.vertex_areas
        inv_areas = np.zeros_like(vertex_areas)
        nonzero_mask = ~np.isclose(vertex_areas, 0.0)
        inv_areas[nonzero_mask] = 1.0 / vertex_areas[nonzero_mask]
        return sparse.diags(inv_areas)

    @property
    def star1(self):
        """
        Hodge star operator on 1-forms (edges).

        Returns
        -------
        star1 : sparse.coo_matrix
            (e,e) Sparse diagonal matrix of the Hodge star operator on 1-forms
        """
        # hedge_can = self.mesh.canonical_hedges  # (e,)
        # cots = self.cotan_weights_hedge[hedge_can]  # (e)

        # hedge_can_opp = self.mesh.opposites[hedge_can]
        # is_can_he_interior = self.mesh.is_interior(hedge_can)
        # # canonical_int = hedge_can[interior_he]

        # if self.tufted_border:
        #     cots_opp = np.where(
        #         is_can_he_interior, self.cotan_weights_hedge[hedge_can_opp], cots
        #     )
        # else:
        #     cots_opp = np.where(
        #         is_can_he_interior, self.cotan_weights_hedge[hedge_can_opp], 0
        #     )

        cotan_weights = self.cotan_weight_edge

        return sparse.diags(cotan_weights)

    @property
    def star1_inv(self):
        """
        Inverse Hodge star operator on 1-forms (edges).

        Returns
        -------
        star1_inv : sparse.coo_matrix
            (e,e) Sparse diagonal matrix of the inverse Hodge star operator on 1-forms
        """
        cotan_weights = self.cotan_weight_edge
        inv_cotans = np.zeros_like(cotan_weights)
        nonzero_mask = ~np.isclose(cotan_weights, 0.0)
        inv_cotans[nonzero_mask] = 1.0 / cotan_weights[nonzero_mask]
        return sparse.diags(inv_cotans)

    @property
    def star2(self):
        """
        Hodge star operator on 2-forms (faces).

        Returns
        -------
        star2 : sparse.coo_matrix
            (f,f) Sparse diagonal matrix of the Hodge star operator on 2-forms
        """
        face_areas = self.face_areas
        return sparse.diags(face_areas)

    @property
    def star2_inv(self):
        """
        Inverse Hodge star operator on 2-forms (faces).

        Returns
        -------
        star2_inv : sparse.coo_matrix
            (f,f) Sparse diagonal matrix of the inverse Hodge star operator on 2-forms
        """
        face_areas = self.face_areas
        inv_areas = np.zeros_like(face_areas)
        nonzero_mask = ~np.isclose(face_areas, 0.0)
        inv_areas[nonzero_mask] = 1.0 / face_areas[nonzero_mask]
        return sparse.diags(inv_areas)

    def connection_d0(self, connection_angle):
        """
        Connection angle is (e,) array of connection angles on each undirected edge (in radians between -pi and pi)
        """

        hedges_can = self.mesh.canonical_hedges
        v0, v1 = self.mesh.get_vertices(hedges_can)

        # connection = np.exp(1j * connection_angle)  # (n_he)
        connection_can = np.exp(1j * connection_angle)

        n_edges = len(hedges_can)

        I = np.tile(np.arange(n_edges), (2,))
        J = np.concatenate((v0, v1))
        V = np.concatenate((-connection_can, np.ones(n_edges)))

        d0 = sparse.csr_matrix((V, (I, J)), shape=(n_edges, self.n_vertices))
        return d0

    def connection_laplacian(self, connection_angle):

        d0 = self.connection_d0(connection_angle)

        lap_conn = get_dec_laplacian(d0, self.star1)

        return lap_conn

    def bundle_laplacian_fem(self, connection_angle, curvature, cutoff=1e-2):
        """
        Connection Laplacian operator for connection 1-form of the mesh, discretized via finite element method.

        Parameters
        ----------
        connection : np.array
            (e,) Array of connection angles on each undirected edge (in radians between -pi and pi)
        curvature : np.array
            (m,) Array of curvature on each face (in radians, any value)

        Returns
        -------
        lap_bundle : sparse.coo_matrix
            (n,n) Sparse matrix of the Connection Laplacian operator for connection 1-form of the mesh
        """
        N = self.n_vertices
        face_hedges = self.mesh.face_hedges  # (m,3)

        face_areas = compute_face_areas_from_len(
            self.hedge_lengths[face_hedges]
        )  # (m,)

        hedge_connection_angle = np.where(
            self.mesh.is_canonical_mask,
            connection_angle[self.mesh.hedge_to_edge],
            -connection_angle[self.mesh.hedge_to_edge],
        )

        face_connection_angle = hedge_connection_angle[face_hedges]  # (m,3)

        sqlen_opp = np.square(self.hedge_lengths)[face_hedges]  # (m,3)

        # sqlen_adj1: Edge lengths adjacent to vertex i (CCW next)
        # For col 0 (opp v0), this is edge opp v2 (v0->v1) -> index 2
        sqlen_adj1 = sqlen_opp[:, [2, 0, 1]]  # (m,3)

        # sqlen_adj2: Edge lengths adjacent to vertex i (CCW prev)
        # For col 0 (opp v0), this is edge opp v1 (v2->v0) -> index 1
        sqlen_adj2 = sqlen_opp[:, [1, 2, 0]]  # (m,3)

        # Dot product of edge vectors at vertex i: <e_ij, e_ik>
        inprod_edges = (sqlen_adj1 + sqlen_adj2 - sqlen_opp) / 2

        # 3. Diagonal Term (V_diag)
        # Formula: 1/(4A) * ( |e_jk|^2 + Omega^2/90 * (|e_ij|^2 + <e_ij, e_ik> + |e_ik|^2) )
        V_diag = sqlen_opp + curvature[:, None] ** 2 / 90 * (
            sqlen_adj1 + sqlen_adj2 + inprod_edges
        )
        V_diag = V_diag / (4 * face_areas[:, None])  # (m,3)

        I_diag = self.mesh.faces.flatten()
        J_diag = self.mesh.faces.flatten()
        V_diag = V_diag.flatten()

        # V_diag = np.tile(faces_area / 6, 3)

        # Define source (j) and target (k) indices for these blocks
        # Col 0 of off_weights corresponds to edge opposite v0 => v1->v2
        I_off_block = self.mesh.faces[:, [1, 2, 0]]
        J_off_block = self.mesh.faces[:, [2, 0, 1]]

        f1_vals = fem_lap.get_f1(curvature, cutoff=cutoff)  # (m,)
        f2_vals = fem_lap.get_f2(curvature, cutoff=cutoff)  # (m,)

        # Note: f2 converges to -0.25, effectively providing the -1/4 factor
        # for the standard cotan laplacian when omega=0.
        off_weights = (sqlen_adj1 + sqlen_adj2) * f1_vals[
            :, None
        ] + inprod_edges * f2_vals[:, None]
        off_weights = (
            off_weights / face_areas[:, None] * np.exp(-1j * face_connection_angle)
        )  # (m,3)

        I = np.concatenate([I_diag, I_off_block.flatten(), J_off_block.flatten()])
        J = np.concatenate([J_diag, J_off_block.flatten(), I_off_block.flatten()])
        V = np.concatenate(
            [V_diag, off_weights.flatten(), np.conj(off_weights).flatten()]
        )

        L = sparse.csr_matrix((V, (I, J)), shape=(N, N))

        return L

    def bundle_mass_fem(self, connection_angle, curvature, cutoff=1e-2):
        """
        Mass matrix for connection 1-form of the mesh, discretized via finite element method.

        Parameters
        ----------
        connection_angle : np.array
            (e,) Array of connection angles on each undirected edge (in radians between -pi and pi)
        curvature : np.array
            (m,) Array of curvature on each face (in radians, any value)

        Returns
        -------
        mass_bundle : sparse.coo_matrix
            (n,n) Sparse matrix of the mass matrix for connection 1-form of the mesh
        """
        # 1. Orient the connection angles

        face_hedges = self.mesh.face_hedges  # (m,3)

        face_areas = compute_face_areas_from_len(
            self.hedge_lengths[face_hedges]
        )  # (m,)
        c_factors = fem_lap.get_curvature_factor(curvature, cutoff=cutoff)  # (m,)

        hedge_connection_angle = np.where(
            self.mesh.is_canonical_mask,
            connection_angle[self.mesh.hedge_to_edge],
            -connection_angle[self.mesh.hedge_to_edge],
        )

        face_connection_angle = hedge_connection_angle[face_hedges]  # (m,3)

        off_weights = (face_areas * c_factors)[:, None] * np.exp(
            -1j * face_connection_angle
        )  # (m,3)

        I_diag = self.mesh.faces.flatten()
        J_diag = self.mesh.faces.flatten()
        V_diag = np.repeat(face_areas / 6, 3)

        I_off_block = self.mesh.faces[:, [1, 2, 0]]
        J_off_block = self.mesh.faces[:, [2, 0, 1]]

        I = np.concatenate([I_diag, I_off_block.flatten(), J_off_block.flatten()])
        J = np.concatenate([J_diag, J_off_block.flatten(), I_off_block.flatten()])
        V = np.concatenate(
            [V_diag, off_weights.flatten(), np.conj(off_weights).flatten()]
        )

        N = self.n_vertices
        M = sparse.csr_matrix((V, (I, J)), shape=(N, N))
        return M

    def compute_rotation_form(self, section, connection_angle):
        """
        Computes the rotation 1-form from a section and a connection 1-form.

        Parameters
        ----------
        section : np.array
            (n_vertices,) or (n_vertices,p) Array of section values (complex numbers)
        connection_angle : np.array
            (n_edges,) or (n_edges, p) Array of connection angles on each undirected edge (in radians between -pi and pi)
        """
        vi, vj = self.mesh.get_vertices(self.mesh.canonical_hedges)
        section_j = section[vj]
        section_i = section[vi]

        if connection_angle.ndim == section.ndim:
            rot_form = np.angle(
                section_j * np.conj((section_i * np.exp(1j * connection_angle)))
            )
        elif connection_angle.ndim + 1 == section.ndim:
            rot_form = np.angle(
                section_j
                * np.conj((section_i * np.exp(1j * connection_angle[:, None])))
            )
        else:
            raise ValueError(
                "connection_angle and section have incompatible dimensions."
            )

        return rot_form

    def compute_index_form(
        self, curvature_form, rotform=None, section=None, connection_angle=None
    ):
        """
        Computes the index 2-form from a curvature 2-form and a rotation 1-form.
        """

        if rotform is None:
            assert section is not None
            assert connection_angle is not None
            rotform = self.compute_rotation_form(section, connection_angle)

        if curvature_form.ndim == rotform.ndim:
            index_form = (self.d1 @ rotform + curvature_form) / (2 * np.pi)  # (F,)
        elif curvature_form.ndim + 1 == rotform.ndim:
            index_form = (self.d1 @ rotform + curvature_form[:, None]) / (
                2 * np.pi
            )  # (F,p)
        else:
            raise ValueError("curvature_form and rotform have incompatible dimensions.")

        return index_form

    def get_basis_functions_at_barycentric(
        self, face_inds, bary_coords, connection_angle
    ):
        target_face_hedges = self.mesh.face_hedges[face_inds]  # (p,3)

        hedge_connection_angle = np.where(
            self.mesh.is_canonical_mask,
            connection_angle[self.mesh.hedge_to_edge],
            -connection_angle[self.mesh.hedge_to_edge],
        )  # (n_he,)

        rho_vals = hedge_connection_angle[target_face_hedges]

        # 2. Barycentric Coordinates (Linear)
        # Assuming bary_coords is shape (N, 3) and rows sum to 1.
        b_i = bary_coords[:, 0]
        b_j = bary_coords[:, 1]
        b_k = bary_coords[:, 2]

        rho_ij = rho_vals[:, 2]  # opp v0
        rho_jk = rho_vals[:, 0]  # opp v1
        rho_ki = rho_vals[:, 1]  # opp v2

        phi_i = b_i * np.exp(1j * rho_ij * b_j) * np.exp(-1j * rho_ki * b_k)
        phi_j = b_j * np.exp(1j * rho_jk * b_k) * np.exp(-1j * rho_ij * b_i)
        phi_k = b_k * np.exp(1j * rho_ki * b_i) * np.exp(-1j * rho_jk * b_j)

        return np.stack([phi_i, phi_j, phi_k], axis=1)  # Shape (N, 3)

    def evaluate_section_at_barycentric(
        self, section, face_inds, bary_coords, connection_angle
    ):
        z_vals = section[self.mesh.faces[face_inds]]  # (p, 3)

        Phi_vals = self.get_basis_functions_at_barycentric(
            face_inds, bary_coords, connection_angle
        )  # (p, 3)

        # section_eval = z_i * phi_i + z_j * phi_j + z_k * phi_k
        section_eval = np.sum(z_vals * Phi_vals, axis=1)  # (p,)

        return section_eval

    def build_prolongation_matrix(self, face_inds, bary_coords, connection_angle):
        """
        Constructs a sparse matrix (N_fine, N_coarse) that interpolates values.
        """
        n_fine = len(face_inds)
        n_coarse = self.n_vertices

        weights = self.get_basis_functions_at_barycentric(
            face_inds, bary_coords, connection_angle
        )  # (n_fine, 3)

        # 2. Build CSR Matrix indices
        # We need to place weights[i, 0] at (i, v0), weights[i, 1] at (i, v1), etc.

        coarse_triangles = self.faces[face_inds]  # (n_fine, 3)

        # Flatten arrays for COO construction
        row_indices = np.repeat(np.arange(n_fine), 3)  # [0, 0, 0, 1, 1, 1, ...]
        col_indices = coarse_triangles.ravel()  # [v0_0, v1_0, v2_0, v0_1, ...]
        data = weights.ravel()  # [w0_0, w1_0, w2_0, w0_1, ...]

        # Construct Sparse Matrix
        P = sparse.csr_matrix(
            (data, (row_indices, col_indices)), shape=(n_fine, n_coarse)
        )

        return P

    def find_zeros(
        self,
        section,
        connection_angle,
        curvature_form,
        index_form=None,
        verbose=False,
    ):
        """
        Finds an arbitraty zero of one or multiple sections on the mesh.

        Parameters
        ----------
        section : np.array
            (n_vertices,) Array of section values (complex numbers)
        connection_angle : np.array
            (n_edges,) Array of connection angles on each undirected edge (in radians between -pi and pi)

        Returns
        -------
        zero_faces : list of int
            List of face indices that contain zeros.
        """
        if index_form is None:
            index_form = self.compute_index_form(
                curvature_form,
                rotform=None,
                section=section,
                connection_angle=connection_angle,
            )

        target_faces = np.flatnonzero(np.abs(index_form) > 0.5)  # (p)

        n_zeros = target_faces.shape[0]
        target_indices = index_form[target_faces]

        if verbose:
            print(f"Found {n_zeros} zeros")

        if not np.allclose(np.abs(target_indices), 1):
            unique, counts = np.unique(target_indices, return_counts=True)
            count_dict = dict(zip(unique, counts))
            print(f"Warning: Index > 1 found: {count_dict}")

        # --- B. Prepare Values for Zero Finding ---
        z_vals = section[self.mesh.faces[target_faces]]  # (p, 3)
        omega_vals = curvature_form[target_faces]  # (p,)

        target_face_hedges = self.mesh.face_hedges[target_faces]  # (p,3)

        hedge_connection_angle = np.where(
            self.mesh.is_canonical_mask,
            connection_angle[self.mesh.hedge_to_edge],
            -connection_angle[self.mesh.hedge_to_edge],
        )  # (n_he,)

        rho_vals = hedge_connection_angle[target_face_hedges]  # (p,3)

        bary_coods, valid_mask = find_zeros_in_triangle(
            z_vals,
            rho_vals,
            omega_vals,
            n_steps=10,
            n_newton=100,
            skip_converged=True,
            convergence_tol=1e-8,
            return_history=False,
            verbose=verbose,
        )

        bary_coods = np.where(
            valid_mask[:, None], bary_coods, np.ones(3)[None] / 3
        )  # (p,3)

        return target_faces, bary_coods

    def find_single_zero_on_sections(
        self, sections, connection_angle, curvature_form, index_form=None, verbose=False
    ):
        """
        Finds an arbitraty zero of one or multiple sections on the mesh.

        Parameters
        ----------
        section : np.array
            (n_vertices,p) Array of sections values (complex numbers)
        connection_angle : np.array
            (n_edges,) or (n_edges, p) Array of connection angles on each undirected edge (in radians between -pi and pi)
        curvature_form : np.array
            (n_faces,) or (n_faces,p) Array of curvature on each face (in radians, any value)

        Returns
        -------
        zero_faces : list of int
            List of face indices that contain zeros.
        """
        if index_form is None:
            # (f1, p)
            index_form = self.compute_index_form(
                curvature_form,
                rotform=None,
                section=sections,
                connection_angle=connection_angle,
            )
        p = sections.shape[1]

        zeros_mask = np.abs(index_form) > 0.5  # (f1, p)

        target_faces = np.argmax(zeros_mask & (index_form > 0), axis=0)  # (p)

        n_zeros_per_section = zeros_mask.sum(0)  # (p,)
        target_indices = index_form[target_faces, np.arange(p)]  # (p,)

        # if verbose:
        #     print(f"Found {n_zeros_per_section} zeros")
        if not np.allclose(n_zeros_per_section, 1):
            unique, counts = np.unique(n_zeros_per_section, return_counts=True)
            count_dict = dict(zip(unique, counts))
            print(f"Warning: Not all sections have exactly one zero: {count_dict}")
        if not np.allclose(np.abs(target_indices), 1):
            unique, counts = np.unique(target_indices, return_counts=True)
            count_dict = dict(zip(unique, counts))
            print(f"Warning: Index != +-1 found: {count_dict}")

        # --- B. Prepare Values for Zero Finding ---

        target_face_vertices = self.mesh.faces[target_faces]  # (p, 3)

        z_vals = sections[target_face_vertices, np.arange(p)[:, None]]  # (p, 3)

        # Get curvature values for target faces
        if curvature_form.ndim == 1:
            omega_vals = curvature_form[target_faces]  # (p,)
        else:
            omega_vals = curvature_form[target_faces, np.arange(p)]  # (p,)

        target_face_hedges = self.mesh.face_hedges[target_faces]  # (p,3)

        if connection_angle.ndim == 1:
            hedge_connection_angle = np.where(
                self.mesh.is_canonical_mask,
                connection_angle[self.mesh.hedge_to_edge],
                -connection_angle[self.mesh.hedge_to_edge],
            )  # (n_he,)
            rho_vals = hedge_connection_angle[target_face_hedges]  # (p, 3)
        else:
            # connection_angle is (n_edges, p)
            hedge_connection_angle = np.where(
                self.mesh.is_canonical_mask[:, None],
                connection_angle[self.mesh.hedge_to_edge],
                -connection_angle[self.mesh.hedge_to_edge],
            )  # (n_he, p)
            rho_vals = hedge_connection_angle[
                target_face_hedges, np.arange(p)[:, None]
            ]  # (p, 3)

        bary_coods, valid_mask = find_zeros_in_triangle(
            z_vals,
            rho_vals,
            omega_vals,
            n_steps=10,
            n_newton=100,
            skip_converged=True,
            convergence_tol=1e-8,
            return_history=False,
            verbose=verbose,
        )

        bary_coods = np.where(
            valid_mask[:, None], bary_coods, np.ones(3)[None] / 3
        )  # (p,3)

        return target_faces, bary_coods
