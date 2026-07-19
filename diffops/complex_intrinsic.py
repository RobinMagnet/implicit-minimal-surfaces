import numpy as np

from .intrinsic_mesh import IntrinsicTriangulation
from .geom_utils import compute_internal_angles_from_length_stable, compute_angle_at_c
from .utils import wrap_to_pi


class ComplexIntrinsicTriangulation(IntrinsicTriangulation):
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        tufted_border: bool = False,
        compute_cotan_with_len: bool = True,
        allow_self_loops: bool = True,
    ):
        # 1. Initialize Topology & Geometry (Current State)
        super().__init__(
            vertices,
            faces,
            tufted_border=tufted_border,
            compute_cotan_with_len=compute_cotan_with_len,
            allow_self_loops=allow_self_loops,
        )

        # 2. Complex Structure Attributes (Current State)
        self._vertex_angle_sums = None
        self.hedge_phi = np.zeros(self.mesh.n_hedges, dtype=float)

        self.is_boundary_vertex = np.zeros(self.n_vertices, dtype=bool)
        self.is_boundary_vertex[self.mesh.boundary_vertices] = True

        # Calculate initial Phis and Vertex sums
        self._start_he_orig = None
        self._initialize_complex_structure()

        # 3. Snapshot of the ORIGINAL Structure
        # We need this because 'flip_edge' reuses indices for new diagonals,
        # so we lose the notion of "original edge i" in the main structure.
        # self.mesh_original = copy.deepcopy(self.mesh)
        self.mesh_original = self.mesh.copy()
        self.hedge_phi_original = self.hedge_phi.copy()
        self.hedge_lengths_original = self.hedge_lengths.copy()

    @property
    def faces_extrinsic(self):
        """Return the faces of the mesh in extrinsic coordinates."""
        return self.mesh_original.faces

    @property
    def vertex_angle_sums(self):
        """Sum of internal angles at each vertex."""
        return self._vertex_angle_sums

    @property
    def normalized_angles(self):
        """
        (F, 3) Internal angles scaled so they sum to 2pi at every vertex.
        This gives the current angles even if edge flips were performed
        Used for the 'flat' complex metric.
        """
        # Get raw Euclidean angles from current edge lengths
        raw_angles = self.face_angles  # (F, 3)

        # Map faces to vertices to divide by vertex sums
        # self.faces is (F, 3) vertex indices
        v_sums = self._vertex_angle_sums[self.faces]  # (F, 3)

        is_boundary = self.is_boundary_vertex[self.faces]
        target_sums = np.where(is_boundary, np.pi, 2 * np.pi)

        return raw_angles * (target_sums / v_sums)

    def _initialize_complex_structure(self):
        """Calculates initial vertex angle sums and assigns initial phi coordinates."""

        # --- A. Compute Vertex Angle Sums ---
        n_v = self.n_vertices
        self._vertex_angle_sums = np.zeros(n_v)

        # face_angles is (F, 3). Flatten and add to vertex indices
        flat_angles = self.face_angles.flatten()
        flat_vids = self.faces.flatten()

        np.add.at(self._vertex_angle_sums, flat_vids, flat_angles)

        # --- B. Compute Hedge Phi (Angle Coordinates) ---

        # 1. Get starting edges for every vertex (Handles borders correctly)
        # start_he = self._get_vertex_start_hedges(self.mesh)  # (n_v,)
        start_he = self.mesh.v_hedge
        self._start_he_orig = start_he.copy()

        # 2. Get normalized angles for the flat metric
        norm_angles = self.normalized_angles  # (F, 3)

        # 3. Propagate around vertices
        # Initialize phi to 0 for the starting edge of every vertex
        curr_he = start_he.copy()  # (n_v,)
        self.hedge_phi[curr_he] = 0.0

        # Keep track of active vertices (those that haven't hit a border or wrapped around)
        active_mask = np.ones(n_v, dtype=bool)

        # Safety limit for iteration (max degree usually < 50)
        for _ in range(self.mesh.n_faces + 1):
            if not np.any(active_mask):
                break

            active_indices = np.flatnonzero(active_mask)

            # Get current active half-edges
            h_curr = curr_he[active_indices]
            h_prev = self.mesh.get_prev(h_curr)
            h_next = self.mesh.opposites[h_prev]

            hit_border = self.mesh.is_boundary(h_prev)
            wrapped = h_next == start_he[active_indices]

            keep_going = ~(hit_border | wrapped)
            if not np.any(keep_going):
                break

            survivor_curr = h_curr[keep_going]
            survivor_next = h_next[keep_going]

            # Incoming angle is the angle opposite to the next hedge in the face
            angles = self.mesh.gather_from_face_data(
                norm_angles, self.mesh.get_next(survivor_curr)
            )

            self.hedge_phi[survivor_next] = self.hedge_phi[survivor_curr] + angles

            # 3. Advance State
            # Update pointers for survivors
            curr_he[active_indices[keep_going]] = survivor_next

            # Update the global mask efficiently
            # We simply write the 'keep_going' status back into the active slots
            active_mask[active_indices] = keep_going

    def flip_edge(self, he_id):
        r"""
        Flips the shared edge (v1-v3) to (v2-v4) and updates the complex
        angle coordinates (hedge_phi) for the new diagonal.

        Transformation Diagram
        ----------------------

        Before Flip:                 After Flip:
        Shared Edge: (v1, v3)        Shared Edge: (v2, v4)

              v2                            v2
             /  \                         / | \
            / f0  \                      /  |  \
          v1------v3                    v1  |   v3
            \ f1 /                       \  |  /
             \  /                         \ | /
              v4                            v4

        Vertices & Faces (CCW):      Vertices & Faces (CCW):
        f0: (v1, v3, v2)             f0: (v1, v4, v2)
        f1: (v1, v4, v3)             f1: (v4, v3, v2)

        Parameters
        ----------
        he_id : int
            Index of the half-edge to be flipped.

        Returns
        -------
        bool
            True if the flip was successful, None if the edge is a border edge.
        """
        mesh = self.mesh
        he_13_pre = he_id
        he_31_pre = mesh.opposites[he_13_pre]

        # Cannot flip a boundary edge
        if he_31_pre == -1:
            return None

        he_21_pre = mesh.get_prev(he_13_pre)
        he_43_pre = mesh.get_prev(he_31_pre)

        v2 = mesh.get_source_vid(he_21_pre)
        v4 = mesh.get_source_vid(he_43_pre)

        phi_21 = self.hedge_phi[he_21_pre]
        phi_43 = self.hedge_phi[he_43_pre]
        phi_32 = self.hedge_phi[mesh.get_next(he_13_pre)]
        phi_14 = self.hedge_phi[mesh.get_next(he_31_pre)]

        new_indices = super().flip_edge(he_id)
        if new_indices is None:
            return None

        he_42, he_24 = new_indices

        # Remap saved phis to new half-edges indices
        he_21 = mesh.get_next(he_42)
        he_14 = mesh.get_prev(he_42)
        he_32 = mesh.get_prev(he_24)
        he_43 = mesh.get_next(he_24)

        self.hedge_phi[he_21] = phi_21
        self.hedge_phi[he_14] = phi_14
        self.hedge_phi[he_32] = phi_32
        self.hedge_phi[he_43] = phi_43

        # 3. Compute Angles Locally
        # We can now reliably fetch the new faces using the new handles
        f_A = mesh.get_face(he_42)
        f_B = mesh.get_face(he_24)

        # hes_A = [he_42, mesh.get_next(he_42), mesh.get_prev(he_42)]
        l_42 = self.hedge_lengths[he_42]
        l_21 = self.hedge_lengths[he_21]
        l_14 = self.hedge_lengths[he_14]

        l_43 = self.hedge_lengths[he_43]
        l_32 = self.hedge_lengths[he_32]

        len_mat = np.array(
            [
                [l_42, l_21, l_14],
                [l_42, l_43, l_32],
            ]
        )
        angle_v2, angle_v4 = compute_angle_at_c(
            len_mat[:, 0], len_mat[:, 1], len_mat[:, 2]
        )
        # angle_v2 = compute_angle_at_c(l_42[None], l_21[None], l_14[None])[0]
        # angle_v4 = compute_angle_at_c(l_42[None], l_43[None], l_32[None])[0]

        # 4. UPDATE PHI
        # -------------
        # Update he_42 (v4->v2): Connects to saved phi_43
        scale_v4 = (
            np.pi if self.is_boundary_vertex[v4] else 2 * np.pi
        ) / self._vertex_angle_sums[v4]
        self.hedge_phi[he_42] = phi_43 + angle_v4 * scale_v4

        # Update he_24 (v2->v4): Connects to saved phi_21
        # angle_v2 = angles_A[2]  # Angle at v2 in Face A
        scale_v2 = (
            np.pi if self.is_boundary_vertex[v2] else 2 * np.pi
        ) / self._vertex_angle_sums[v2]
        self.hedge_phi[he_24] = phi_21 + angle_v2 * scale_v2

        return new_indices

    def rebuild_vfield(self, vfield, vertices, source_indices=None):
        """
        Reconstructs 3D vectors from the intrinsic field on the ORIGINAL mesh.
        """
        if source_indices is None:
            source_indices = np.arange(self.n_vertices)

        source_indices = np.asarray(source_indices)

        assert vfield.shape[0] == len(source_indices)

        # 1. SETUP: Use the ORIGINAL mesh topology/geometry
        # We must trace on the original mesh because that is where the 3D coordinates exist.
        mesh = self.mesh_original
        hedge_phi = self.hedge_phi_original

        if not self.mesh.has_flipped:
            # l_orig = self.hedge_lengths
            angles_orig = self.face_angles
            v_sums_orig = self.vertex_angle_sums
        else:
            l_orig = self._compute_initial_hedge_lengths(
                vertices, mesh.faces
            )  # (n_he,)
            angles_orig = compute_internal_angles_from_length_stable(
                l_orig[mesh.face_hedges]
            )  # (m,3)
            # Original Vertex Sums (for normalization)
            v_sums_orig = np.zeros(self.n_vertices)
            np.add.at(v_sums_orig, mesh.faces.flatten(), angles_orig.flatten())
            # raise ValueError("qeqdqs")
        # # Original Edge Lengths & Angles
        # l_orig = self._compute_initial_lengths(vertices, mesh.faces)
        # angles_orig = compute_internal_angles_from_length_stable(
        #     l_orig[mesh.face_hedges]
        # )

        # # Original Vertex Sums (for normalization)
        # v_sums_orig = np.zeros(self.n_vertices)
        # np.add.at(v_sums_orig, mesh.faces.flatten(), angles_orig.flatten())

        # -------------------------------------------------
        # 2. PREPARE VECTORS
        # -------------------------------------------------
        # Filter to requested vertices
        magnitudes = np.abs(vfield)

        # Ensure angles are [0, 2pi)
        target_phis = np.angle(vfield)
        target_phis[target_phis < 0] += 2 * np.pi

        # -------------------------------------------------
        # 3. SEARCH (INTRINSIC METRIC)
        # -------------------------------------------------
        # We start at the canonical edge for every vertex
        # curr_hedges = self._get_vertex_start_hedges(mesh)[source_indices]
        start_he = self._start_he_orig
        curr_hedges = start_he[source_indices]

        final_vectors = np.zeros((len(source_indices), 3))
        found_mask = np.zeros(len(source_indices), dtype=bool)
        subset_idx = np.arange(len(source_indices))

        for _ in range(self.mesh.n_faces + 1):  # Safe iteration limit
            if len(curr_hedges) == 0:
                break

            # --- A. Compute Intrinsic Width of this Wedge ---
            # Wedge is the corner between curr_he and next(curr_he)

            # 1. Start Phi
            phi_start = hedge_phi[curr_hedges]

            # 2. Geometric Angle (from Original Mesh)
            # The angle is at the source of curr_he.
            # In the faces table, this is the angle opposite the NEXT edge.
            next_hes = mesh.get_next(curr_hedges)
            geo_angles = angles_orig[next_hes // 3, next_hes % 3]

            # 3. Scaling Factor (Euclidean -> Flat/Cone Metric)
            # This maps the physical angle sum to 2pi (or pi for border)
            v_ids = source_indices[subset_idx]
            is_boundary = self.is_boundary_vertex[v_ids]
            target_sum = np.where(is_boundary, np.pi, 2 * np.pi)

            scale = target_sum / v_sums_orig[v_ids]

            # 4. End Phi
            phi_width = geo_angles * scale
            phi_end = phi_start + phi_width

            # --- B. Check Intersection ---
            targets = target_phis[subset_idx]

            # Tolerance for float errors
            in_wedge = (targets >= phi_start - 1e-8) & (targets <= phi_end + 1e-8)

            # Check if we hit a boundary wall (end of fan)
            prev_he = mesh.get_prev(curr_hedges)
            next_in_fan = mesh.opposites[prev_he]
            at_fan_end = mesh.is_boundary(prev_he)

            # If at end of fan and vector is beyond end, we clamp/accept it
            should_process = in_wedge | (at_fan_end & (targets > phi_end))

            if np.any(should_process):
                hit_idx = subset_idx[should_process]

                # --- C. RECONSTRUCTION (EUCLIDEAN METRIC) ---
                # 1. Calculate Ratio 't' (0.0 to 1.0) inside the wedge
                # This is the purely topological position of the vector
                t = (targets[should_process] - phi_start[should_process]) / phi_width[
                    should_process
                ]
                t = np.clip(t, 0.0, 1.0)

                # 2. Map 't' to Euclidean Angle
                # We forget about phi now. We just use the physical angle.
                theta_geo = t * geo_angles[should_process]

                # 3. Build 3D Basis
                # We need the two edges of the face in 3D: u-v and w-v
                h_hits = curr_hedges[should_process]

                # v (Source)
                v_idx = v_ids[should_process]
                # u (Target of h)

                u_idx = mesh.get_dest_vid(h_hits)
                w_idx = mesh.get_dest_vid(mesh.get_next(h_hits))

                p_v = vertices[v_idx]
                p_u = vertices[u_idx]
                p_w = vertices[w_idx]

                # Basis vector 1: Edge along start of wedge
                e1 = p_u - p_v
                e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12

                # Basis vector 2: Orthonormal in the plane
                # Project edge (v->w) onto local frame
                vec_w = p_w - p_v
                dot = np.einsum("ij,ij->i", vec_w, e1)
                e2 = vec_w - dot[:, None] * e1
                e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12

                # 4. Final Vector
                mags = magnitudes[hit_idx]
                vecs = mags[:, None] * (
                    np.cos(theta_geo)[:, None] * e1 + np.sin(theta_geo)[:, None] * e2
                )

                final_vectors[hit_idx] = vecs
                found_mask[hit_idx] = True

            # --- D. Advance ---
            # Keep only those not found
            still_active = ~found_mask[subset_idx]

            # If hit boundary end, we can't go further
            valid_next = still_active & (~at_fan_end)

            if not np.any(valid_next):
                break

            subset_idx = subset_idx[valid_next]
            curr_hedges = next_in_fan[valid_next]

        return final_vectors

    def vfield_to_face(self, vfield, vertices, source_indices=None, on_intrinsic=True):
        """
        Find the face index that each vector in the vector field points into.

        Parameters
        ----------
        vfield : np.ndarray
            (n,) Complex array representing vectors in phi coordinates
        source_indices : np.ndarray, optional
            (n,) Vertex indices where vectors originate. If None, uses all vertices.
        on_intrinsic : bool, default=True
            If True, search on current (possibly flipped) mesh.
            If False, search on original mesh.

        Returns
        -------
        face_indices : np.ndarray
            (n,) Face index for each vector. -1 if not found (shouldn't happen for interior vertices).
        """
        if source_indices is None:
            source_indices = np.arange(self.n_vertices)

        source_indices = np.asarray(source_indices)
        assert vfield.shape[0] == len(source_indices)

        # Choose which mesh to search on
        if on_intrinsic:
            mesh = self.mesh
            hedge_phi = self.hedge_phi
            angles = self.normalized_angles
            v_sums = self.vertex_angle_sums
        else:
            mesh = self.mesh_original
            hedge_phi = self.hedge_phi_original

            if not self.mesh.has_flipped:
                angles = self.face_angles
                v_sums = self.vertex_angle_sums
            else:
                l_orig = self._compute_initial_hedge_lengths(vertices, mesh.faces)
                angles = compute_internal_angles_from_length_stable(
                    l_orig[mesh.face_hedges]
                )
                v_sums = np.zeros(self.n_vertices)
                np.add.at(v_sums, mesh.faces.flatten(), angles.flatten())

        # Extract target angles from complex vectors
        target_phis = np.angle(vfield)
        target_phis[target_phis < 0] += 2 * np.pi

        # Start at canonical edge for each vertex
        start_he = self._start_he_orig if not on_intrinsic else mesh.v_hedge
        curr_hedges = start_he[source_indices]

        face_indices = np.full(len(source_indices), -1, dtype=int)
        found_mask = np.zeros(len(source_indices), dtype=bool)
        subset_idx = np.arange(len(source_indices))

        for _ in range(mesh.n_faces + 1):  # Safe iteration limit
            if len(curr_hedges) == 0:
                break

            # Compute wedge boundaries in phi coordinates
            phi_start = hedge_phi[curr_hedges]

            # Get geometric angle at source vertex
            next_hes = mesh.get_next(curr_hedges)
            geo_angles = angles[next_hes // 3, next_hes % 3]

            # Scale to flat metric
            v_ids = source_indices[subset_idx]
            is_boundary = self.is_boundary_vertex[v_ids]
            target_sum = np.where(is_boundary, np.pi, 2 * np.pi)
            scale = target_sum / v_sums[v_ids]

            phi_width = geo_angles * scale
            phi_end = phi_start + phi_width

            # Check if vector falls in this wedge
            targets = target_phis[subset_idx]
            in_wedge = (targets >= phi_start - 1e-8) & (targets <= phi_end + 1e-8)

            # Handle boundary case
            prev_he = mesh.get_prev(curr_hedges)
            next_in_fan = mesh.opposites[prev_he]
            at_fan_end = mesh.is_boundary(prev_he)

            should_process = in_wedge | (at_fan_end & (targets > phi_end))

            if np.any(should_process):
                hit_idx = subset_idx[should_process]
                # The face is the one containing the current half-edge
                face_indices[hit_idx] = mesh.get_face(curr_hedges[should_process])
                found_mask[hit_idx] = True

            # Advance to next wedge
            still_active = ~found_mask[subset_idx]
            valid_next = still_active & (~at_fan_end)

            if not np.any(valid_next):
                break

            subset_idx = subset_idx[valid_next]
            curr_hedges = next_in_fan[valid_next]

        return face_indices

    def bary_to_vfield(self, face_indices, bary_coords, on_intrinsic=True):

        mesh = self.mesh if on_intrinsic else self.mesh_original
        faces = mesh.faces
        hedge_lengths = (
            self.hedge_lengths if on_intrinsic else self.hedge_lengths_original
        )
        hedge_phi = self.hedge_phi if on_intrinsic else self.hedge_phi_original

        if on_intrinsic:
            normalized_angles = self.normalized_angles  # (F, 3)
        else:
            # Get raw Euclidean angles from original edge lengths
            raw_angles = compute_internal_angles_from_length_stable(
                hedge_lengths[mesh.face_hedges]
            )  # (m,3)
            # Original Vertex Sums (for normalization)
            v_sums_orig = np.zeros(self.n_vertices)  # (n,)
            np.add.at(v_sums_orig, faces.flatten(), raw_angles.flatten())
            # Normalized angles for original mesh
            is_boundary = self.is_boundary_vertex[faces]  # (m,3)
            target_sums = np.where(is_boundary, np.pi, 2 * np.pi)  # (m,3)
            normalized_angles = raw_angles * (target_sums / v_sums_orig[faces])

        N = len(face_indices)
        v_ids = faces[face_indices]  # (N, 3)

        pivots = np.argmax(bary_coords, axis=1)  # (N,)

        # ref_vids = v_ids[:, 0]  # Use v0 as reference
        # ref_vids = v_ids[np.arange(N), pivots]  # Use max bary as reference

        v_ids = faces[face_indices]  # (N, 3)
        source_vids = v_ids[np.arange(N), pivots]

        idx_p = pivots
        idx_n = (pivots + 1) % 3  # Next vertex
        idx_prev = (pivots + 2) % 3  # Previous vertex

        # face_hedges = self.mesh.face_hedges[face_indices]  # (N, 3)

        # he_01 = face_hedges[:, 2]  # v0 -> v1
        # he_01 = face_hedges[np.arange(N), (bary_max + 2) % 3]  # v0 -> v1
        he_axis = mesh.face_hedges[face_indices, idx_prev]
        he_axis_prev = mesh.get_prev(he_axis)  # v2 -> v0

        l_axis = hedge_lengths[he_axis]
        l_axis_prev = hedge_lengths[he_axis_prev]  # v2 -> v0

        # Angle at v0 in the intrinsic face (normalized)
        theta_p = normalized_angles[face_indices, idx_p]

        # Barycentric coords relative to pivot
        # P = b_p * V_p + b_n * V_n + b_prev * V_prev
        # Local coords: V_p=(0,0), V_n=(l_axis, 0), V_prev=(l_axis_prev*cos, l_axis_prev*sin)
        b_n = bary_coords[np.arange(N), idx_n]
        b_prev = bary_coords[np.arange(N), idx_prev]

        # Conversion to Local Cartesian
        p_x = b_n * l_axis + b_prev * l_axis_prev * np.cos(theta_p)
        p_y = b_prev * l_axis_prev * np.sin(theta_p)

        # Magnitude and angle within the face (relative to edge v0->v1)
        magnitudes = np.sqrt(p_x**2 + p_y**2)
        angle_in_face = np.arctan2(p_y, p_x)
        angle_in_face[angle_in_face < 0] += 2 * np.pi

        # Get the phi coordinate of the starting edge (v0->v1)
        phi_start = hedge_phi[he_axis]

        # The actual phi angle of our vector is:
        phi_angles = phi_start + angle_in_face

        # Create complex vector field in phi coordinates
        vfield = magnitudes * np.exp(1j * phi_angles)

        return source_vids, vfield

    def intrinsic_bary_to_original(self, face_indices, bary_coords, vertices):

        source_vids, vfield = self.bary_to_vfield(
            face_indices, bary_coords, on_intrinsic=True
        )

        # Rebuild in 3D using the original mesh
        displacement_vectors = self.rebuild_vfield(
            vfield, vertices, source_indices=source_vids
        )

        # Add displacement to reference vertex positions
        # positions = vertices[ref_vids] + displacement_vectors

        return source_vids, displacement_vectors

    def get_uedge_connection_angle_LC(self):
        """
        Return the Levi-Civita connection 1-form rho

        Parameters
        structure : DelaunayStructure
            DelaunayStructure of the mesh
        Returns
        -------
        rho_uedge : np.array
            (e,) Array of connection angles on each undirected edge (in radians between -pi and pi)
        """

        border_mask = self.mesh.boundary_hedges_mask

        hedge_can = self.mesh.canonical_hedges

        rho_can = self.hedge_phi[hedge_can]

        rho_opp = np.where(
            border_mask[hedge_can],
            np.pi,
            self.hedge_phi[self.mesh.opposites[hedge_can]],
        )

        rho_edge = -rho_can + rho_opp - np.pi

        rho_edge = wrap_to_pi(rho_edge)

        return rho_edge

    def get_hedge_connection_angle_LC(self):
        """
        Return the Levi-Civita connection 1-form rho

        Parameters
        structure : DelaunayStructure
            DelaunayStructure of the mesh
        Returns
        -------
        rho_hedge : np.array
            (N-he,) Array of connection angles on each directed edge (in radians between -pi and pi)
        """

        border_mask = self.mesh.boundary_hedges_mask

        rho_hedge = self.hedge_phi

        rho_opp = np.where(border_mask, np.pi, self.hedge_phi[self.mesh.opposites])

        rho_hedge = -rho_hedge + rho_opp - np.pi

        rho_hedge = wrap_to_pi(rho_hedge)

        return rho_hedge

    def get_curvature_LC(self):
        return self.normalized_angles.sum(1) - np.pi
