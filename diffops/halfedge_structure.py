import numpy as np


class HalfEdgeStructure:
    def __init__(self, faces, allow_self_loops=True):
        self.faces = np.array(faces).copy().astype(int)

        self.n_vertices = np.max(self.faces) + 1

        self.has_flipped = False
        self.opposites = np.full(self.n_hedges, -1, dtype=int)
        self._build_opposites()

        self._boundary_vertices = None

        self.v_hedge = self._get_vertex_start_hedges()

        self.allow_self_loops = allow_self_loops

        self._canonical_hedges = None
        self._hedge_to_edge = None

    @property
    def n_faces(self):
        """
        Number of faces.

        Returns
        -------
        n_faces : int
            Number of faces
        """
        return self.faces.shape[0]

    @property
    def n_hedges(self):
        """
        Number of half edges. Each half edge is part of a triangle
        A boundary edge only has one half edge.

        Returns
        -------
        n_hedges : int
            Number of half edges
        """
        return 3 * self.n_faces

    @property
    def face_hedges(self):
        """
        Get the halfedge indices for each face.

        Returns
        -------
        face_hedges : np.ndarray
            (n_faces, 3) array where face_hedges[f, k] is the halfedge
            index for the edge opposite to vertex k in face f.
        """
        # return np.arange(self.n_hedges).reshape(self.n_faces, 3)
        return self.hedge_data_to_face_data(np.arange(self.n_hedges))

    def hedge_data_to_face_data(self, data):
        """
        Converts half-edge data to face data by gathering.

        Parameters
        ----------
        data : np.ndarray
            An (N_hedges,) array of data per half-edge.

        Returns
        -------
        face_data : np.ndarray
            An (N_faces, 3) array where face_data[f, k] is the data
            for the half-edge opposite to vertex k in face f.
        """
        assert data.shape[0] == self.n_hedges
        return data.reshape(self.n_faces, 3)

    def face_data_to_hedge_data(self, data):
        """
        Converts face data to half-edge data by scattering.

        Parameters
        ----------
        data : np.ndarray
            An (N_faces, 3) array where data[f, k] is the data
            for the half-edge opposite to vertex k in face f.

        Returns
        -------
        hedge_data : np.ndarray
            An (N_hedges,) array of data per half-edge.
        """
        assert data.shape == (self.n_faces, 3)
        return data.reshape(self.n_hedges)

    def get_face_hedges(self, face_ids):
        """
        Get the halfedge indices for given face IDs.

        Parameters
        ----------
        face_ids : int or np.ndarray
            The face index or indices.

        Returns
        -------
        face_hedges : np.ndarray
            If face_ids is an integer, returns a (3,) array of halfedge indices.
            If face_ids is an array, returns a (len(face_ids), 3) array of halfedge indices.
        """
        face_ids = np.asarray(face_ids)
        return np.stack((3 * face_ids + 0, 3 * face_ids + 1, 3 * face_ids + 2), axis=-1)

    def is_boundary(self, he_ids=None):
        """Checks if a half-edge is on the boundary (no opposite)."""
        if he_ids is None:
            return self.opposites == -1
        return self.opposites[he_ids] == -1

    def is_interior(self, he_ids=None):
        """Checks if a half-edge is interior (has opposite)."""
        return ~self.is_boundary(he_ids)

    @property
    def boundary_vertices(self):
        """Returns the indices of all boundary vertices."""
        if self._boundary_vertices is None:
            bhedges = self.boundary_hedges  # 1D array of halfedge ids
            src, tgt = self.get_vertices(bhedges)
            self._boundary_vertices = np.unique(np.concatenate([src, tgt]))
        return self._boundary_vertices

    @property
    def boundary_hedges(self):
        """Returns the indices of all boundary half-edges."""
        return np.flatnonzero(self.boundary_hedges_mask)

    @property
    def boundary_hedges_mask(self):
        """Boolean mask of boundary half-edges."""
        return self.is_boundary()

    @property
    def interior_hedges_mask(self):
        """Boolean mask of interior half-edges."""
        return ~self.is_boundary()

    def get_face(self, he_id):
        """Halfedge list is just concatenation of all halfedges of each face."""
        return he_id // 3

    def get_next(self, he_id):
        """
        Implicit: In a triangle (0,1,2), next is (i+1)%3.
        Formula: 3 * (h // 3) + (h + 1) % 3
        """
        # This math works for scalars or numpy arrays
        return 3 * (he_id // 3) + (he_id + 1) % 3

    def get_prev(self, he_id):
        """Implicit: Previous is (i+2)%3."""
        return 3 * (he_id // 3) + (he_id + 2) % 3

    def get_opposite(self, he_id):
        """Returns the opposite half-edge index."""
        return self.opposites[he_id]

    def gather_from_face_data(self, data, he_ids):
        """
        Fetches data[face, corner] corresponding to the given half-edge(s).

        Parameters
        ----------
        data : np.ndarray
            An (N, 3) array (e.g., self.face_angles).
        he_ids : int or np.ndarray
            The half-edge index or indices.

        Returns
        -------
        The value(s) from data at the face/corner of he_ids.
        """
        assert data.shape[1] == 3 and data.shape[0] == self.n_faces
        return data[he_ids // 3, he_ids % 3]

    @property
    def n_edges(self):
        """Number of undirected edges.
        DO NOT USE DURING EDGE FLIPS, else will be very slow"""
        return len(self.canonical_hedges)

    def is_canonical(self, he_ids):
        """Checks if half-edge is the canonical representative of its undirected edge."""
        opp = self.opposites[he_ids]
        if np.issubdtype(type(he_ids), np.integer):
            return self.is_boundary(he_ids) or he_ids < opp

        return (he_ids < opp) | self.is_boundary(he_ids)

    @property
    def is_canonical_mask(self):
        return self.is_canonical(np.arange(self.n_hedges))

    def _reset_canonical_cache(self):
        self._canonical_hedges = None
        self._hedge_to_edge = None

    def _build_hedge_to_edge_mapping(self):
        """
        Builds an array mapping half-edge index to index in the canonical array (index of corresponding canonical half-edge).
        """
        canonical_hes = self.canonical_hedges

        hedge_to_edge = np.full(self.n_hedges, -1, dtype=int)
        hedge_to_edge[canonical_hes] = np.arange(len(canonical_hes))

        opp_can_hes = self.opposites[canonical_hes]

        is_can_border = self.is_boundary(canonical_hes)

        hedge_to_edge[opp_can_hes[~is_can_border]] = hedge_to_edge[
            canonical_hes[~is_can_border]
        ]
        return hedge_to_edge

    @property
    def canonical_hedges(self):
        """Returns all canonical half-edge indices."""
        if self._canonical_hedges is None:
            self._canonical_hedges = np.flatnonzero(self.is_canonical_mask)
        return self._canonical_hedges

    @property
    def hedge_to_edge(self):
        """Returns an array mapping half-edge index to index in the canonical array (index of corresponding canonical half-edge)."""
        if self._hedge_to_edge is None:
            self._hedge_to_edge = self._build_hedge_to_edge_mapping()
        return self._hedge_to_edge

    def canonical(self, he_ids):
        """Returns the unique undirected ID (min of h, opp)."""
        opp = self.opposites[he_ids]
        if np.issubdtype(type(he_ids), np.integer):
            if self.is_boundary(he_ids) or he_ids < opp:
                return he_ids
            return opp

        return np.where((he_ids < opp) | self.is_boundary(he_ids), he_ids, opp)

    def get_vertices(self, he_ids):
        """
        Get the (source, target) vertex indices for half-edges.

        Returns
        -------
        source, target : tuple of np.ndarray or int
        """
        f = he_ids // 3
        k = he_ids % 3

        src = self.faces[f, (k + 1) % 3]
        tgt = self.faces[f, (k + 2) % 3]
        return src, tgt

    def get_source_vid(self, he_ids):
        """Returns the vertex index at the *start* of half-edge h."""
        f_id = he_ids // 3
        k = he_ids % 3
        return self.faces[f_id, (k + 1) % 3]

    def get_dest_vid(self, he_ids):
        """Returns the vertex index at the *end* of half-edge h."""
        # The destination of h is the start of next(h)
        return self.get_source_vid(self.get_next(he_ids))

    def get_local_src_tgt(self, he_ids):
        """
        Return the local (0, 1, 2) face-vertex slots of the source and target
        of each half-edge. Mirrors get_vertices but returns local indices
        instead of global vertex IDs — useful for indexing barycentric arrays.
        """
        k = he_ids % 3
        return (k + 1) % 3, (k + 2) % 3

    def _get_vertex_start_hedges(self):
        """
        Returns an array (N_vert,) of half-edge indices.
        If the vertex is on the boundary, this GUARANTEES the most 'clockwise'
        edge (starting the fan) to ensure full circulation.
        """
        # 1. Initialize with -1 (Safety for isolated vertices)
        v_starts = np.full(self.n_vertices, -1, dtype=int)

        # 2. Default: take any outgoing edge for every vertex
        all_hes = np.arange(self.n_hedges)
        sources = self.get_source_vid(all_hes)

        # We overwrite; the last seen edge for a vertex becomes the default start.
        # This is faster and safer than np.unique for dense arrays.
        v_starts[sources] = all_hes

        # 3. Fix Boundary Vertices
        # The boundary half-edge (void on right) is the Clockwise limit.
        # It is the only valid start for a full CCW rotation.
        boundary_hes = self.boundary_hedges

        if len(boundary_hes) > 0:
            fan_starts = self.get_source_vid(boundary_hes)
            v_starts[fan_starts] = boundary_hes

        return v_starts

    # Initial Topology
    def _build_opposites(self):
        """
        Builds opposites.
        Note: Edge `3*f + i` connects faces[f, i+1] -> faces[f, i+2].
        """
        n_he = self.n_hedges

        # 1. Edges array (N_he, 2)
        # For every face (v0, v1, v2):
        # Edge 0 (Opp v0): v1 -> v2
        # Edge 1 (Opp v1): v2 -> v0
        # Edge 2 (Opp v2): v0 -> v1

        v0 = self.faces[:, 0]  # (m,)
        v1 = self.faces[:, 1]  # (m,)
        v2 = self.faces[:, 2]  # (m,)

        edges = np.zeros((n_he, 2), dtype=int)
        edges[0::3] = np.stack((v1, v2), axis=1)  # 0 is opp v0
        edges[1::3] = np.stack((v2, v0), axis=1)  # 1 is opp v1
        edges[2::3] = np.stack((v0, v1), axis=1)  # 2 is opp v2

        # 2. Sort to find pairs (Standard logic)
        min_v = np.minimum(edges[:, 0], edges[:, 1])  # (N_he,)
        max_v = np.maximum(edges[:, 0], edges[:, 1])  # (N_he,)

        # this sorts by min_v, then by max_v.
        order = np.lexsort((max_v, min_v))  # (N_he,)

        sorted_min = min_v[order]  # (N_he,)
        sorted_max = max_v[order]  # (N_he,)

        diff_min = np.diff(sorted_min)  # (N_he - 1,)
        diff_max = np.diff(sorted_max)  # (N_he - 1,)

        is_pair = (diff_min == 0) & (diff_max == 0)  # (N_he - 1,)
        # pair_start = np.where(is_pair)[0]
        pair_start = np.flatnonzero(is_pair)  # (n_pairs,)

        he_A = order[pair_start]
        he_B = order[pair_start + 1]

        self.opposites[he_A] = he_B
        self.opposites[he_B] = he_A

    # Edge Flipping
    def flip_edge(self, he_id):
        r"""
        Rotates the shared edge between two triangles (Flip operation).

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
            MUST be the edge opposite to v2 in f0 (representing v1->v3).

        Returns
        -------
        bool
            True if the flip was successful, False if the edge is a boundary edge.
        """

        if not self.is_flippable(he_id):
            return None
        # 1. Get the shared edge (h_a) and its opposite (h_b)
        he_13 = he_id
        he_31 = self.opposites[he_13]

        # Cannot flip a boundary edge
        if he_31 == -1:
            return None

        # 2. Identify Face Indices
        f0 = self.get_face(he_13)
        f1 = self.get_face(he_31)

        # 3. Identify Neighboring Half-Edges (The Outer Boundary)
        he_32 = self.get_next(he_13)
        he_21 = self.get_prev(he_13)

        he_14 = self.get_next(he_31)
        he_43 = self.get_prev(he_31)

        # 4. Identify Vertices
        # In f0: Edge is v1->v3, so Vertices are (v1, v3, v2) relative to the edge.
        # k_v2 points to v2.
        v1, v3 = self.get_vertices(he_13)
        v2 = self.get_source_vid(he_21)
        v4 = self.get_source_vid(he_43)

        # 5. Save External Neighbors (Pointers to outside world)
        he_23 = self.opposites[he_32]
        he_12 = self.opposites[he_21]
        he_41 = self.opposites[he_14]
        he_34 = self.opposites[he_43]

        # 6. UPDATE GEOMETRY (Rewrite Faces)
        # We construct the new faces using the vertices we extracted
        self.faces[f0] = [v1, v4, v2]  # New Face A
        self.faces[f1] = [v4, v3, v2]  # New Face B

        # 7. UPDATE TOPOLOGY (Rewire Opposites)
        # Implicit Slots: 0=Opp(v_0), 1=Opp(v_1), 2=Opp(v_2)

        # --- New f0 (v1, v4, v2) ---
        he_42_new = 3 * f0 + 0
        he_21_new = 3 * f0 + 1  # Same geometry as he_21_old
        he_14_new = 3 * f0 + 2  # Same geometry as he_14_old

        # --- New f1 (v4, v3, v2) ---
        he_32_new = 3 * f1 + 0  # Same geometry as he_32_old
        he_24_new = 3 * f1 + 1
        he_43_new = 3 * f1 + 2  # Same geometry as he_43_old

        # 7. MAP OLD INTERNAL EDGES TO NEW LOCATIONS
        # If an external neighbor was actually one of our internal edges,
        # we must redirect the connection to where that edge MOVED.
        old_to_new = {
            he_21: he_21_new,
            he_14: he_14_new,
            he_32: he_32_new,
            he_43: he_43_new,
            he_13: he_42_new,  # Technically shared moves to shared
            he_31: he_24_new,
        }

        def get_target(h_idx):
            # If the neighbor is -1 (boundary), stay -1
            if h_idx == -1:
                return -1
            # If the neighbor is part of the flip, use its new index
            # Otherwise, keep the old index
            return old_to_new.get(h_idx, h_idx)

        # Slot 1 (Opp v4): Edge v2->v1. (Was he_21)
        # Edge v2->v1 (he_21_new) connects to OLD neighbor he_12
        # But he_12 might have moved!
        target_12 = get_target(he_12)
        self.opposites[he_21_new] = target_12
        if target_12 != -1:
            self.opposites[target_12] = he_21_new

        # Slot 2 (Opp v2): Edge v1->v4. (Was he_14)
        target_41 = get_target(he_41)
        self.opposites[he_14_new] = target_41
        if target_41 != -1:
            self.opposites[target_41] = he_14_new

        # Slot 0 (Opp v4): Edge v3->v2. (Was he_32)
        target_23 = get_target(he_23)
        self.opposites[he_32_new] = target_23
        if target_23 != -1:
            self.opposites[target_23] = he_32_new

        # Slot 2 (Opp v2): Edge v4->v3. (Was he_43)
        target_34 = get_target(he_34)
        self.opposites[he_43_new] = target_34
        if target_34 != -1:
            self.opposites[target_34] = he_43_new

        # 7. Link the New Shared Edge
        self.opposites[he_42_new] = he_24_new
        self.opposites[he_24_new] = he_42_new

        # 8 - Starting Vertices
        if self.v_hedge[v1] == he_13:
            self.v_hedge[v1] = he_14_new
        if self.v_hedge[v3] == he_31:
            self.v_hedge[v3] = he_32_new
        # v2 and v4 pointers might also need updates if they pointed to edges that moved
        if self.v_hedge[v2] == he_21:
            self.v_hedge[v2] = he_21_new
        if self.v_hedge[v4] == he_43:
            self.v_hedge[v4] = he_43_new
        self.has_flipped = True
        # self._uedges = None
        # self._uedge_to_hedge = None
        # self._hedge_to_uedge = None

        self._reset_canonical_cache()

        return (he_42_new, he_24_new)

    def is_degree_1(self, v_id):
        start_he = self.v_hedge[v_id]
        if start_he == -1:
            return False  # Isolated vertex

        # Move to the next outgoing half-edge around the vertex
        # Formula for "swing" around vertex: next(opposite(he))

        # One step
        opp = self.opposites[start_he]
        if opp == -1:
            return False  # Boundary logic (omitted for brevity)

        h_next = self.get_next(opp)

        # If we are back at start immediately, it's degree 1
        return h_next == start_he

    def is_flippable(self, he_id, verbose=False):
        """
        Check if edge can be flipped while preserving manifoldness.

        Assumes the mesh is currently valid (no degenerate faces).
        """
        opp = self.opposites[he_id]
        if opp == -1:
            if verbose:
                print("Cannot flip boundary edge.")
            return False

        # Get the two vertices that will form the new edge
        if not self.allow_self_loops:
            v2 = self.get_source_vid(self.get_prev(he_id))
            v4 = self.get_source_vid(self.get_prev(opp))

            if v2 == v4:
                if verbose:
                    print("Cannot flip edge that would create self-loop.")
                return False

        # Check if either vertex is degree 1
        v1 = self.get_source_vid(he_id)
        v3 = self.get_dest_vid(he_id)

        if self.is_degree_1(v1) or self.is_degree_1(v3):
            if verbose:
                print("Cannot flip edge adjacent to degree-1 vertex.")
            return False

        # Cannot flip if it would create a degenerate edge
        return True

    def copy(self):
        new_mesh = HalfEdgeStructure(
            self.faces.copy(), allow_self_loops=self.allow_self_loops
        )
        new_mesh.opposites = self.opposites.copy()
        if self._boundary_vertices is not None:
            new_mesh._boundary_vertices = self._boundary_vertices.copy()
        new_mesh.v_hedge = self.v_hedge.copy()
        new_mesh.has_flipped = self.has_flipped

        return new_mesh
