import numpy as np

from .halfedge_structure import HalfEdgeStructure
from .geom_utils import (
    compute_face_areas_from_len,
    compute_internal_angles_from_length_stable,
    compute_cotan_weights_from_length,
    compute_flipped_edge_length,
)


class IntrinsicTriangulation:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        tufted_border: bool = False,
        compute_cotan_with_len: bool = True,
        allow_self_loops: bool = True,
    ):
        # Topology
        self.mesh = HalfEdgeStructure(faces, allow_self_loops=allow_self_loops)
        self.n_vertices = vertices.shape[0]
        self.tufted_border = tufted_border

        self.compute_cotan_with_len = compute_cotan_with_len

        # Geometry (Stored per Half-Edge for O(1) access)
        # Invariant: self.hedge_lengths[h] == self.hedge_lengths[opp[h]]
        self.hedge_lengths = self._compute_initial_hedge_lengths(
            vertices, self.mesh.faces
        )
        # self.edge_lengths = self._compute_initial_edge_lengths(
        #     vertices, self.mesh.faces
        # )

        # Cached properties
        self._face_angles = None

    def _compute_initial_edge_lengths(self, vertices: np.ndarray) -> np.ndarray:
        """Computes Euclidean edge lengths for the initial mesh."""
        ind0, ind1 = self.mesh.get_vertices(self.mesh.canonical_hedges)

        v0 = vertices[ind0]
        v1 = vertices[ind1]

        edge_lengths = np.linalg.norm(v1 - v0, axis=1)

        return edge_lengths

    def _compute_initial_hedge_lengths(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """Computes Euclidean edge lengths for the initial mesh."""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Edge 0: v1-v2
        # Edge 1: v2-v0
        # Edge 2: v0-v1
        l0 = np.linalg.norm(v1 - v2, axis=1)
        l1 = np.linalg.norm(v2 - v0, axis=1)
        l2 = np.linalg.norm(v0 - v1, axis=1)

        n_faces = faces.shape[0]
        lengths = np.zeros(n_faces * 3)
        lengths[0::3] = l0
        lengths[1::3] = l1
        lengths[2::3] = l2
        return lengths

    @property
    def euler_char(self):
        return self.n_vertices + self.n_faces - self.n_edges

    @property
    def genus(self):
        return (2 - self.euler_char) // 2

    @property
    def edge_lengths(self):
        """(p,) Lengths of undirected edges."""
        he_lengths = self.hedge_lengths  # (n_he,)
        canonical_hes = self.mesh.canonical_hedges  # (n_ue,)
        return he_lengths[canonical_hes]

    @property
    def n_faces(self):
        return self.mesh.n_faces

    @property
    def faces(self):
        return self.mesh.faces

    @property
    def edges(self):
        """(p, 2) Undirected edges as vertex index pairs."""
        he_canonical = self.mesh.canonical_hedges  # (n_ue,)

        v0, v1 = self.mesh.get_vertices(he_canonical)  # (n_ue,)

        edges = np.stack((v0, v1), axis=1)  # (n_ue, 2)

        return edges

    @property
    def n_edges(self):
        return self.mesh.n_edges

    @property
    def n_hedges(self):
        return self.mesh.n_hedges

    @property
    def face_lengths(self):
        """(m, 3) Lengths of edges per face."""
        return self.hedge_lengths[self.mesh.face_hedges]

    @property
    def face_areas(self):
        """(m,) Area of each face."""
        return compute_face_areas_from_len(self.face_lengths)

    @property
    def vertex_areas(self):
        """(n,) Area associated to each vertex (1/3 of adjacent faces)."""
        face_areas = self.face_areas  # (m,)

        vals = np.repeat(face_areas, 3) / 3.0

        # Flatten faces to get the vertex indices corresponding to those values
        # self.mesh.faces is (N_faces, 3) -> flatten to (3*N_faces,)
        v_indices = self.mesh.faces.flatten()

        # 3. Accumulate
        vertex_areas = np.zeros(self.n_vertices, dtype=float)
        np.add.at(vertex_areas, v_indices, vals)

        if self.tufted_border:
            # Boundary vertices get half area
            boundary_verts = self.mesh.boundary_vertices
            vertex_areas[boundary_verts] *= 2

        return vertex_areas

    @property
    def face_angles(self):
        """(m, 3) Internal angles."""
        if self._face_angles is None:
            self._face_angles = compute_internal_angles_from_length_stable(
                self.hedge_lengths[self.mesh.face_hedges]
            )
        return self._face_angles

    @property
    def face_cotan_weights(self):
        """(m, 3) Cotan(angle) per corner."""
        if self.compute_cotan_with_len:
            face_length = self.mesh.hedge_data_to_face_data(
                self.hedge_lengths
            )  # (m, 3)
            cotan_angle = compute_cotan_weights_from_length(face_length)
        else:
            angles = self.face_angles  # (m, 3)
            cotan_angle = 1.0 / np.tan(angles)

        return 0.5 * cotan_angle

    @property
    def cotan_weights_hedge(self):
        """Returns 0.5 * cotan(angle) per half-edge."""
        # Flat array of angles matching half-edges
        cots_face = self.face_cotan_weights  # (m, 3)
        cots = self.mesh.face_data_to_hedge_data(cots_face)  # (n_he,)

        return cots

    @property
    def cotan_weight_edge(self):
        hedge_can = self.mesh.canonical_hedges  # (e,)

        cots = self.cotan_weights_hedge[hedge_can]  # (e)

        hedge_can_opp = self.mesh.opposites[hedge_can]
        is_can_he_interior = self.mesh.is_interior(hedge_can)

        if self.tufted_border:
            cots_opp = np.where(
                is_can_he_interior, self.cotan_weights_hedge[hedge_can_opp], cots
            )
        else:
            cots_opp = np.where(
                is_can_he_interior, self.cotan_weights_hedge[hedge_can_opp], 0
            )

        return cots + cots_opp

    def is_delaunay(self, he_id, tol=1e-8):
        """
        Checks if the edge `he_id` satisfies the intrinsic Delaunay condition.
        Condition: sum of opposite angles <= pi.

             v2
            /  \
          v1----v3  (Shared Edge: v1-v3)
            \  /
             v4

        We check: angle(v2) + angle(v4) <= pi
        """
        mesh = self.mesh
        opp_id = mesh.opposites[he_id]

        # Boundary edges are always Delaunay
        if opp_id == -1:
            return True

        angle_v2, angle_v4 = self.mesh.gather_from_face_data(
            self.face_angles, np.array([he_id, opp_id])
        )

        return (angle_v2 + angle_v4) <= (np.pi + tol)  # Epsilon for float stability

    def _is_valid_triangle(self, a, b, c):
        """Checks if side lengths a, b, c form a valid triangle."""
        if a <= 0 or b <= 0 or c <= 0:
            return False
        if np.isnan(a) or np.isnan(b) or np.isnan(c):
            return False
        # Triangle Inequality: sum of any two sides > third side
        return (a + b > c) and (a + c > b) and (b + c > a)

    def flip_edge(self, he_id):
        r"""
        Flips the shared edge (v1-v3) to (v2-v4) intrinsically.

        Updates:
          1. Topology (connectivity)
          2. Geometry (edge lengths)
          3. Resets caches

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

        """
        mesh = self.mesh
        he_13 = he_id
        he_31 = mesh.opposites[he_13]

        if not self.mesh.is_flippable(he_id):
            return None

        # ---------------------------------------------------------
        # 1. READ GEOMETRY (Before Topo Change)
        # ---------------------------------------------------------

        # f_A = mesh.get_face(he_13)
        # f_B = mesh.get_face(he_31)

        # Identify the half-edges for the perimeter
        # In Face 0 (v1, v3, v2):
        #   he_id (l13) is opp v2.
        #   next  (l23) is opp v1.
        #   prev  (l12) is opp v3.
        he_32 = mesh.get_next(he_13)
        he_21 = mesh.get_prev(he_13)

        # In Face 1 (v1, v4, v3):
        #   opp_id (l31) is opp v4.
        #   next   (l14) is opp v3.
        #   prev   (l43) is opp v1.
        he_14 = mesh.get_next(he_31)
        he_43 = mesh.get_prev(he_31)

        l_32 = self.hedge_lengths[he_32]
        l_21 = self.hedge_lengths[he_21]
        l_14 = self.hedge_lengths[he_14]
        l_43 = self.hedge_lengths[he_43]
        l_13 = self.hedge_lengths[he_13]

        # ---------------------------------------------------------
        # 3. COMPUTE DIAGONAL (v2-v4)
        # ---------------------------------------------------------
        # We need the angle at v1 (corner between l_21 and l_14)
        # v1 is the vertex *opposite* he_32 in Face A, and *opposite* he_43 in Face B.
        l_24 = compute_flipped_edge_length(l_21, l_13, l_14, l_32, l_43)

        valid_face_A = self._is_valid_triangle(l_24, l_21, l_14)

        # New Face B: (v4, v3, v2) -> Edges: l_32, l_24, l_43
        valid_face_B = self._is_valid_triangle(l_32, l_24, l_43)

        if not (valid_face_A and valid_face_B):
            return None  # ABORT: This flip would create broken geometry

        # ---------------------------------------------------------
        # 4. EXECUTE FLIP (Topology)
        # ---------------------------------------------------------
        new_indices = mesh.flip_edge(he_13)
        if new_indices is None:
            return None

        he_42, he_24 = new_indices

        f_A = mesh.get_face(he_42)
        f_B = mesh.get_face(he_24)

        # ---------------------------------------------------------
        # 5. UPDATE GEOMETRY (Map lengths to NEW structure)
        # ---------------------------------------------------------
        # New Face A: (v1, v4, v2)
        # Slot 0 (opp v1): v4->v2 (New Shared) -> l_24
        # Slot 1 (opp v4): v2->v1 (Old he_21)  -> l_21
        # Slot 2 (opp v2): v1->v4 (Old he_14)  -> l_14

        self.hedge_lengths[3 * f_A + 0] = l_24
        self.hedge_lengths[3 * f_A + 1] = l_21
        self.hedge_lengths[3 * f_A + 2] = l_14

        # New Face B: (v4, v3, v2)
        # Slot 0 (opp v4): v3->v2 (Old he_32)  -> l_32
        # Slot 1 (opp v3): v2->v4 (New Shared) -> l_24
        # Slot 2 (opp v2): v4->v3 (Old he_43)  -> l_43

        self.hedge_lengths[3 * f_B + 0] = l_32
        self.hedge_lengths[3 * f_B + 1] = l_24
        self.hedge_lengths[3 * f_B + 2] = l_43

        angles_A, angles_B = compute_internal_angles_from_length_stable(
            np.array([[l_24, l_21, l_14], [l_32, l_24, l_43]])
        )

        # self._face_angles = None
        self._face_angles[f_A] = angles_A
        self._face_angles[f_B] = angles_B

        return (he_42, he_24)

    def make_delaunay(self, verbose=False, tol=1e-5):
        """
        Iteratively flips non-Delaunay edges until the mesh is Intrinsic Delaunay.

        Algorithm:
        ----------
        Uses the Lawson Flip algorithm with a stack (deque).
        1. Initialize stack with all internal edges that might be non-Delaunay.
        2. Pop an edge. If it violates the Delaunay condition, flip it.
        3. If flipped, the 4 edges of the surrounding quadrilateral might
           now violate the condition, so push them onto the stack.
        """
        from collections import deque

        n_he = self.mesh.n_hedges
        opposites = self.mesh.opposites

        mask_candidate = self.mesh.interior_hedges_mask & (np.arange(n_he) < opposites)
        candidates = np.flatnonzero(mask_candidate)
        cand_opps = opposites[candidates]

        ang_a = self.mesh.gather_from_face_data(self.face_angles, candidates)
        ang_b = self.mesh.gather_from_face_data(self.face_angles, cand_opps)

        mask_flip = (ang_a + ang_b) > (np.pi + tol)

        initial_bad_edges = candidates[mask_flip]
        stack = deque(initial_bad_edges)
        in_stack = np.zeros(n_he, dtype=bool)
        in_stack[initial_bad_edges] = True

        n_flips = 0

        while stack:
            h = stack.pop()
            h = self.mesh.canonical(h)
            in_stack[h] = False

            if not self.mesh.is_flippable(h, verbose=True):
                continue

            if not self.is_delaunay(h):
                result = self.flip_edge(h)
                if result is not None:

                    he_42, he_24 = result

                    # The 4 neighbors of the new diagonal that need checking
                    neighbor_indices = [
                        self.mesh.get_next(he_42),  # Edge after he_42 in new face A
                        self.mesh.get_prev(he_42),  # Edge before he_42 in new face A
                        self.mesh.get_next(he_24),  # Edge after he_24 in new face B
                        self.mesh.get_prev(he_24),  # Edge before he_24 in new face B
                    ]

                    for n_h in neighbor_indices:
                        canonical = self.mesh.canonical(n_h)
                        if not in_stack[canonical]:
                            stack.append(canonical)
                            in_stack[canonical] = True
                    n_flips += 1

                    # print(self.is_delaunay(h), self.is_delaunay(new_shared_h))
        if verbose:
            print(f"Made Delaunay with {n_flips} flips.")
