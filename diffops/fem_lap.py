import numpy as np
from scipy import sparse


def get_curvature_factor(omega, cutoff=1e-1):
    """
    Computes C(Omega) for the Mass Matrix.
    Vectorized for numpy arrays.

    Parameters
    ----------
    omega : array_like
        (m,) The input Omega values.
    """
    omega = np.asarray(omega)
    omega2 = omega * omega
    omega4 = omega2 * omega2

    # Taylor Series for small Omega
    # re_small = 1 / 12.0 - omega2 / 360.0 + omega4 / 20160.0
    re_small = 1 / 12.0 + omega2 * (-1 / 360.0 + omega2 / 20160.0)
    # im_small = omega * (1 / 60.0 - omega2 / 2520.0 + omega4 / 181440.0)
    im_small = omega * (1 / 60.0 + omega2 * (-1 / 2520.0 + omega2 / 181440.0))
    res_small = re_small + 1j * im_small

    # Analytical Formula for large Omega
    omega3 = omega2 * omega
    # 6*cos(w) - 6 + 3w^2
    num_re = 6 * np.cos(omega) - 6 + 3 * omega2
    # 6*sin(w) - 6w + w^3
    num_im = 6 * np.sin(omega) - 6 * omega + omega3
    res_large = (num_re + 1j * num_im) / (3 * omega4)

    return np.where(np.abs(omega) < cutoff, res_small, res_large)


def get_f1(omega, cutoff=1e-1):
    """Computes f1(Omega) for the Laplacian."""
    omega = np.asarray(omega)
    omega2 = omega * omega
    omega4 = omega2 * omega2

    # Taylor
    # re_small = omega2 * (1 / 120.0 - omega2 / 2688.0 + omega4 / 129600.0)
    # im_small = omega * (-1 / 24.0 + omega2 / 504.0 - omega4 / 17280.0)
    re_small = omega2 * (1 / 120.0 + omega2 * (-1 / 2688.0 + omega2 / 129600.0))
    im_small = omega * (-1 / 24.0 + omega2 * (1 / 504.0 - omega2 / 17280.0))
    res_small = re_small + 1j * im_small

    # Analytical
    # Term 1: 3 + w^4/24 + i(w - w^5/60)
    term1 = (3 + omega4 / 24.0) + 1j * (omega - (omega4 * omega) / 60.0)
    # Term 2: (-3 + w^2/2 + 2iw) * exp(iw)
    term2 = (-3 + omega2 / 2.0 + 1j * 2 * omega) * np.exp(1j * omega)

    res_large = (term1 + term2) / omega4

    return np.where(np.abs(omega) < cutoff, res_small, res_large)


def get_f2(omega, cutoff=1e-1):
    """Computes f2(Omega) for the Laplacian."""
    omega = np.asarray(omega)
    omega2 = omega * omega
    omega4 = omega2 * omega2

    # Taylor
    # re_small = -0.25 + omega2 * (1 / 45.0  -omega2 / 1120.0 + omega4 / 56700.0)
    # im_small = omega * (-1 / 24.0 + omega2 * (5 / 1008.0) - omega4 * (7 / 51840.0))
    re_small = -0.25 + omega2 * (1 / 45.0 + omega2 * (-1 / 1120.0 + omega2 / 56700.0))
    im_small = omega * (-1 / 24.0 + omega2 * (5 / 1008.0 + omega2 * (-7 / 51840.0)))
    res_small = re_small + 1j * im_small

    # Analytical
    # Term 1: 4 - w^4/12 + i(w - w^3/6 + w^5/30)
    term1 = (4 - omega4 / 12.0) + 1j * (
        omega - (omega * omega2) / 6.0 + (omega4 * omega) / 30.0
    )
    # Term 2: (-4 + w^2 + 3iw) * exp(iw)
    term2 = (-4 + omega2 + 1j * 3 * omega) * np.exp(1j * omega)

    res_large = (term1 + term2) / omega4

    return np.where(np.abs(omega) < cutoff, res_small, res_large)


def compute_face_areas_heron(edge_lengths, face_ue):
    """
    Compute per-face areas of a triangular mesh from lengths of edges.

    Parameters
    -----------------------------

    edge_lengths : (E,) float array
        Lengths of the undirected edges.
    face_ue : (F, 3) int array
        Indices of the 3 undirected edges for each face.
        Typically: [edge_opp_v0, edge_opp_v1, edge_opp_v2]

    Returns
    -----------------------------
    faces_areas : np.ndarray or float
        (m,) or float array of per-face areas
    """
    lengths = edge_lengths[face_ue]

    assert lengths.ndim == 2 and lengths.shape[1] == 3, "lengths must be (m,3) array"

    length_sorted = np.sort(lengths, axis=1)  # (m,3) sort ascending

    # a >= b >= c
    a = length_sorted[:, 2]
    b = length_sorted[:, 1]
    c = length_sorted[:, 0]

    assert np.all(c - (a - b) >= 0), "Triangle inequality violated"

    areas = 0.25 * np.sqrt(
        (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    )

    return areas


def bundle_fem_mass_matrix(
    faces,
    edge_lengths,
    omega,
    connection,
    face_opposite_ue,
    n_vertices,
    cutoff=1e-1,
):
    """
    Compute the FEM mass matrix for mesh laplacian using finite elements method
    with curvature correction.

    Entry (i,i) is 1/6 of the sum of the area of surrounding triangles
    Entry (i,j) is 1/12 of the sum of the area of triangles using edge (i,j)
                 (weighted by curvature and transport)

    Parameters
    -----------------------------
    faces      :
        (m,3) array of vertex indices defining faces
    faces_area :
        (m,) array of per-face area
    omega      :
        (m,) array of per-face curvature values (Integrated Gaussian Curvature)
    connection :
        (e,) array of per-undirected edge connection ANGLES (rho).
        CRITICAL: This must be angles (float), not complex numbers.
        Undirected edges are assumed to be (u,v) with u < v.
    face_opposite_ue :
        (m,3) Indices of undirected edges opposite to vertices in `faces`.
        - [t, 0] is edge opposite v0 (i.e., edge v1 -> v2)
        - [t, 1] is edge opposite v1 (i.e., edge v2 -> v0)
        - [t, 2] is edge opposite v2 (i.e., edge v0 -> v1)
    n_vertices :
        int, total number of vertices (to shape the matrix)

    Returns
    -----------------------------
    M : scipy.sparse.csc_matrix
        (n,n) sparse Hermitian mass matrix in csc format
    """

    # 1. Orient the connection angles

    # Check orientation of the 3 edges of the face:
    # Col 0: Edge v1->v2 (Opposite v0)
    # Col 1: Edge v2->v0 (Opposite v1)
    # Col 2: Edge v0->v1 (Opposite v2)
    is_oriented = np.stack(
        [
            faces[:, 2] > faces[:, 1],  # v1 < v2 ?
            faces[:, 0] > faces[:, 2],  # v2 < v0 ?
            faces[:, 1] > faces[:, 0],  # v0 < v1 ?
        ],
        axis=1,
    )  # (m,3)

    # Fetch undirected values
    rho_opp = connection[face_opposite_ue]  # (m,3)

    # Flip sign if half-edge is against the canonical undirected order
    rho_opp = np.where(is_oriented, rho_opp, -rho_opp)

    # face_opposite_con = np.where(
    #     is_oriented, face_opposite_con, -face_opposite_con
    # )  # (m,3)

    N = n_vertices

    # 2. Compute curvature correction factors C(Omega)
    c_factors = get_curvature_factor(omega, cutoff=cutoff)  # (m,)

    Vi = faces[:, 0]  # (m,)
    Vj = faces[:, 1]  # (m,)
    Vk = faces[:, 2]  # (m,)

    I_diag = np.concatenate([Vi, Vj, Vk])
    J_diag = I_diag.copy()

    face_areas = compute_face_areas_heron(edge_lengths, face_opposite_ue)
    V_diag = np.tile(face_areas / 6, 3)

    off_weights = (face_areas * c_factors)[:, None] * np.exp(-1j * rho_opp)  # (m,3)

    # The columns of off_weights correspond to edges:
    # 0: v1->v2, 1: v2->v0, 2: v0->v1

    # Define source (j) and target (k) indices for these blocks
    # Col 0: j=v1, k=v2
    # Col 1: j=v2, k=v0
    # Col 2: j=v0, k=v1
    I_off_block = faces[:, [1, 2, 0]]
    J_off_block = faces[:, [2, 0, 1]]

    I = np.concatenate([I_diag, I_off_block.ravel(), J_off_block.ravel()])
    J = np.concatenate([J_diag, J_off_block.ravel(), I_off_block.ravel()])
    V = np.concatenate([V_diag, off_weights.ravel(), np.conj(off_weights).ravel()])

    M = sparse.csr_matrix((V, (I, J)), shape=(N, N))

    return M


def bundle_fem_lap_matrix(
    faces,
    edge_length,
    omega,
    connection,
    face_opposite_ue,
    n_vertices,
    cutoff=1e-1,
):
    """
    Compute the FEM lap matrix for mesh laplacian using finite elements method
    with curvature correction.

    Parameters
    -----------------------------
    faces      :
        (m,3) array of vertex indices defining faces
    edge_length :
        (e,) array of per-undirected edge lengths (not squared).
    faces_area :
        (m,) array of per-face area
    omega      :
        (m,) array of per-face curvature values (Integrated Gaussian Curvature)
    connection :
        (e,) array of per-undirected edge connection angles (rho)
    face_opposite_ue :
        (m,3) Indices of undirected edges opposite to vertices in `faces`.
    n_vertices :
        int, total vertices

    Returns
    -----------------------------
    L : scipy.sparse.csc_matrix
        (n,n) sparse matrix in csc format
    """
    N = n_vertices

    # 1. Orient the connection angles

    # Check orientation of the 3 edges of the face:
    # Col 0: Edge v1->v2 (Opposite v0)
    # Col 1: Edge v2->v0 (Opposite v1)
    # Col 2: Edge v0->v1 (Opposite v2)
    is_oriented = np.stack(
        [
            faces[:, 2] > faces[:, 1],  # v1 < v2 ?
            faces[:, 0] > faces[:, 2],  # v2 < v0 ?
            faces[:, 1] > faces[:, 0],  # v0 < v1 ?
        ],
        axis=1,
    )  # (m,3)

    # Fetch undirected values
    rho_opp = connection[face_opposite_ue]  # (m,3)

    # Flip sign if half-edge is against the canonical undirected order
    rho_opp = np.where(is_oriented, rho_opp, -rho_opp)

    sqlen_opp = (edge_length**2)[face_opposite_ue]  # (m,3)

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
    V_diag = sqlen_opp + omega[:, None] ** 2 / 90 * (
        sqlen_adj1 + sqlen_adj2 + inprod_edges
    )

    face_areas = compute_face_areas_heron(edge_length, face_opposite_ue)
    V_diag = V_diag / (4 * face_areas[:, None])  # (m,3)

    I_diag = faces.ravel()
    J_diag = faces.ravel()
    V_diag = V_diag.ravel()

    # V_diag = np.tile(faces_area / 6, 3)

    # Define source (j) and target (k) indices for these blocks
    # Col 0 of off_weights corresponds to edge opposite v0 => v1->v2
    I_off_block = faces[:, [1, 2, 0]]
    J_off_block = faces[:, [2, 0, 1]]

    f1_vals = get_f1(omega, cutoff=cutoff)  # (m,)
    f2_vals = get_f2(omega, cutoff=cutoff)  # (m,)

    # Note: f2 converges to -0.25, effectively providing the -1/4 factor
    # for the standard cotan laplacian when omega=0.
    off_weights = (sqlen_adj1 + sqlen_adj2) * f1_vals[:, None] + inprod_edges * f2_vals[
        :, None
    ]
    off_weights = off_weights / face_areas[:, None] * np.exp(-1j * rho_opp)  # (m,3)

    I = np.concatenate([I_diag, I_off_block.ravel(), J_off_block.ravel()])
    J = np.concatenate([J_diag, J_off_block.ravel(), I_off_block.ravel()])
    V = np.concatenate([V_diag, off_weights.ravel(), np.conj(off_weights).ravel()])

    L = sparse.csr_matrix((V, (I, J)), shape=(N, N))

    return L


def fem_area_mat(vertices, faces, faces_areas=None):
    """
    Compute the area matrix for mesh laplacian using finite elements method.

    Entry (i,i) is 1/6 of the sum of the area of surrounding triangles
    Entry (i,j) is 1/12 of the sum of the area of triangles using edge (i,j)

    Parameters
    -----------------------------
    vertices   :
        (n,3) array of vertices coordinates
    faces      :
        (m,3) array of vertex indices defining faces
    faces_area :
        (m,) - Optional, array of per-face area

    Returns
    -----------------------------
    A : scipy.sparse.csc_matrix
        (n,n) sparse area matrix in csc format
    """
    N = vertices.shape[0]

    # Compute face area
    if faces_areas is None:
        v1 = vertices[faces[:, 0]]  # (m,3)
        v2 = vertices[faces[:, 1]]  # (m,3)
        v3 = vertices[faces[:, 2]]  # (m,3)
        faces_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)  # (m,)

    # Use similar construction as cotangent weights
    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])  # (3m,)
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])  # (3m,)
    S = np.concatenate([faces_areas, faces_areas, faces_areas])  # (3m,)

    In = np.concatenate([I, J, I])  # (9m,)
    Jn = np.concatenate([J, I, I])  # (9m,)
    Sn = 1 / 12 * np.concatenate([S, S, 2 * S])  # (9m,)

    A = sparse.coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()
    return A
