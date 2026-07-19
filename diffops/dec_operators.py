import numpy as np
from scipy import sparse

import einops


def get_dec_laplacian(d0, star1):
    """
    Compute the DEC Laplacian operator on 0-forms for a mesh

    The DEC Laplacian is given by: Δ = d0^T * star1 * d0

    Parameters
    --------------------------
    d0 : scipy.sparse.csr_matrix
        (e,n) sparse matrix of D0 operator with n the number of vertices
    star1 : scipy.sparse.dia_matrix
        (e,e) sparse matrix of star1 operator with e the number of edges

    Output
    --------------------------
    laplacian : scipy.sparse.csr_matrix
        (n,n) sparse matrix of the DEC Laplacian operator on 0-forms
    """
    laplacian = d0.conj().T @ (star1 @ d0)

    return laplacian


def get_connection_product(eta1, eta2, n_verts1, n_verts2):
    """
    Compute the connection 1-form on the product manifold.

    The connection 1-form just applies horizontally or vertically depending on the direction of the edge.

    Parameters
    ----------
    eta1 : np.array
        (n_e1,) Array of connection angles on each undirected edge of the first manifold
    eta2 : np.array
        (n_e2,) Array of connection angles on each undirected edge of the second manifold
    n_verts1 : int
        Number of vertices in the first manifold
    n_verts2 : int
        Number of vertices in the second manifold

    Returns
    -------
    eta_prod : np.array
        (n_e1*n_v2 + n_v1*n_e2,) Array of connection angles on each undirected edge of the product manifold

    """
    eta1_ext = np.kron(eta1, np.ones(n_verts2))  # (n_e1*n_v2,)
    eta2_ext = np.kron(np.ones(n_verts1), eta2)  # (n_v1*n_e2,)

    eta_prod = np.concatenate([eta1_ext, eta2_ext])  # (n_e1*n_v2 + n_v1*n_e2,)
    return eta_prod


def get_curvature_product(omega1, omega2, n_verts1, n_verts2, n_edges1, n_edges2):
    """
    Compute the curvature 2-form on the product manifold.

    The curvature 2-form just adds up the curvatures of each manifold.

    Parameters
    ----------
    omega1 : np.array
        (m1,) Array of curvature on each face of the first manifold
    omega2 : np.array
        (m2,) Array of curvature on each face of the second manifold
    n_faces1 : int
        Number of faces in the first manifold
    n_faces2 : int
        Number of faces in the second manifold

    Returns
    -------
    omega_prod : np.array
        (f_A * n_B + e_A*e_B + f_B * n_A) Array of curvature on each face of the product manifold
    """
    omega1_ext = np.kron(omega1, np.ones(n_verts2))  # (m1*n2,)
    omega2_ext = np.kron(np.ones(n_verts1), omega2)  # (n1*m2,)

    omega_edges = np.zeros(n_edges1 * n_edges2)  # (e1*e2,)

    omega_prod = np.concatenate([omega1_ext, omega_edges, omega2_ext], axis=0)
    return omega_prod


def get_d0_product(d0_A, d0_B):
    """
    Compute the gradient operator on the product mesh of two meshes A and B

    Parameters
    --------------------------
    d0_A : scipy.sparse.csr_matrix
        (e_A,n_A) sparse matrix of d0 operator for mesh A with n_A the number of vertices in A
    d0_B : scipy.sparse.csr_matrix
        (e_B,n_B) sparse matrix of d0 operator for mesh B with n_B the number of vertices in B

    Output
    --------------------------
    d0_product : scipy.sparse.csr_matrix
        ((e_A * n_B + e_B * n_A), (n_A * n_B)) sparse matrix of D0 operator for the product mesh
    """
    n_verts_A = d0_A.shape[1]
    n_verts_B = d0_B.shape[1]

    # Gradient in the A direction
    d0_A_kron_Iv_B = sparse.kron(
        d0_A, sparse.eye(n_verts_B, format="csr"), format="csr"
    )

    # Gradient in the B direction
    Iv_A_kron_d0_B = sparse.kron(
        sparse.eye(n_verts_A, format="csr"), d0_B, format="csr"
    )

    d0_AB = sparse.vstack([d0_A_kron_Iv_B, Iv_A_kron_d0_B], format="csr")

    return d0_AB


def get_d1_product(d0_A, d0_B, d1_A, d1_B):
    """
    Compute the d_1 operator on the product mesh of two meshes A and B

    Parameters
    --------------------------
    d0_A : scipy.sparse.csr_matrix
        (e_A,n_A) sparse matrix of d0 operator for mesh A with n_A the number of vertices in A
    d0_B : scipy.sparse.csr_matrix
        (e_B,n_B) sparse matrix of d0 operator for mesh B with n_B the number of vertices in B
    d1_A : scipy.sparse.csr_matrix
        (f_A,e_A) sparse matrix of d1 operator for mesh A with f_A the number of faces in A
    d1_B : scipy.sparse.csr_matrix
        (f_B,e_B) sparse matrix of d1 operator for mesh B with f_B the number of faces in B

    Output
    --------------------------
    d1_product : scipy.sparse.csr_matrix
        ((f_A * n_B + e_A*e_B + f_B * n_A), (e_A * n_B + e_B * n_A)) sparse matrix of D1 operator for the product mesh
    """
    n_verts_A = d0_A.shape[1]
    n_verts_B = d0_B.shape[1]

    n_edges_A = d0_A.shape[0]
    n_edges_B = d0_B.shape[0]

    n_faces_A = d1_A.shape[0]
    n_faces_B = d1_B.shape[0]

    # Gradient in the A direction
    d1_A_kron_Iv_B = sparse.kron(
        d1_A, sparse.eye(n_verts_B, format="csr"), format="csr"
    )
    d0_A_kron_Ie_B = sparse.kron(
        d0_A, sparse.eye(n_edges_B, format="csr"), format="csr"
    )

    # Gradient in the B direction
    Iv_A_kron_d1_B = sparse.kron(
        sparse.eye(n_verts_A, format="csr"), d1_B, format="csr"
    )
    Ie_A_kron_d0_B = sparse.kron(
        sparse.eye(n_edges_A, format="csr"), d0_B, format="csr"
    )

    # Combine both parts into a block matrix
    d1_AB = sparse.bmat(
        [
            [d1_A_kron_Iv_B, None],
            [-Ie_A_kron_d0_B, d0_A_kron_Ie_B],
            [None, Iv_A_kron_d1_B],
        ],
        format="csr",
    )

    return d1_AB


def get_d2_product(d0_A, d0_B, d1_A, d1_B):
    """
    Compute the d_2 operator on the product mesh of two meshes A and B

    Parameters
    --------------------------
    d0_A : scipy.sparse.csr_matrix
        (e_A,n_A) sparse matrix of d0 operator for mesh A with n_A the number of vertices in A
    d0_B : scipy.sparse.csr_matrix
        (e_B,n_B) sparse matrix of d0 operator for mesh B with n_B the number of vertices in B
    d1_A : scipy.sparse.csr_matrix
        (f_A,e_A) sparse matrix of d1 operator for mesh A with f_A the number of faces in A
    d1_B : scipy.sparse.csr_matrix
        (f_B,e_B) sparse matrix of d1 operator for mesh B with f_B the number of faces in B

    Output
    --------------------------
    d2_product : scipy.sparse.csr_matrix
        (f_A*e_B + e_A*f_B, f_A * n_B + e_A * e_B + f_B * n_A) sparse matrix of D2 operator for the product mesh
    """
    n_edges_A = d0_A.shape[0]
    n_edges_B = d0_B.shape[0]

    n_faces_A = d1_A.shape[0]
    n_faces_B = d1_B.shape[0]

    # Gradient in the A direction
    If_A_kron_d0_B = sparse.kron(
        sparse.eye(n_faces_A, format="csr"), d0_B, format="csr"
    )

    d1_A_kron_Ie_B = sparse.kron(
        d1_A, sparse.eye(n_edges_B, format="csr"), format="csr"
    )

    Ie_A_kron_d1_B = sparse.kron(
        sparse.eye(n_edges_A, format="csr"), d1_B, format="csr"
    )

    d0_A_kron_If_B = sparse.kron(
        d0_A, sparse.eye(n_faces_B, format="csr"), format="csr"
    )

    # Combine parts into a block matrix
    d2_AB = sparse.bmat(
        [
            [If_A_kron_d0_B, d1_A_kron_Ie_B, None],
            [None, -Ie_A_kron_d1_B, d0_A_kron_If_B],
        ],
        format="csr",
    )

    return d2_AB


def get_star0_product(star0_A, star0_B):
    """
    Compute the hodge star operator on 0-forms for the product mesh of two meshes A and B

    Parameters
    --------------------------
    star0_A : scipy.sparse.dia_matrix
        (n_A,n_A) sparse matrix of star0 operator for mesh A with n_A the number of vertices in A
    star0_B : scipy.sparse.dia_matrix
        (n_B,n_B) sparse matrix of star0 operator for mesh B with n_B the number of vertices in B

    Output
    --------------------------
    star0_product : scipy.sparse.dia_matrix
        ((n_A * n_B), (n_A * n_B)) sparse matrix of star0 operator for the product mesh
    """

    star0_AB = sparse.kron(star0_A, star0_B, format="dia")
    return star0_AB


def get_star1_product(star0_A, star0_B, star1_A, star1_B):
    """
    Compute the hodge star operator on 1-forms for the product mesh of two meshes A and B

    Parameters
    --------------------------
    star0_A : scipy.sparse.dia_matrix
        (n_A,n_A) sparse matrix of star0 operator for mesh A with n_A the number of vertices in A
    star0_B : scipy.sparse.dia_matrix
        (n_B,n_B) sparse matrix of star0 operator for mesh B with n_B the number of vertices in B
    star1_A : scipy.sparse.dia_matrix
        (e_A,e_A) sparse matrix of star1 operator for mesh A with e_A the number of edges in A
    star1_B : scipy.sparse.dia_matrix
        (e_B,e_B) sparse matrix of star1 operator for mesh B with e_B the number of edges in B

    Output
    --------------------------
    star1_product : scipy.sparse.dia_matrix
        ((e_A * n_B + e_B * n_A), (e_A * n_B + e_B * n_A)) sparse matrix of star1 operator for the product mesh
    """

    star1_A_star0_B = sparse.kron(star1_A, star0_B, format="dia")
    star0_A_star1_B = sparse.kron(star0_A, star1_B, format="dia")

    star1_AB = sparse.block_diag([star1_A_star0_B, star0_A_star1_B], format="dia")
    return star1_AB


def get_star2_product(star0_A, star0_B, star1_A, star1_B, star2_A, star2_B):
    """
    Compute the hodge star operator on 2-forms for the product mesh of two meshes A and B

    Parameters
    --------------------------
    star0_A : scipy.sparse.dia_matrix
        (n_A,n_A) sparse matrix of star0 operator for mesh A with n_A the number of vertices in A
    star0_B : scipy.sparse.dia_matrix
        (n_B,n_B) sparse matrix of star0 operator for mesh B with n_B the number of vertices in B
    star1_A : scipy.sparse.dia_matrix
        (e_A,e_A) sparse matrix of star1 operator for mesh A with e_A the number of edges in A
    star1_B : scipy.sparse.dia_matrix
        (e_B,e_B) sparse matrix of star1 operator for mesh B with e_B the number of edges in B
    star2_A : scipy.sparse.dia_matrix
        (f_A,f_A) sparse matrix of star2 operator for mesh A with f_A the number of faces in A
    star2_B : scipy.sparse.dia_matrix
        (f_B,f_B) sparse matrix of star2 operator for mesh B with f_B the number of faces in B

    Output
    --------------------------
    star2_product : scipy.sparse.dia_matrix
        ((f_A * n_B + e_A*e_B + f_B * n_A), (f_A * n_B + e_A*e_B + f_B * n_A)) sparse matrix of star2 operator for the product mesh
    """

    star2_A_star0_B = sparse.kron(star2_A, star0_B, format="dia")
    star1_A_star1_B = sparse.kron(star1_A, star1_B, format="dia")
    star0_A_star2_B = sparse.kron(star0_A, star2_B, format="dia")

    star2_AB = sparse.block_diag(
        [star2_A_star0_B, star1_A_star1_B, star0_A_star2_B], format="dia"
    )
    return star2_AB


#### PRODUCT OF OPERATORS


def build_selection_matrices(n_verts, edges):
    """
    Builds the static topological selectors.
    S_src: Moves value at u to edge (u, v)
    S_tgt: Moves value at v to edge (u, v)
    """
    n_edges = len(edges)

    # Rows are edges (0 to E-1)
    rows = np.arange(n_edges)

    # S_src has 1 at (edge_idx, u)
    S_src = sparse.csr_matrix(
        (np.ones(n_edges), (rows, edges[:, 0])), shape=(n_edges, n_verts)
    )

    # S_tgt has 1 at (edge_idx, v)
    S_tgt = sparse.csr_matrix(
        (np.ones(n_edges), (rows, edges[:, 1])), shape=(n_edges, n_verts)
    )

    return S_src, S_tgt


def apply_slice_laplacian(Z, S_src, S_tgt, edge_weights, R):
    """
    Computes L * Z efficiently for multiple slices simultaneously.

    Args:
        Z:       (n_verts, n_slices) - The section data
        S_src:   (n_edges, n_verts)  - Sparse source selector
        S_tgt:   (n_edges, n_verts)  - Sparse target selector
        edge_weights:  (n_edges,)          - Geometric weights (Diagonal M)
        R:       (n_edges, n_slices) - Connection rotations (The index-dependent part) i->j for each edge
    """
    # --- 1. Gradient (d * Z) ---
    # Gather values to edges
    val_u = S_src @ Z  # (n_edges, n_slices)
    val_v = S_tgt @ Z  # (n_edges, n_slices)

    # Apply connection (Parallel Transport)
    # This single line handles the unique rotation for every slice j
    # Mathematical equivalent: d_0^(j) z on each slice j
    grad = val_v - val_u * R

    # --- 2. Inner Product (M * d * Z) ---
    # Apply cotangent weights
    # Broadcasting (n_edges, 1) * (n_edges, n_slices)
    W = edge_weights[:, None] * grad  # (n_edges, n_slices)

    # --- 3. Divergence (d_dagger * M * d * Z) ---
    # Sum back to vertices.
    # The adjoint of (v - u*R) is (v^T - conj(R)*u^T)
    div_v = S_tgt.T @ W  # (n_verts, n_slices)
    div_u = S_src.T @ (np.conj(R) * W)  # (n_verts, n_slices)

    return div_v - div_u  # (n_verts, n_slices)


def apply_slice_laplacian_block(Z_3d, S_src, S_tgt, cotans, R):
    """
    Applies L^(j) to Z[:, j, k] for all j (slices) and all k (vectors).

    Args:
        Z: (N_Verts,N_Slices, K_Vecs) - 3D Dense Input
        S_src, S_tgt: (N_Edges, N_Verts)      - Static Sparse Topology
        cotans:  (N_Edges,)                   - Static Geometry
        R:       (N_Edges, N_Slices)          - Dynamic Rotations
    """
    n_edges = S_src.shape[0]
    n_verts, n_slices, k = Z_3d.shape

    # --- 1. Collapse dimensions for Sparse Matmul ---
    # We treat (Slices * K) as a single huge batch of vectors.
    # Z_flat: (N_Verts, N_Slices * K)
    Z_flat = einops.rearrange(Z_3d, "v s k -> v (s k)")

    # --- 2. Gradient (Sparse Gather) ---
    # (N_Edges, N_Verts) @ (N_Verts, Batch) -> (N_Edges, Batch)
    val_u_flat = S_src @ Z_flat
    val_v_flat = S_tgt @ Z_flat

    # --- 3. Apply Rotations (The "Slice" dependent part) ---
    val_u = einops.rearrange(val_u_flat, "e (s k) -> e s k", s=n_slices)
    val_v = einops.rearrange(val_v_flat, "e (s k) -> e s k", s=n_slices)

    grad = val_v - val_u * R[:, :, None]  # (n_edges, n_slices, k)

    # 4. Weights
    # cotans is (n_edges,)
    W = grad * cotans[:, None, None]  # (n_edges, n_slices, k)

    # 5. Prepare for Divergence (Flatten back)
    # We need (n_edges, n_slices * k) for sparse matmul
    # W_flat = W_3d.reshape(n_edges, n_slices * k)
    W_flat = einops.rearrange(W, "e s k -> e (s k)")

    # 6. Apply Conj Rotation (Adjoint) - Careful with broadcasting again
    # We need (W * conj(R)). But W is flattened.
    # Easiest to do the multiply in 3D, then flatten.

    W_rot_flat = einops.rearrange(W * np.conj(R[:, :, None]), "e s k -> e (s k)")

    # 7. Divergence
    div_v = S_tgt.T @ W_flat
    div_u = S_src.T @ W_rot_flat

    res_flat = div_v - div_u  # (n_verts, n_slices * k)

    return einops.rearrange(res_flat, "v (s k) -> v s k", s=n_slices)


def get_tensor_product_mass_op_diag(Op1, Op2):
    n1 = Op1.shape[0]
    n2 = Op2.shape[0]
    shape = (n1 * n2, n1 * n2)
    weights = np.outer(Op1.diagonal(), Op2.diagonal()).ravel()
    dtype = weights.dtype

    def matvec(v):
        if v.ndim == 1:
            return weights * v
        else:
            return weights[:, None] * v

    def rmatvec(v):
        # The adjoint (conjugate transpose) for complex numbers
        # If real, this is identical to matvec
        w_conj = weights.conj()
        if v.ndim == 1:
            return w_conj * v
        else:
            return w_conj[:, None] * v

    Op_tensor = sparse.linalg.LinearOperator(
        shape,
        matvec=matvec,
        matmat=matvec,
        rmatvec=rmatvec,
        rmatmat=rmatvec,
        dtype=dtype,
    )

    return Op_tensor


def get_slice_product_laplacian(
    slice_connection,
    edges1,
    edges2,
    n_verts1,
    n_verts2,
    star0_1,
    star0_2,
    star1_1,
    star1_2,
    return_precondition=False,
):
    e1 = edges1.shape[0]
    e2 = edges2.shape[0]

    n1 = n_verts1
    n2 = n_verts2

    weights1 = star1_1.diagonal()  # (e1,)
    weights2 = star1_2.diagonal()  # (e2,)

    areas1 = star0_1.diagonal()  # (n1,)
    areas2 = star0_2.diagonal()  # (n2,)

    S_src1, S_tgt1 = build_selection_matrices(n1, edges1)  # (e1,n1)
    S_src2, S_tgt2 = build_selection_matrices(n2, edges2)  # (e2,n2)

    slice_eta1 = einops.rearrange(
        slice_connection[: e1 * n2], " (e1 v2) -> e1 v2", e1=e1, v2=n2
    )

    slice_eta2 = einops.rearrange(
        slice_connection[e1 * n2 :], " (v1 e2) -> e2 v1", e2=e2, v1=n1
    )

    R_1 = np.exp(1j * slice_eta1)  # (e1, n2)
    R_2 = np.exp(1j * slice_eta2)  # (e2, n1)

    if return_precondition:
        diag_L_A = (S_src1.T @ weights1) + (S_tgt1.T @ weights1)  # (n1,)
        diag_L_B = (S_src2.T @ weights2) + (S_tgt2.T @ weights2)  # (n2,)

    def matvec_laplacian(v):
        add_dim = False
        if v.ndim > 1:
            assert v.shape[1] == 1
            v = v.squeeze(-1)
            add_dim = True

        Z = einops.rearrange(
            v,
            "(v1 v2) -> v1 v2",
            v1=n1,
            v2=n2,
        )  # (n1, n2)

        # Term 1: L_A acting on columns, coupled by Mass B
        # Op: (L_A_batched @ Z) * M_B
        L_A_Z = apply_slice_laplacian(Z, S_src1, S_tgt1, weights1, R_1)  # (n1, n2)
        # Term1 = L_A_Z @ star0_2  # (n1, n2)
        # Term1 = (star0_2 @ L_A_Z.T).T  # (n1, n2)
        Term1 = L_A_Z * areas2[None, :]  # (n1, n2)

        # Term 2: L_B acting on rows, coupled by Mass A
        # We work on Z.T to treat rows as columns
        L_B_ZT = apply_slice_laplacian(Z.T, S_src2, S_tgt2, weights2, R_2)  # (n2, n1)
        # Transpose result back: M_A * (L_B * Z.T).T
        # Term2 = star0_1 @ L_B_ZT.T  # (n1, n2)
        Term2 = areas1[:, None] * L_B_ZT.T  # (n1, n2)

        res = (Term1 + Term2).ravel()
        if add_dim:
            res = res[:, None]
        return res

    def matvec_mass(v):
        add_dim = False
        if v.ndim > 1:
            assert v.shape[1] == 1
            v = v.squeeze(-1)
            add_dim = True

        Z = einops.rearrange(
            v,
            "(v1 v2) -> v1 v2",
            v1=n_verts1,
            v2=n_verts2,
        )  # (n1, n2)
        # Mass = M_A \otimes M_B  =>  M_A * Z * M_B
        # res = star0_1 @ (Z @ star0_2)
        # res = star0_1 @ (star0_2 @ Z.T).T
        res = areas1[:, None] * (areas2[None, :] * Z)
        res = res.ravel()
        if add_dim:
            res = res[:, None]
        return res

    def matmat_laplacian(X):
        k = X.shape[1]
        if k == 1:
            return matvec_laplacian(X.squeeze(-1))[:, None]

        # View as 3D Tensor: (N1, N2, K)
        Z = einops.rearrange(X, "(n1 n2) k -> n1 n2 k", n1=n1, n2=n2)

        # --- Term 1: Horizontal (L_A acts on dim 0) ---
        # Z is (N1, N2, K). Verts=N1, Slices=N2
        L_A_Z = apply_slice_laplacian_block(Z, S_src1, S_tgt1, weights1, R_1)

        # Multiply by Mass B (Contract dim 1)
        # (N1, N2, K) -> (N1, K, N2) @ (N2, N2) -> (N1, K, N2) -> (N1, N2, K)
        # term1 = einops.rearrange(
        #     (star0_2 @ einops.rearrange(L_A_Z, "n1 n2 k -> (n1 k) n2").T).T,
        #     "(n1 k) n2 -> n1 n2 k",
        #     n1=n1,
        # )
        term1 = L_A_Z * areas2[None, :, None]  # (n1, n2, k)

        # --- Term 2: Vertical (L_B acts on dim 1) ---
        # Transpose so N2 is dim 0 (Verts) and N1 is dim 1 (Slices)
        Z_T = einops.rearrange(Z, "n1 n2 k -> n2 n1 k")

        # (n2, n1, k)
        L_B_ZT = apply_slice_laplacian_block(Z_T, S_src2, S_tgt2, weights2, R_2)

        # Multiply by Mass A (Contract dim 1, which is now N1)
        # term2_T = einops.rearrange(
        #     (star0_1 @ einops.rearrange(L_B_ZT, "n2 n1 k -> (n2 k) n1").T).T,
        #     "(n2 k) n1 -> n2 n1 k",
        #     n2=n2,
        # )
        term2_T = L_B_ZT * areas1[None, :, None]  # (n2, n1, k)

        # Transpose back to (N1, N2, K)
        term2 = einops.rearrange(term2_T, "n2 n1 k -> n1 n2 k")

        # return (term1 + term2).reshape(n1 * n2, k)
        return einops.rearrange(term1 + term2, "n1 n2 k -> (n1 n2) k")

    def matmat_mass(X):
        r"""
        Mass = M_A \otimes M_B.
        Operation: M_A * Z * M_B
        """
        if X.shape[-1] == 1:
            return matvec_mass(X.squeeze(-1))[:, None]

        k = X.shape[1]
        Z = einops.rearrange(X, "(n1 n2) k -> n1 n2 k", n1=n1, n2=n2)

        # 1. Apply M_B to dimension 1 (Columns/Slices)
        # (N1, N2, K) -> (N1, K, N2) -> flatten -> @ M_B
        # step1 = einops.rearrange(
        #     (star0_2 @ einops.rearrange(Z, "n1 n2 k -> (n1 k) n2").T).T,
        #     "(n1 k) n2 -> n1 n2 k",
        #     n1=n1,
        #     n2=n2,
        #     k=k,
        # )
        step1 = Z * areas2[None, :, None]  # (n1, n2, k)

        # tmp = Z.transpose(0, 2, 1).reshape(n1 * k, n2)
        # step1 = (tmp @ self.mesh2.mass_matrix).reshape(n1, k, n2).transpose(0, 2, 1)

        # 2. Apply M_A to dimension 0 (Rows)
        # (N1, N2, K) -> (N2, K, N1) -> flatten -> @ M_A
        # step2 = einops.rearrange(
        #     (star0_1 @ einops.rearrange(step1, "n1 n2 k -> (n2 k) n1").T).T,
        #     "(n2 k) n1 -> n1 n2 k",
        #     n1=n1,
        #     n2=n2,
        #     k=k,
        # )
        step2 = step1 * areas1[:, None, None]  # (n1, n2, k)

        return einops.rearrange(step2, "n1 n2 k -> (n1 n2) k")

    shape = (n1 * n2, n1 * n2)

    Linop_Lap = sparse.linalg.LinearOperator(
        shape, matvec=matvec_laplacian, matmat=matmat_laplacian, dtype=np.complex128
    )
    # Linop_Mass = sparse.linalg.LinearOperator(
    #     shape, matvec=matvec_mass, matmat=matmat_mass, dtype=np.float64
    # )

    Linop_Mass = get_tensor_product_mass_op_diag(star0_1, star0_2)

    if return_precondition:
        term1 = np.outer(diag_L_A, star0_2.diagonal())
        term2 = np.outer(star0_1.diagonal(), diag_L_B)

        precond = term1.ravel() + term2.ravel()

        return Linop_Lap, Linop_Mass, sparse.diags(1 / (precond + 1e-10))

    return Linop_Lap, Linop_Mass


def get_tensor_product_laplacian_op(L_A, M_A, L_B, M_B):
    """
    Returns a LinearOperator for L_prod = L_A (x) M_B + M_A (x) L_B
    L_A, M_A: Sparse matrices for Mesh 1 (N1 x N1)
    L_B, M_B: Sparse matrices for Mesh 2 (N2 x N2)
    """
    n1 = L_A.shape[0]
    n2 = L_B.shape[0]
    shape = (n1 * n2, n1 * n2)

    def _matmat(X):
        # X shape is (N1*N2, K)
        k = X.shape[1]

        if k == 1:
            return _matvec(X.squeeze(-1))[:, None]

        # View as (N1, N2, K)
        # Z = X.reshape(n1, n2, k)
        Z = einops.rearrange(X, "(n1 n2) k -> n1 n2 k", n1=n1, n2=n2)

        # --- Term 1: L_A (x) M_B ---
        # Action: L_A @ Z @ M_B.T
        # We process dim 0 with L_A, dim 1 with M_B

        # 1. Apply M_B to columns (dim 1)
        # (N1, N2, K) -> flatten to (N1*K, N2) -> @ M_B
        tmp = einops.rearrange(Z, "n1 n2 k -> (n1 k) n2")
        # Z_MB = (tmp @ M_B).reshape(n1, k, n2)  # shape (N1, K, N2)
        # Z_MB = tmp @ M_B.T
        Z_MB = (M_B @ tmp.T).T  # shape (N1*K, N2)
        Z_MB = einops.rearrange(Z_MB, "(n1 k) n2 -> n1 n2 k", n1=n1, n2=n2, k=k)

        # 2. Apply L_A to rows (dim 0)
        # Permute to (N2*K, N1) -> @ L_A.T (if symmetric, just L_A)
        # Ideally: L_A @ Z_reshaped
        tmp2 = einops.rearrange(Z_MB, "n1 n2 k -> n1 (k n2)")
        term1 = L_A @ tmp2
        term1 = einops.rearrange(term1, "n1 (k n2) -> n1 n2 k", k=k)

        # --- Term 2: M_A (x) L_B ---
        # Action: M_A @ Z @ L_B.T

        # 1. Apply L_B to columns (dim 1)
        tmp3 = einops.rearrange(Z, "n1 n2 k -> (n1 k) n2")
        # Z_LB = tmp3 @ L_B.T
        Z_LB = (L_B @ tmp3.T).T  # shape (N1*K, N2)
        Z_LB = einops.rearrange(Z_LB, "(n1 k) n2 -> n1 n2 k", n1=n1, n2=n2, k=k)

        # 2. Apply M_A to rows (dim 0)
        tmp4 = einops.rearrange(Z_LB, "n1 n2 k -> n1 (k n2)")
        term2 = M_A @ tmp4
        term2 = einops.rearrange(term2, "n1 (k n2) -> n1 n2 k", k=k)

        res = term1 + term2
        res = einops.rearrange(res, "n1 n2 k -> (n1 n2) k", n1=n1, n2=n2, k=k)

        return res

    # Reuse matmat logic for matvec (K=1)
    def _matvec(v):
        # View vector as Matrix (N1, N2)
        # This is O(1) - no copy
        add_dim = False
        if v.ndim > 1:
            assert v.shape[1] == 1
            v = v.squeeze(-1)
            add_dim = True

        # Z = v.reshape(n1, n2)
        Z = einops.rearrange(v, "(n1 n2) -> n1 n2", n1=n1, n2=n2)  # (n1, n2)

        # Term 1: (L_A (x) M_B) * v  =>  L_A * Z * M_B^T
        # Operations: Sparse @ Dense @ Sparse
        # M_B is usually symmetric, so M_B.T == M_B, but we write .T for correctness
        # term1 = L_A @ (Z @ M_B.T)  # (n1, n2)
        term1 = L_A @ (M_B @ Z.T).T  # (n1, n2)

        # Term 2: (M_A (x) L_B) * v  =>  M_A * Z * L_B^T
        # term2 = M_A @ (Z @ L_B.T)  # (n1, n2)
        term2 = M_A @ (L_B @ Z.T).T  # (n1, n2)

        res = term1 + term2
        res = einops.rearrange(res, "n1 n2 -> (n1 n2)", n1=n1, n2=n2)
        if add_dim:
            res = res[:, None]
        return res

        # return _matmat(v.reshape(-1, 1)).ravel()

    return sparse.linalg.LinearOperator(
        shape, matvec=_matvec, matmat=_matmat, dtype=np.complex128
    )
