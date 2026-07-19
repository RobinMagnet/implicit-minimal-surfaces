import numpy as np
from scipy import sparse
from .utils import wrap_to_pi, max_angle_diff


def sqrt_lc_undirected(d1_csr, ue_angles, Omega, tol=1e-10, dense_threshold=150000):
    """
    d1_csr   : (F,E) csr_matrix with entries in {-1,0,+1} (signed DEC boundary)
    ue_angles: (E,) LC undirected edge angles in (-pi,pi]
    Omega    : (F,) face curvature from normalized angles (your cone-LC)
    returns  : (E,) square-root LC angles on undirected edges
    """

    # 1) naive half angles
    u0 = 0.5 * ue_angles

    # 2) face-parity bits r_f via signed d1 (holonomy residual ≈ 0 vs ≈ π)
    R = np.exp(1j * (d1_csr @ u0 - 0.5 * Omega))  # uses signed ±1
    r = (np.real(R) < 0).astype(np.uint8)  # r in {0,1}

    # 3) build |d1| mod 2 as a boolean CSR (orientation drops in GF(2))
    A_bool = (d1_csr != 0).astype(np.uint8)  # CSR with data ∈ {0,1}

    # choose solver: dense when small, sparse-sets when large
    F, E = A_bool.shape
    if F * E <= dense_threshold**2:
        b = _solve_mod2_dense(A_bool.toarray(), r)  # fast & simple
    else:
        b = _solve_mod2_sparse_sets(A_bool, r)  # memory-friendly

    # 4) apply parity flips: add π on edges with b_e=1
    eta_half = wrap_to_pi(u0 + np.pi * b)

    # --- sanity checks (S^1 equalities) ---
    # edge_err = np.max(np.abs(np.angle(np.exp(1j * (2 * eta_half - ue_angles)))))
    edge_err = max_angle_diff(2 * eta_half, ue_angles)
    if edge_err > tol:
        raise AssertionError(f"[sqrt LC] edge square mismatch (max {edge_err:.2e} rad)")

    face_res = np.angle(np.exp(1j * (d1_csr @ eta_half - 0.5 * Omega)))
    if np.max(np.abs(face_res)) > tol:
        raise AssertionError("[sqrt LC] face holonomy mismatch after parity fix")

    return eta_half


def _solve_mod2_dense(A01, y):
    """Dense Gaussian elimination over GF(2). A01: (F,E) uint8/0-1; y: (F,) uint8/0-1."""
    A = (A01.copy().astype(np.uint8)) % 2
    b = y.copy().astype(np.uint8)
    m, n = A.shape
    row = 0
    pivcols = []
    for col in range(n):
        piv = next((i for i in range(row, m) if A[i, col]), None)
        if piv is None:
            continue
        if piv != row:
            A[[row, piv]] = A[[piv, row]]
            b[[row, piv]] = b[[piv, row]]
        pivcols.append(col)
        # eliminate column from all other rows
        nz = np.nonzero(A[:, col])[0]
        nz = nz[nz != row]
        if nz.size:
            A[nz, :] ^= A[row, :]
            b[nz] ^= b[row]
        row += 1
        if row == m:
            break
    # consistency check
    if np.any((A[row:, :].sum(1) == 0) & (b[row:] == 1)):
        raise ValueError("Parity system inconsistent (check inputs).")
    # back-sub with free vars = 0
    x = np.zeros(n, dtype=np.uint8)
    for ridx, col in enumerate(pivcols):
        x[col] = b[ridx]
    return x


def _solve_mod2_sparse_sets(A_bool_csr, y):
    """
    Sparse elimination over GF(2) using Python sets.
    Memory ∝ nnz; fast enough for big meshes without densifying.
    """
    F, E = A_bool_csr.shape
    indptr, indices = A_bool_csr.indptr, A_bool_csr.indices
    rows = [set(indices[indptr[i] : indptr[i + 1]]) for i in range(F)]
    rhs = [int(v) for v in y.tolist()]
    used_row = [False] * F
    pivot_row_for_col = {}  # col -> row

    for col in range(E):
        # find a pivot row not yet used that contains 'col'
        piv = None
        for i in range(F):
            if not used_row[i] and col in rows[i]:
                piv = i
                break
        if piv is None:
            continue
        used_row[piv] = True
        pivot_row_for_col[col] = piv

        # eliminate this column from all other rows that contain it
        for i in range(F):
            if i != piv and col in rows[i]:
                # rows[i] ^= rows[piv]  (symmetric difference mod 2)
                # if len(rows[i]) < len(rows[piv]):
                #     # swap to minimize churn
                #     rows[i], rows[piv] = rows[piv], rows[i]
                #     rhs[i], rhs[piv] = rhs[piv], rhs[i]
                rows[i] ^= rows[piv]
                rhs[i] ^= rhs[piv]

    # consistency: empty row with rhs=1 ⇒ no solution (shouldn't happen on closed meshes)
    for i in range(F):
        if not rows[i] and rhs[i]:
            raise ValueError("Parity system inconsistent (component-level).")

    # build solution with free vars = 0 (columns without pivot stay 0)
    x = np.zeros(E, dtype=np.uint8)
    for col, piv in pivot_row_for_col.items():
        x[col] = rhs[piv] & 1
    return x
