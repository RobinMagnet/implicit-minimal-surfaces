import numpy as np
from scipy import sparse

from .utils import max_angle_diff, wrap_to_pi


def solve_min_quad_with_fixed(A, C, d):
    """
    solve min ||x||_A^2  s.t. C x = d
    """
    n = A.shape[1]
    assert A.shape[0] == n

    lhs = sparse.bmat(
        [
            [A, C.T],
            [C, None],
        ],
        format="csc",
    )

    if d.ndim == 1:
        rhs = np.concatenate([np.zeros(n), d], axis=0)
        # res = sparse.linalg.spsolve(lhs, rhs)[:n]
    else:
        rhs = np.concatenate([np.zeros((n, d.shape[1])), d], axis=0)
        # solver = sparse.linalg.splu(lhs)
        # res = solver.solve(rhs)[:n]
    res = sparse.linalg.spsolve(lhs, rhs)[:n]
    if d.ndim == 2 and d.shape[1] == 1:
        return res[:, None]
    return res


def solve_poisson_hodge(rhs, d1, star1_inv, return_potential=False):

    lhs = d1 @ (star1_inv @ d1.T)

    phi = sparse.linalg.spsolve(lhs, rhs)

    res = star1_inv @ (d1.T @ phi)

    if rhs.ndim == 2 and rhs.shape[1] == 1:
        res = res[:, None]
        phi = phi[:, None]

    if return_potential:
        return res, phi
    return res


def modify_connection_to_curvature_hodge(
    d1,
    star1_inv,
    initial_connection_angle,
    initial_curvature,
    target_curvature,
    check_solution=True,
    return_potential=False,
) -> np.ndarray:
    """
    Supports batched inputs:
      initial_connection_angle: (E,) or (E, p)
      initial_curvature:        (F,) or (F, p)
      target_curvature:         (F,) or (F, p)
    p is determined as the max explicit 2nd dimension across all inputs.
    Output is squeezed to 1D only when all inputs are 1D (p_effective == 1).

    If return_potential=True, returns (connection_angle_new, phi) where phi is the
    Hodge potential on faces, same shape as target_curvature (up to squeezing).
    """

    # Assert shapes
    n_f, n_e = d1.shape
    assert initial_connection_angle.shape[0] == n_e
    assert initial_curvature.shape[0] == n_f
    assert target_curvature.shape[0] == n_f

    p_effective = max(
        arr.shape[1] if arr.ndim == 2 else 1
        for arr in [initial_connection_angle, initial_curvature, target_curvature]
    )
    squeeze_output = p_effective == 1

    # Normalize to 2D; numpy broadcasting handles mixed (F,) / (F, p) cases
    conn_2d = (
        initial_connection_angle[:, None]
        if initial_connection_angle.ndim == 1
        else initial_connection_angle
    )
    init_curv_2d = (
        initial_curvature[:, None] if initial_curvature.ndim == 1 else initial_curvature
    )
    tgt_curv_2d = (
        target_curvature[:, None] if target_curvature.ndim == 1 else target_curvature
    )

    curvdiff = tgt_curv_2d - init_curv_2d  # (F, p) via broadcasting

    if not np.allclose(curvdiff.sum(axis=0), 0):
        diff_sum = curvdiff.sum(axis=0)
        raise ValueError(
            f"Curvature mismatch! Sums: 2 pi * {diff_sum / (2*np.pi)}. "
            "Ensure that input curvatures are consistent."
        )

    res_hodge, phi = solve_poisson_hodge(
        curvdiff, d1, star1_inv, return_potential=True
    )  # (E, p), (F, p)

    connection_angle_new = conn_2d + res_hodge  # (E, p)

    connection_angle_new = wrap_to_pi(connection_angle_new)

    if check_solution:
        test_curv = d1 @ connection_angle_new  # (F, p)
        tgt_for_check = np.broadcast_to(tgt_curv_2d, test_curv.shape)
        err = max_angle_diff(test_curv.ravel(), np.array(tgt_for_check).ravel())
        if err > 1e-5:
            print(f"Warning: Max curvature error: {err}")

    if squeeze_output:
        connection_angle_new = connection_angle_new.squeeze(-1)
        phi = phi.squeeze(-1)

    if return_potential:
        return connection_angle_new, phi
    return connection_angle_new


def modify_connection_to_curvature_l2(
    d1,
    initial_connection_angle,
    initial_curvature,
    target_curvature,
    check_solution=True,
    star1=None,
) -> np.ndarray:
    """
    target curvature can be (m,) or (m,p) for several computations
    """

    n_faces, n_edges = d1.shape
    assert initial_curvature.ndim == 1, "initial curvature must be 1d for now"

    if target_curvature.ndim == 1:
        n_parallel = 1
        curvdiff = (target_curvature - initial_curvature)[:, None]  # (m,1)
    else:
        n_parallel = target_curvature.shape[1]
        curvdiff = target_curvature - initial_curvature[:, None]  # (m, p)

    if not np.allclose(curvdiff.sum(axis=0), 0):
        diff_sum = curvdiff.sum(axis=0)
        raise ValueError(
            f"Curvature mismatch! Sums: 2 pi * {diff_sum/ (2*np.pi)}. "
            "Ensure that input curvatures are consistent."
        )

    if star1 is not None:
        star1 = sparse.eye(n_edges)

    res_l2 = solve_min_quad_with_fixed(
        star1,
        d1,
        curvdiff,
    )  # (n_e, p) or (n_e,1)

    connection_angle_new = initial_connection_angle[:, None] + res_l2  # (n_e, p)

    connection_angle_new = wrap_to_pi(connection_angle_new)

    if check_solution:
        # Verify the result has curvature concentrated ONLY at the target face
        test_curv = d1 @ connection_angle_new  # (n_f, p)
        # Deviation from the perfect spike
        err = max_angle_diff(
            test_curv.ravel(),
            target_curvature.ravel(),
        )
        if err > 1e-5:
            print(f"Warning: Max curvature error: {err}")

    if n_parallel == 1:
        return connection_angle_new.squeeze(-1)
    return connection_angle_new


def get_trivial_connection_from_spin(
    d1, spin_connection, spin_curvature, face_idx, check_solution=True, star1=None
) -> np.ndarray:
    """Get the trivial connection at a given face index.

    Args:
        face_idx (int or np.array): The index of the face.

    Returns:
        np.ndarray: The trivial connection at the specified face.
    """

    n_faces, n_edges = d1.shape

    n_parallel = 1 if np.issubdtype(type(face_idx), np.integer) else len(face_idx)

    target_curvature = np.zeros((n_faces, n_parallel))  # (m, p)
    target_curvature[face_idx, np.arange(n_parallel)] = 2 * np.pi

    curvdiff = target_curvature - spin_curvature[:, None]  # (m, p)
    if not np.allclose(curvdiff.sum(axis=0), 0):
        diff_sum = curvdiff.sum(axis=0)
        raise ValueError(
            f"Curvature mismatch! Sums: {diff_sum}. "
            "Ensure input is actually a Spin connection (sums to 2pi)."
        )

    if star1 is not None:
        star1 = sparse.eye(n_edges)

    constrain_mat = sparse.bmat(
        [
            [star1, d1.T],
            [d1, None],
        ],
        format="csc",
    )  # (n_e + n_f, n_e + n_f)
    rhs = np.concatenate(
        [np.zeros((n_edges, n_parallel)), curvdiff], axis=0
    )  # (n_e + n_f, p)

    if n_parallel == 1:
        eta_l2 = sparse.linalg.spsolve(constrain_mat, rhs)[:n_edges]
        eta_l2 = eta_l2[:, None]  # (n_e, 1)
    else:
        solver = sparse.linalg.splu(constrain_mat)
        eta_l2 = solver.solve(rhs)[:n_edges]  # (n_e, p)

    eta_correction = spin_connection[:, None] + eta_l2  # (n_e, p)

    eta_correction = wrap_to_pi(eta_correction)

    if check_solution:
        # Verify the result has curvature concentrated ONLY at the target face
        test_curv = d1 @ eta_correction  # (n_f, p)
        # Deviation from the perfect spike
        err = max_angle_diff(
            test_curv.ravel(),
            target_curvature.ravel(),
        )
        if err > 1e-5:
            print(f"Warning: Max curvature error: {err}")

    if n_parallel == 1:
        return eta_correction.squeeze(-1), target_curvature.squeeze(-1)
    return eta_correction, target_curvature
