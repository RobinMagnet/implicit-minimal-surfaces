import numpy as np


def find_zeros_in_triangle(
    z_vals,
    rho_vals,
    omega_vals,
    n_steps=10,
    n_newton=5,
    skip_converged=True,
    convergence_tol=1e-8,
    return_history=False,
    verbose=False,
):
    """
    Finds zeros of a of complex section over curved triangles.

    Solves equation using homotopy between flat and curved triangle.

    Optionally skips Newton iterations for points that converge early at each level of c

    Args:
        z_vals: (N, 3) Complex field values [zi, zj, zk] for each triangle
        rho_vals: (N, 3) Connection angles [rho_ij, rho_jk, rho_ki] for each triangle
        omega_vals: (N,) Target curvature Omega for each triangle
        n_steps: Number of homotopy steps
        n_newton: Maximum Newton iterations per homotopy step
        skip_converged: If True, skip Newton iterations for converged points at each level
        convergence_tol: Convergence tolerance for residual
        return_history: If True, return convergence history
        verbose: Print diagnostic information

    Returns:
        barycentric_coords: (N, 3) array of [b_i, b_j, b_k] coordinates
        valid_mask: (N,) boolean array indicating successful convergence
        history: (optional) (T, N, 3) convergence history if return_history=True
    """
    N = z_vals.shape[0]

    # Extract components
    # all are (N,)
    zi, zj, zk = z_vals[:, 0], z_vals[:, 1], z_vals[:, 2]
    rho_jk, rho_ki, rho_ij = rho_vals[:, 0], rho_vals[:, 1], rho_vals[:, 2]

    rot_ij = np.angle(zj / (zi * np.exp(1j * rho_ij)))  # (N,)
    rot_jk = np.angle(zk / (zj * np.exp(1j * rho_jk)))  # (N,)
    rot_ki = np.angle(zi / (zk * np.exp(1j * rho_ki)))  # (N,)

    indices = (rot_ij + rot_jk + rot_ki + omega_vals) / (2 * np.pi)
    if np.any(np.abs(indices) < 0.5):
        raise ValueError("Some triangles don't have zeros")

    mask_ij = np.isclose(np.abs(rot_ij), np.pi)
    mask_jk = np.isclose(np.abs(rot_jk), np.pi)
    mask_ki = np.isclose(np.abs(rot_ki), np.pi)

    mask_edge = mask_ij | mask_jk | mask_ki
    mask_face = ~mask_edge

    # Initialize results
    bary_coords = np.zeros((N, 3))

    valid_mask = np.zeros(N, dtype=bool)

    if np.any(mask_edge):
        zi_mod, zj_mod, zk_mod = np.abs(zi), np.abs(zj), np.abs(zk)

        # Zero on edge ij (k=0) -> bj = |zi|/(|zi|+|zj|)
        if np.any(mask_ij):
            idx = mask_ij
            bj_val = zi_mod[idx] / (zi_mod[idx] + zj_mod[idx])
            bary_coords[idx, 1] = bj_val
            bary_coords[idx, 0] = 1.0 - bj_val
            bary_coords[idx, 2] = 0.0

        if np.any(mask_jk):
            idx = mask_jk
            bk_val = zj_mod[idx] / (zj_mod[idx] + zk_mod[idx])
            bary_coords[idx, 2] = bk_val
            bary_coords[idx, 1] = 1.0 - bk_val
            bary_coords[idx, 0] = 0.0

        if np.any(mask_ki):
            idx = mask_ki
            bi_val = zk_mod[idx] / (zk_mod[idx] + zi_mod[idx])
            bary_coords[idx, 0] = bi_val
            bary_coords[idx, 2] = 1.0 - bi_val
            bary_coords[idx, 1] = 0.0

        valid_mask[mask_edge] = True

    if np.any(mask_face):
        idx = mask_face
        bary_face, valid_mask_face, stats = _homotopic_triangle_zero_solve(
            z_vals[idx],
            np.stack([rot_jk[idx], rot_ki[idx], rot_ij[idx]], axis=1),
            omega_vals[idx],
            n_steps=n_steps,
            n_newton=n_newton,
            skip_converged=skip_converged,
            convergence_tol=convergence_tol,
            return_history=False,
            verbose=verbose,
        )
        bary_coords[idx] = bary_face
        valid_mask[idx] = valid_mask_face

        if verbose:
            _print_diagnostics(
                stats["valid_mask"],
                stats["is_inside"],
                stats["has_converged"],
                stats["final_u_mag"],
                np.sum(idx),
                n_edges=np.sum(mask_edge),
            )

    return bary_coords, valid_mask


def _homotopic_triangle_zero_solve(
    z_vals,
    rot_vals,
    omega_vals,
    n_steps=10,
    n_newton=5,
    skip_converged=True,
    convergence_tol=1e-8,
    return_history=False,
    verbose=False,
):
    """
    Finds zeros of a of complex section over curved triangles.

    Solves equation using homotopy between flat and curved triangle.

    Optionally skips Newton iterations for points that converge early at each level of c

    Args:
        z_vals: (N, 3) Complex field values [zi, zj, zk] for each triangle
        rho_vals: (N, 3) Connection angles [rho_ij, rho_jk, rho_ki] for each triangle
        omega_vals: (N,) Target curvature Omega for each triangle
        n_steps: Number of homotopy steps
        n_newton: Maximum Newton iterations per homotopy step
        skip_converged: If True, skip Newton iterations for converged points at each level
        convergence_tol: Convergence tolerance for residual
        return_history: If True, return convergence history
        verbose: Print diagnostic information

    Returns:
        barycentric_coords: (N, 3) array of [b_i, b_j, b_k] coordinates
        valid_mask: (N,) boolean array indicating successful convergence
        history: (optional) (T, N, 3) convergence history if return_history=True
    """
    N = z_vals.shape[0]

    # Extract components
    # all are (N,)
    zi, zj, zk = z_vals[:, 0], z_vals[:, 1], z_vals[:, 2]
    # rho_jk, rho_ki, rho_ij = rho_vals[:, 0], rho_vals[:, 1], rho_vals[:, 2]

    # Compute moduli and relative angles
    zi_mod = np.abs(zi)  # (N,)
    zj_mod = np.abs(zj)  # (N,)
    zk_mod = np.abs(zk)  # (N,)

    rot_jk, rot_ki, rot_ij = rot_vals[:, 0], rot_vals[:, 1], rot_vals[:, 2]

    # Compute deltas for homotopy
    delta_ij = omega_vals - 2 * rot_ij + rot_jk + rot_ki  # (N,)
    delta_jk = omega_vals + rot_ij - 2 * rot_jk + rot_ki  # (N,)
    delta_ki = omega_vals + rot_ij + rot_jk - 2 * rot_ki  # (N,)

    # Initialize state - all points start at centroid
    bj = np.full(N, 1.0 / 3.0)  # (N,)
    bk = np.full(N, 1.0 / 3.0)  # (N,)

    history = []
    if return_history:
        history.append(np.stack([1 - bj - bk, bj, bk], axis=1))  # (N, 3)

    # Homotopy continuation - ALL points go through ALL steps
    for step in range(n_steps + 1):
        t = step / n_steps

        # Compute homotopy parameters for ALL points
        r_ij = rot_ij + (1 - t) * delta_ij / 3  # (N,)
        r_jk = rot_jk + (1 - t) * delta_jk / 3  # (N,)
        r_ki = rot_ki + (1 - t) * delta_ki / 3  # (N,)
        curr_omega = t * omega_vals  # (N,)

        # print((r_ij + r_jk + r_ki + curr_omega) / (2 * np.pi))

        # Track which points still need Newton iterations at this level
        if skip_converged:
            active_mask = np.ones(N, dtype=bool)  # (N,)

        if verbose and step % max(1, n_steps // 10) == 0:
            print(f"Homotopy step {step}/{n_steps} (t={t:.2f})")

        # Newton iteration
        for newton_it in range(n_newton):
            # Compute residual for ALL points
            if skip_converged:
                # Compute residual only for active points
                u_active, du_dbj_active, du_dbk_active = _compute_residual_and_jacobian(
                    bj[active_mask],
                    bk[active_mask],
                    zi_mod[active_mask],
                    zj_mod[active_mask],
                    zk_mod[active_mask],
                    r_ij[active_mask],
                    r_ki[active_mask],
                    curr_omega[active_mask],
                )

                # Check convergence
                u_mag_active = np.abs(u_active)  # (num_active,)
                newly_converged_local = u_mag_active < convergence_tol  # (num_active,)

                # Update global active mask
                active_indices = np.where(active_mask)[0]  # (num_active,)
                active_mask[active_indices[newly_converged_local]] = False

                # update only still-active points
                u_active = u_active[~newly_converged_local]
                du_dbj_active = du_dbj_active[~newly_converged_local]
                du_dbk_active = du_dbk_active[~newly_converged_local]

                # If all points converged at this level, move to next homotopy step
                if not active_mask.any():
                    if verbose:
                        print(
                            f"  All points converged after {newton_it + 1} Newton iterations"
                        )
                    break

                # Newton step only for still-active points
                dbj_active, dbk_active = _newton_step(
                    u_active, du_dbj_active, du_dbk_active
                )

                # Update only active points
                bj[active_mask] += dbj_active
                bk[active_mask] += dbk_active
            else:
                # Standard: compute and update all points
                u, du_dbj, du_dbk = _compute_residual_and_jacobian(
                    bj, bk, zi_mod, zj_mod, zk_mod, r_ij, r_ki, curr_omega
                )

                u_mag = np.abs(u)
                if np.all(u_mag < convergence_tol):
                    if verbose:
                        print(
                            f"  All points converged after {newton_it + 1} Newton iterations"
                        )
                    break

                dbj, dbk = _newton_step(u, du_dbj, du_dbk)
                bj += dbj
                bk += dbk

            # Clip to valid region just in case
            bj = np.clip(bj, 0, 1.0)
            bk = np.clip(bk, 0, 1.0)

            if return_history:
                history.append(np.stack([1 - bj - bk, bj, bk], axis=1))  # (N, 3)

        u = _compute_vfield(bj, bk, zi_mod, zj_mod, zk_mod, r_ij, r_ki, curr_omega)
        final_u_mag = np.abs(u)
        if not np.all(final_u_mag < convergence_tol):
            if verbose:
                n_not_converged = np.sum(final_u_mag >= convergence_tol)
                print(
                    f"  Warning: {n_not_converged}/{N} points did not converge at homotopy step {step}"
                )

    # Final evaluation
    bi = 1.0 - bj - bk
    u = _compute_vfield(bj, bk, zi_mod, zj_mod, zk_mod, r_ij, r_ki, curr_omega)
    final_u_mag = np.abs(u)

    # Validity checks
    is_inside = (bi >= -1e-3) & (bj >= -1e-3) & (bk >= -1e-3)
    has_converged = final_u_mag < 1e-4
    valid_mask = is_inside & has_converged

    stats = {
        "valid_mask": valid_mask,
        "is_inside": is_inside,
        "has_converged": has_converged,
        "final_u_mag": final_u_mag,
    }
    # if verbose:
    #     _print_diagnostics(valid_mask, is_inside, has_converged, final_u_mag, N)

    result = np.stack([bi, bj, bk], axis=1)  # (N, 3)
    # print(result)
    if return_history:
        return result, valid_mask, stats, np.array(history)
    else:
        return result, valid_mask, stats


def find_edge_edge_zeros(z_ik, z_jk, z_il, z_jl, rho_ij, rho_kl):
    """
    Vectorized edge-edge zero finder on quadrilateral faces of the product mesh.

    For each edge-edge face with edge eA = (i→j) on mesh1 and edge eB = (k→l)
    on mesh2, the section is bilinearly interpolated as:

      z(s,t) = (1-s)(1-t)·z_ik + s(1-t)·r_ij·z_jk + (1-s)t·r_kl·z_il
             + st·r_ij·r_kl·z_jl

    where r_ij = exp(iρ_ij) and r_kl = exp(iρ_kl) are the parallel transport
    factors. Dividing by z_ik yields w(s,t) = d + s·b + t·c + s·t·a with d=1.
    Setting w=0 and eliminating s gives a quadratic in t:

      Q2·t² + Q1·t + Q0 = 0

    with Q2 = Im(ā·c), Q1 = Im(ā·d + b̄·c), Q0 = Im(b̄·d).
    Then s = -Re(c·t + d) / Re(a·t + b).

    Parameters
    ----------
    z_ik : np.ndarray, shape (N,) complex
        Section values at corner (s=0, t=0), i.e. Z[i, k].
    z_jk : np.ndarray, shape (N,) complex
        Section values at corner (s=1, t=0), i.e. Z[j, k].
    z_il : np.ndarray, shape (N,) complex
        Section values at corner (s=0, t=1), i.e. Z[i, l].
    z_jl : np.ndarray, shape (N,) complex
        Section values at corner (s=1, t=1), i.e. Z[j, l].
    rho_ij : np.ndarray, shape (N,)
        Connection angle along edge A (mesh1), for edge i→j.
    rho_kl : np.ndarray, shape (N,)
        Connection angle along edge B (mesh2), for edge k→l.

    Returns
    -------
    s_vals : np.ndarray, shape (N,)
        Parameter along edge A (NaN if no valid intersection).
    t_vals : np.ndarray, shape (N,)
        Parameter along edge B (NaN if no valid intersection).
    valid : np.ndarray, shape (N,) bool
        Whether a valid intersection was found in [0,1]×[0,1].
    """
    N = len(z_ik)

    r_ij = np.exp(1j * rho_ij)
    r_kl = np.exp(1j * rho_kl)

    d = 1
    b = z_jk / (r_ij * z_ik) - 1
    c = z_il / (r_kl * z_ik) - 1
    a = z_jl / (r_ij * r_kl * z_ik) - b - c - 1

    # Solve Q2·t² + Q1·t + Q0 = 0 for t, then recover s from t.
    Q2 = np.imag(np.conj(a) * c)
    Q1 = np.imag(np.conj(a) * d + np.conj(b) * c)
    Q0 = np.imag(np.conj(b) * d)

    delta = Q1**2 - 4 * Q2 * Q0

    # Initialize output
    s_vals = np.full(N, np.nan)
    t_vals = np.full(N, np.nan)
    valid = np.zeros(N, dtype=bool)

    eps = 1e-10  # tolerance for [0,1] range check

    # --- Linear case: Q2 ≈ 0 ---
    linear = np.abs(Q2) < 1e-12
    lin_ok = linear & (np.abs(Q1) > 1e-12)

    if np.any(lin_ok):
        idx = np.where(lin_ok)[0]
        t_cand = -Q0[idx] / Q1[idx]
        denom = np.real(a[idx] * t_cand + b[idx])
        good_denom = np.abs(denom) > 1e-12
        s_cand = np.full_like(t_cand, np.nan)
        s_cand[good_denom] = (
            -np.real(c[idx[good_denom]] * t_cand[good_denom] + d) / denom[good_denom]
        )

        ok = (
            (t_cand >= -eps)
            & (t_cand <= 1 + eps)
            & (s_cand >= -eps)
            & (s_cand <= 1 + eps)
        )
        sel = idx[ok]
        t_vals[sel] = np.clip(t_cand[ok], 0, 1)
        s_vals[sel] = np.clip(s_cand[ok], 0, 1)
        valid[sel] = True

    # --- Quadratic case ---
    quad = ~linear
    if np.any(quad):
        idx_q = np.where(quad)[0]
        disc = delta[idx_q]
        real_roots = disc >= -eps  # allow small negative for numerical noise
        disc_clipped = np.clip(disc[real_roots], 0, None)

        if np.any(real_roots):
            idx = idx_q[real_roots]
            sqrt_d = np.sqrt(disc_clipped)
            inv_2Q2 = 0.5 / Q2[idx]

            t1 = (-Q1[idx] + sqrt_d) * inv_2Q2
            t2 = (-Q1[idx] - sqrt_d) * inv_2Q2

            denom1 = np.real(a[idx] * t1 + b[idx])
            denom2 = np.real(a[idx] * t2 + b[idx])

            s1 = np.where(
                np.abs(denom1) > 1e-12,
                -np.real(c[idx] * t1 + d) / denom1,
                np.nan,
            )
            s2 = np.where(
                np.abs(denom2) > 1e-12,
                -np.real(c[idx] * t2 + d) / denom2,
                np.nan,
            )

            ok1 = (
                (t1 >= -eps)
                & (t1 <= 1 + eps)
                & (s1 >= -eps)
                & (s1 <= 1 + eps)
                & ~np.isnan(s1)
            )
            ok2 = (
                (t2 >= -eps)
                & (t2 <= 1 + eps)
                & (s2 >= -eps)
                & (s2 <= 1 + eps)
                & ~np.isnan(s2)
            )

            # Prefer root 1; fall back to root 2
            sel1 = idx[ok1]
            t_vals[sel1] = np.clip(t1[ok1], 0, 1)
            s_vals[sel1] = np.clip(s1[ok1], 0, 1)
            valid[sel1] = True

            use2 = ok2 & ~ok1
            sel2 = idx[use2]
            t_vals[sel2] = np.clip(t2[use2], 0, 1)
            s_vals[sel2] = np.clip(s2[use2], 0, 1)
            valid[sel2] = True

    return s_vals, t_vals, valid


# Helper functions


def _compute_vfield(bj, bk, zi_mod, zj_mod, zk_mod, r_ij, r_ki, curr_omega):
    """Compute vector field modulus for given barycentric coords."""
    bi = 1.0 - bj - bk

    # Symmetric basis terms
    phi_j = r_ij + curr_omega * bk
    phi_k = -r_ki - curr_omega * bj

    Ti = zi_mod * bi
    Tj = bj * zj_mod * np.exp(1j * phi_j)
    Tk = bk * zk_mod * np.exp(1j * phi_k)
    u = Ti + Tj + Tk

    return u


def _compute_residual_and_jacobian(
    bj, bk, zi_mod, zj_mod, zk_mod, r_ij, r_ki, curr_omega
):
    """Compute residual u and Jacobian derivatives for Newton's method."""
    bi = 1.0 - bj - bk

    # Symmetric basis terms
    phi_j = r_ij + curr_omega * bk
    phi_k = -r_ki - curr_omega * bj

    Ej = zj_mod * np.exp(1j * phi_j)
    Ek = zk_mod * np.exp(1j * phi_k)

    Ti = zi_mod * bi
    Tj = bj * Ej
    Tk = bk * Ek
    u = Ti + Tj + Tk

    # Jacobian
    du_dbj = -zi_mod + Ej + Tk * (-1j * curr_omega)
    du_dbk = -zi_mod + Tj * (1j * curr_omega) + Ek

    return u, du_dbj, du_dbk


def _newton_step(u, du_dbj, du_dbk):
    """
    Compute Newton step by solving 2x2 linear system.

    Args:
        u: residual
        du_dbj, du_dbk: Jacobian
    """
    # Extract real and imaginary parts
    A = du_dbj.real
    B = du_dbk.real
    C = du_dbj.imag
    D = du_dbk.imag
    R1 = -u.real
    R2 = -u.imag

    # Solve 2x2 system
    det = A * D - B * C
    inv_det = np.where(np.abs(det) > 1e-12, 1.0 / det, 0.0)

    dbj = (D * R1 - B * R2) * inv_det
    dbk = (-C * R1 + A * R2) * inv_det

    return dbj, dbk


def _print_diagnostics(valid_mask, is_inside, has_converged, final_u_mag, N, n_edges=0):
    """Print diagnostic information about convergence."""
    root_found = has_converged
    geom_valid = is_inside

    drifted = root_found & (~geom_valid)
    failed_convergence = ~root_found

    print(f"Stats: Total {N}")
    print(f"  - On edges: {n_edges}")
    print(f"  - In face: {N - n_edges}")
    print(f"    - Perfect (converged & inside): {valid_mask.sum()}")
    print(f"    - Drifted (converged but outside): {drifted.sum()}")
    print(f"    - Diverged (failed to converge): {failed_convergence.sum()}")
    if failed_convergence.any():
        max_residual = final_u_mag[failed_convergence].max()
        print(f"    Max residual: {max_residual:.2e}")
