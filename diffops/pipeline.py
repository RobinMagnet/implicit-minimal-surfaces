import numpy as np
import scipy
from scipy import sparse

from . import gl_energy
from . import dec_operators


# Initial map
def precise_map_to_vfield(mesh, v2f, bary_coords):
    """Convert a vertex-to-barycentric map on the extrinsic structure into a vertex to face map for the intrinsic triangulation.


    Parameters
    ----------
    mesh : SurfaceMesh
        Mesh on which the target points live (i.e. the codomain of the map).
    v2f : np.ndarray
        (n,) Index of the target face (on the *original* triangulation) for each point.
    bary_coords : np.ndarray
        (n, 3) Barycentric coordinates of each target point inside its face.

    Returns
    -------
    v2f_intrinsic : np.ndarray
        (n,) Face index -- on the *intrinsic* (Delaunay) triangulation -- that each
        tangent vector points into.
    """

    if not mesh.mesh.has_flipped:
        return v2f

    # Target point (face + barycentric coords) -> tangent vector in vertex coordinates.
    source_inds, vfield_complex = mesh.bary_to_vfield(
        v2f, bary_coords, on_intrinsic=False
    )
    # Tangent vector -> the intrinsic face it points into.
    return mesh.vfield_to_face(
        vfield_complex, mesh.vertices, source_indices=source_inds, on_intrinsic=True
    )


# CLB structure
def compute_surface_connection(mesh):
    """Build the complex line bundle connection on a mesh

    Parameters
    ----------
    mesh : SurfaceMesh

    Returns
    -------
    curvature : np.ndarray
        (n,) Per-vertex spin curvature (half the Gaussian curvature).
    connection : np.ndarray
        (e,) Connection 1-form (per-edge rotation angle) compatible with ``curvature``.
    """
    curvature = mesh.curvature_spin.copy()
    connection = mesh.get_connection_from_skyscraper(curvature, method="hodge")
    return curvature, connection


# All the differential operators
def build_bundle_operators(
    mesh1, mesh2, connection1, connection2, curvature1, curvature2
):
    """Assemble the product-space connection Laplacian and mass operators.

    Parameters
    ----------
    mesh1, mesh2 : SurfaceMesh
    connection1, connection2 : np.ndarray
        Per-surface connection 1-forms (from :func:`compute_surface_connection`).
    curvature1, curvature2 : np.ndarray
        Per-surface spin curvatures (from :func:`compute_surface_connection`).

    Returns
    -------
    Lap_fem : scipy.sparse.linalg.LinearOperator
        (n1*n2, n1*n2) Product-space connection Laplacian ``L1 (x) M2 + M1 (x) L2``.
    M_scal : scipy.sparse.linalg.LinearOperator
        (n1*n2, n1*n2) Lumped scalar mass (dual volumes on ``A x B``).
    factor_operators : tuple
        ``(L1, M1, L2, M2)`` the per-surface bundle operators, reusable e.g. for
        :func:`critical_epsilon`.
    """
    L1 = mesh1.bundle_laplacian_fem(connection_angle=connection1, curvature=curvature1)
    M1 = mesh1.bundle_mass_fem(connection_angle=connection1, curvature=curvature1)

    L2 = mesh2.bundle_laplacian_fem(connection_angle=connection2, curvature=curvature2)
    M2 = mesh2.bundle_mass_fem(connection_angle=connection2, curvature=curvature2)

    Lap_fem = dec_operators.get_tensor_product_laplacian_op(L1, M1, L2, M2)
    M_scal = dec_operators.get_tensor_product_mass_op_diag(mesh1.star0, mesh2.star0)

    return Lap_fem, M_scal, (L1, M1, L2, M2)


def critical_epsilon(L1, M1, L2, M2):
    """Critical Ginzburg-Landau parameter ``eps_crit`` for the product space.

    Below the smallest eigenvalue of the product-space connection Laplacian, the only
    critical point of the energy is ``z == 0``.
    The value ``eps_crit = 1 / sqrt(mu1 + mu2)`` gives a scale for the interface-width parameter ``epsilon``.

    Parameters
    ----------
    L1, M1, L2, M2 : sparse matrices
        Per-surface bundle Laplacians and mass matrices
        (from :func:`build_bundle_operators`).

    Returns
    -------
    float
        The critical epsilon.
    """
    mu1 = _compute_smallest_eigenvalue(L1, M1)
    mu2 = _compute_smallest_eigenvalue(L2, M2)
    return 1.0 / np.sqrt(mu1 + mu2)


# Map conversion
def section_to_map(
    pmesh, section, connections, curvatures, direction="21", verbose=False
):
    """Extract a vertex correspondence from a complex section on the product space.

    Locates, for each vertex, the single zero of the section along its slice (the
    encoded image point), then converts the intrinsic barycentric location into a 3D
    displacement on the original triangulation. Returns the resolved image points;
    the caller typically wraps them in a ``densemaps`` precise map for transfer/plots
    (kept in the notebook so this module stays free of the ``densemaps`` dependency).

    Parameters
    ----------
    pmesh : ProductMesh
    section : np.ndarray
        The complex section on the product space.
    connections : tuple
        ``(connection1, connection2)`` per-surface connection 1-forms.
    curvatures : tuple
        ``(curvature1, curvature2)`` per-surface spin curvatures.
    direction : {"21", "12"}
        ``"21"`` maps vertices of ``mesh2`` onto ``mesh1`` (via ``find_map_21``), so the
        image points land on ``pmesh.mesh1``; ``"12"`` maps vertices of ``mesh1`` onto
        ``mesh2``, with image points on ``pmesh.mesh2``. The target mesh is picked
        automatically from ``pmesh``.
    verbose : bool

    Returns
    -------
    image_points : np.ndarray
        (n, 3) The 3D image of each mapped vertex, living on the target mesh.
    source : np.ndarray
        (n,) Index of the target-mesh vertex each displacement is anchored at.
    displ : np.ndarray
        (n, 3) The 3D displacement from ``target_mesh.vertices[source]`` to the image.
    """
    if direction == "21":
        # find_map_21 sends vertices of mesh2 to faces of mesh1 -> images live on mesh1.
        mesh_target = pmesh.mesh1
        v2f, bary = pmesh.find_map_21(
            section,
            connection_angle=connections,
            curvature_form=curvatures,
            verbose=verbose,
        )
    elif direction == "12":
        # find_map_12 sends vertices of mesh1 to faces of mesh2 -> images live on mesh2.
        mesh_target = pmesh.mesh2
        v2f, bary = pmesh.find_map_12(
            section,
            connection_angle=connections,
            curvature_form=curvatures,
            verbose=verbose,
        )
    else:
        raise ValueError(f"direction must be '21' or '12', got {direction!r}")

    # Intrinsic barycentric location -> displacement on the target ORIGINAL triangulation.
    source, displ = mesh_target.intrinsic_bary_to_original(
        v2f, bary, vertices=mesh_target.vertices
    )
    image_points = mesh_target.vertices[source] + displ
    return image_points, source, displ


def _compute_smallest_eigenvalue(
    L, M, X0=None, tol=1e-6, maxiter=4000, verbose=False, return_evec=False
):
    """Computes the smallest eigenvalue of the generalized problem Lx = lambda Mx."""
    # Use random start for LOBPCG
    n = L.shape[0]
    if X0 is None:
        X = np.random.randn(n, 1) + 1j * np.random.randn(n, 1)
    else:
        X = X0

    evals, evecs = sparse.linalg.lobpcg(
        L, X, B=M, largest=False, tol=tol, maxiter=maxiter, verbosityLevel=int(verbose)
    )
    if return_evec:
        return evals[0], evecs[:, 0]
    return evals[0]


def _run_gl_optimization(
    u_start,
    Lap,
    M,
    epsilon,
    lam,
    alpha,
    pin_matrix=None,
    n_iter=1000,
    log_interval=50,
    log_values=False,
    log_values_interval=10,
    log_values_dir=None,
    gtol=1e-5,
):
    """Runs L-BFGS to minimize the Ginzburg-Landau energy."""
    # Normalize input to avoid initial gradient shock
    # u_start = u_start / np.abs(u_start).mean()

    logger = gl_energy.GLLogger(
        Lap,
        M,
        epsilon,
        lam,
        alpha,
        log_interval=log_interval,
        log_values=log_values,
        log_values_interval=log_values_interval,
        save_values_dir=log_values_dir,
    )
    if pin_matrix is None:
        res = scipy.optimize.minimize(
            gl_energy.gl_energy_and_grad,
            x0=gl_energy.cast_to_r2(u_start),
            jac=True,
            args=(Lap, M, epsilon, lam, alpha),
            method="L-BFGS-B",
            callback=logger,
            options={"maxiter": n_iter, "gtol": gtol, "ftol": 1e-12},
        )
    else:
        res = scipy.optimize.minimize(
            gl_energy.gl_energy_and_grad_lmks2,
            x0=gl_energy.cast_to_r2(u_start),
            jac=True,
            args=(Lap, M, epsilon, lam, alpha, pin_matrix),
            method="L-BFGS-B",
            callback=logger,
            options={
                "maxiter": n_iter,
                "gtol": gtol,
                "maxcor": 10,
                "maxls": 20,
            },  # , "ftol": 1e-4},
        )

    if not res.success:
        print(f"  > Warning: Optimization stopped: {res.message}")
    else:
        print(f"  > Optimization converged in {res.nit} iterations: {res.message}")
    u_final = gl_energy.cast_to_complex(res.x)
    return u_final, logger


# Public aliases for the two building blocks above (step (3)).
# ``compute_smallest_eigenvalue`` feeds the epsilon heuristic (see ``critical_epsilon``),
# and ``run_gl_optimization`` performs the actual Ginzburg-Landau minimization.
compute_smallest_eigenvalue = _compute_smallest_eigenvalue
run_gl_optimization = _run_gl_optimization
