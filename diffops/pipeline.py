import os

import numpy as np
import scipy
from scipy import sparse

from . import gl_energy
from . import dec_operators
from . import geom_utils

import potpourri3d as pp3d

from .surface_mesh import SurfaceMesh
from .product_mesh import ProductMesh

import densemaps.numpy.maps as maps_np


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


class Pipeline:
    """Abstract class for the complete pipeline

    Quantities that are reused across steps (the product mesh, the
    connections, the operators) are cached on the instance.

    The constructor arguments are the default hyperparameters. Each step method
    accepts ``None`` for these and falls back to the value stored here.

    Parameters
    ----------
    mesh1, mesh2 : SurfaceMesh
        The two surfaces ``A`` and ``B``.
    lam : float
        Weight of the circular-well potential in the Ginzburg-Landau energy.
    alpha : float
        Weight of the Dirichlet term.
    n_iter : int
        Maximum number of L-BFGS iterations.
    log_interval : int
        Number of iterations between two energy printouts.
    gtol : float
        Gradient tolerance for L-BFGS.
    use_preconditioner : bool
        Whether to precondition the eigensolve building the initial section.
    landmarks : list, optional
        Landmark pairs used to build the pinning potential. See
        `compute_pin_matrix` and `read_pinned`. ``None`` (default)
        uses the unpinned energy.
    sigma : float
        Width of the Gaussian landmark wells, in mesh units. Default is fine on normalized shapes.
    pin_matrix : np.ndarray, optional
        Pre-computed pinning potential, bypassing ``landmarks``.
    seed : int, optional
        If given, seeds the global numpy RNG. Both the initial-section eigensolve
        and :func:`critical_epsilon` start LOBPCG from a random complex vector, so
        this is what makes a run reproducible.
    verbose : bool
        Default verbosity for all steps.

    # Example run

        pipe = Pipeline.from_files("A.obj", "B.obj")
        pipe.set_initial_map_nn()
        pipe.compute_connections()
        section = pipe.initialize_section()
        pipe.build_operators()
        section = pipe.optimize(section, lam=(1e1, 1e2))
        P21 = pipe.extract_map(section, direction="21")

    or, equivalently, ``P21 = Pipeline.from_files("A.obj", "B.obj").run()``.
    """

    def __init__(
        self,
        mesh1,
        mesh2,
        lam=1e2,
        alpha=1.0,
        n_iter=1000,
        log_interval=50,
        gtol=1e-8,
        use_preconditioner=True,
        landmarks=None,
        sigma=1.0 / np.sqrt(2.0),
        pin_matrix=None,
        seed=None,
        verbose=True,
    ):
        self.mesh1 = mesh1
        self.mesh2 = mesh2

        # Default hyperparameters.
        self.lam = lam
        self.alpha = alpha
        self.n_iter = n_iter
        self.log_interval = log_interval
        self.gtol = gtol
        self.use_preconditioner = use_preconditioner
        self.landmarks = landmarks
        self.sigma = sigma
        self.pin_matrix = pin_matrix
        self.seed = seed
        self.verbose = verbose

        if seed is not None:
            # The LOBPCG starts from random
            np.random.seed(seed)

        # Input correspondence.
        self._P21_input = None
        self._P12_input = None
        self._v2f_21 = None
        self._v2f_12 = None

        # Step (1): connections.
        self._curvature1 = None
        self._connection1 = None
        self._curvature2 = None
        self._connection2 = None

        # Step (2): product mesh and initial section.
        self._pmesh = None
        # self._section_init = None
        self._init_connection = None

        # Step (3): operators and optimization.
        self._operators = None
        self._eps_crit = None
        # self._u_optimized = None
        self._logger = None
        self._loggers = []

    @classmethod
    def from_files(cls, path_A, path_B, make_delaunay=True, load_pinned=True, **kwargs):
        """Load two meshes from files

        Both meshes are recentered on their area-weighted centroid and rescaled to
        unit surface area, so that hyperparameters expressed in mesh units (notably
        ``sigma``) are comparable across inputs.

        Parameters
        ----------
        path_A, path_B : str
            Paths to the two surface meshes
        make_delaunay : bool
            Whether to compute the intrinsic Delaunay triangulation of each mesh.
        load_pinned : bool
            If True (default), look for ``.pinned`` landmark files next to each
            mesh and use them as the default landmarks when both are present.
            Explicitly passing ``landmarks`` overrides this.
        **kwargs
            Forwarded to :class:`Pipeline`.

        Returns
        -------
        Pipeline
        """

        verbose = kwargs.get("verbose", True)

        if load_pinned and kwargs.get("landmarks") is None:
            pinned_A = os.path.splitext(path_A)[0] + ".pinned"
            pinned_B = os.path.splitext(path_B)[0] + ".pinned"
            if os.path.isfile(pinned_A) and os.path.isfile(pinned_B):
                kwargs["landmarks"] = cls.read_pinned(pinned_A, pinned_B)
                if verbose:
                    print(f"Loaded {len(kwargs['landmarks'])} landmark pairs.")

        v1, f1 = pp3d.read_mesh(path_A)
        v2, f2 = pp3d.read_mesh(path_B)

        v_areas1 = pp3d.vertex_areas(v1, f1)
        v_areas2 = pp3d.vertex_areas(v2, f2)

        center_mass1 = np.average(v1, axis=0, weights=v_areas1)
        center_mass2 = np.average(v2, axis=0, weights=v_areas2)

        v1_n = (v1 - center_mass1) / np.sqrt(v_areas1.sum())
        v2_n = (v2 - center_mass2) / np.sqrt(v_areas2.sum())

        mesh1 = SurfaceMesh(v1_n.copy(), f1.copy())
        mesh2 = SurfaceMesh(v2_n.copy(), f2.copy())

        if make_delaunay:
            mesh1.make_delaunay(verbose=verbose)
            mesh2.make_delaunay(verbose=verbose)

        return cls(mesh1, mesh2, **kwargs)

    # Convenience accessors

    @property
    def f1(self):
        """(f1, 3) Faces of ``mesh1`` on the ORIGINAL triangulation."""
        return self.mesh1.faces_extrinsic

    @property
    def f2(self):
        """(f2, 3) Faces of ``mesh2`` on the ORIGINAL triangulation."""
        return self.mesh2.faces_extrinsic

    @property
    def pmesh(self):
        """The product mesh ``A x B``, built on first access."""
        if self._pmesh is None:
            self._pmesh = ProductMesh(self.mesh1, self.mesh2)
        return self._pmesh

    @property
    def connections(self):
        """``(connection1, connection2)`` per-surface connection 1-forms."""
        self._require_connections()
        return (self._connection1, self._connection2)

    @property
    def curvatures(self):
        """``(curvature1, curvature2)`` per-surface spin curvatures."""
        self._require_connections()
        return (self._curvature1, self._curvature2)

    @property
    def logger(self):
        """The :class:`~diffops.gl_energy.GLLogger` of the last optimization."""
        return self._logger

    @property
    def eps_crit(self):
        """The critical Ginzburg-Landau parameter, if it has been computed."""
        return self._eps_crit

    @property
    def P21_input(self):
        """The input precise map from ``mesh1`` to ``mesh2``."""
        return self._P21_input

    @property
    def P12_input(self):
        """The input precise map from ``mesh2`` to ``mesh1``."""
        return self._P12_input

    # Input correspondence

    def set_initial_map(self, P21, P12):
        """Set the input correspondence from two precise maps.

        Both maps are converted into the per-face intrinsic vector fields expected
        when building the initial section.

        Parameters
        ----------
        P21 : densemaps precise map
            Map sending vertices of ``mesh1`` onto ``mesh2``.
        P12 : densemaps precise map
            Map sending vertices of ``mesh2`` onto ``mesh1``.

        Returns
        -------
        Pipeline
            ``self``
        """
        self._P21_input = P21
        self._P12_input = P12

        self._v2f_21 = precise_map_to_vfield(self.mesh1, P21.v2f_21, P21.bary_coords)
        self._v2f_12 = precise_map_to_vfield(self.mesh2, P12.v2f_21, P12.bary_coords)
        return self

    def set_initial_map_nn(self):
        """Set the input correspondence to a nearest-neighbor map in ambient space.

        Returns
        -------
        Pipeline
            ``self``
        """

        P21 = maps_np.EmbPreciseMap(self.mesh1.vertices, self.mesh2.vertices, self.f1)
        P12 = maps_np.EmbPreciseMap(self.mesh2.vertices, self.mesh1.vertices, self.f2)
        return self.set_initial_map(P21, P12)

    # Step 1

    def compute_connections(self, verbose=None):
        """Step (1): build the complex line bundle connection on each surface.

        Returns
        -------
        Pipeline
            ``self``
        """
        if self._resolve(verbose, self.verbose):
            print("Computing surface connections...")

        self._curvature1, self._connection1 = compute_surface_connection(self.mesh1)
        self._curvature2, self._connection2 = compute_surface_connection(self.mesh2)
        return self

    # Step 2

    def initialize_section(self, use_preconditioner=None, verbose=None, **kwargs):
        """Build the initial section on the product space. Just an eigenvector problem

        Parameters
        ----------
        use_preconditioner : bool, optional
            Overrides the instance default.
        verbose : bool, optional
            Overrides the instance default.
        **kwargs
            Forwarded to :meth:`ProductMesh.compute_initial_section` (e.g. ``tol``,
            ``method``, ``trivial_con_method``).

        Returns
        -------
        u_init : np.ndarray
            The initial section on the product space.
        """
        self._require_connections()
        if self._v2f_21 is None:
            raise ValueError(
                "No input correspondence: call set_initial_map() or set_initial_map_nn() first."
            )

        section_init, self._init_connection = self.pmesh.compute_initial_section(
            v2f_21=self._v2f_21,
            v2f_12=self._v2f_12,
            con_angle_init_1=self._connection1,
            con_angle_init_2=self._connection2,
            curv_init_1=self._curvature1,
            curv_init_2=self._curvature2,
            use_preconditioner=self._resolve(
                use_preconditioner, self.use_preconditioner
            ),
            verbose=self._resolve(verbose, self.verbose),
            **kwargs,
        )
        return section_init

    # Step 3

    def build_operators(self, verbose=None):
        """Assemble the product-space connection Laplacian and mass operators.

        Returns
        -------
        Pipeline
            ``self``, for chaining.
        """
        self._require_connections()
        if self._resolve(verbose, self.verbose):
            print("Building bundle operators...")

        Lap_fem, M_scal, factors = build_bundle_operators(
            self.mesh1,
            self.mesh2,
            self._connection1,
            self._connection2,
            self._curvature1,
            self._curvature2,
        )
        self._operators = (Lap_fem, M_scal, factors)
        return self

    def compute_critical_epsilon(self):
        """Critical Ginzburg-Landau parameter, computed once and cached.

        Returns
        -------
        float
        """
        if self._eps_crit is None:
            if self._operators is None:
                self.build_operators()
            _, _, (L1, M1, L2, M2) = self._operators
            self._eps_crit = critical_epsilon(L1, M1, L2, M2)
        return self._eps_crit

    @staticmethod
    def read_pinned(path_A, path_B):
        """Read a pair of ``.pinned`` landmark files.

        Line ``i`` of ``path_A`` is paired with line ``i`` of ``path_B``. Each line
        is a whitespace-separated list of vertex indices: a single index denotes a
        point landmark, several indices denote a curve landmark

        Parameters
        ----------
        path_A, path_B : str
            Paths to the ``.pinned`` files of ``mesh1`` and ``mesh2``.

        Returns
        -------
        list of (np.ndarray, np.ndarray)
            Landmark pairs, suitable for :meth:`compute_pin_matrix`.
        """

        def _read(path):
            with open(path) as f:
                return [
                    np.array([int(tok) for tok in line.split()], dtype=np.int64)
                    for line in f
                    if line.strip()
                ]

        lmks1, lmks2 = _read(path_A), _read(path_B)
        if len(lmks1) != len(lmks2):
            raise ValueError(
                f"Landmark files disagree: {path_A} has {len(lmks1)} landmarks, "
                f"{path_B} has {len(lmks2)}."
            )
        return list(zip(lmks1, lmks2))

    def compute_pin_matrix(self, landmarks, sigma=None):
        """Build the landmark pinning potential ``V`` on the product space.

            V = min_k ( 1 - exp( -d1(., L1_k)^2 / (2 sigma^2)
                                 -d2(., L2_k)^2 / (2 sigma^2) ) )

        Parameters
        ----------
        landmarks : list of (int or array-like, int or array-like)
            Landmark pairs, list of vertex indices or tuples
        sigma : float, optional
            Width of the Gaussian wells

        Returns
        -------
        np.ndarray
            ``(n1 * n2,)`` potential, flattened.
        """
        sigma = self._resolve(sigma, self.sigma)

        # The factorization is the expensive part, so build each solver once.
        heat_solver1 = pp3d.MeshHeatMethodDistanceSolver(
            self.mesh1.vertices, self.mesh1.faces_extrinsic
        )
        heat_solver2 = pp3d.MeshHeatMethodDistanceSolver(
            self.mesh2.vertices, self.mesh2.faces_extrinsic
        )

        V = np.ones((self.mesh1.n_vertices, self.mesh2.n_vertices), dtype=np.float64)

        for lmk1, lmk2 in landmarks:
            d1 = heat_solver1.compute_distance_multisource(np.atleast_1d(lmk1))
            d2 = heat_solver2.compute_distance_multisource(np.atleast_1d(lmk2))

            e1 = np.exp(-(d1**2) / (2 * sigma**2))
            e2 = np.exp(-(d2**2) / (2 * sigma**2))

            np.minimum(V, 1.0 - np.outer(e1, e2), out=V)

        return V.ravel()

    def optimize(
        self,
        initial_section,
        landmarks=None,
        sigma=None,
        epsilon=None,
        lam=None,
        alpha=None,
        pin_matrix=None,
        n_iter=None,
        log_interval=None,
        gtol=None,
        log_values=False,
        log_values_interval=10,
        log_values_dir=None,
    ):
        """Step (3): minimize the Ginzburg-Landau energy of the section.

        Any argument left as ``None`` falls back to the instance default, except
        ``epsilon`` which defaults to :meth:`compute_critical_epsilon`.

        Parameters
        ----------
        initial_section : np.ndarray
            Starting section, e.g. from :meth:`initialize_section`.
        landmarks : list, optional
            Landmark pairs (see :meth:`compute_pin_matrix`). Ignored when
            ``pin_matrix`` is given explicitly.
        sigma : float, optional
            Gaussian width for the landmark wells.
        lam : float or sequence of float, optional
            Weight of the circular-well potential. A sequence is treated as an
            annealing schedule: each stage warm-starts from the previous result.
            Passing ``[100, 10]`` reproduces the two-stage schedule used for
            several figures in the paper.
        pin_matrix : np.ndarray, optional
            Pre-computed pinning potential, bypassing ``landmarks``.

        Returns
        -------
        u_optimized : np.ndarray
            The optimized section on the product space.
        """
        if self._operators is None:
            self.build_operators()

        Lap_fem, M_scal, _ = self._operators
        if epsilon is None:
            epsilon = self.compute_critical_epsilon()

        pin_matrix = self._resolve(pin_matrix, self.pin_matrix)
        landmarks = self._resolve(landmarks, self.landmarks)
        if pin_matrix is None and landmarks is not None and len(landmarks) > 0:
            pin_matrix = self.compute_pin_matrix(landmarks, sigma=sigma)

        # A scalar lam and a schedule follow the same path; each stage warm-starts
        # from the previous one.
        lam_schedule = np.atleast_1d(self._resolve(lam, self.lam))
        if lam_schedule.size == 0:
            raise ValueError("`lam` is empty: nothing to optimize.")

        self._loggers = []
        u_optimized = initial_section.copy()
        for lam_i in lam_schedule:
            u_optimized, logger = _run_gl_optimization(
                u_optimized,
                Lap_fem,
                M_scal,
                epsilon,
                lam_i,
                self._resolve(alpha, self.alpha),
                pin_matrix=pin_matrix,
                n_iter=self._resolve(n_iter, self.n_iter),
                log_interval=self._resolve(log_interval, self.log_interval),
                log_values=log_values,
                log_values_interval=log_values_interval,
                log_values_dir=log_values_dir,
                gtol=self._resolve(gtol, self.gtol),
            )
            self._loggers.append(logger)

        self._logger = self._loggers[-1]
        return u_optimized

    # Step 4

    def extract_map(self, section, direction="21", as_precise_map=True, verbose=None):
        """Step (4): read the correspondence off a section.

        Parameters
        ----------
        section : np.ndarray
            Section to evaluate
        direction : {"21", "12"}
            ``"21"`` maps vertices of ``mesh2`` onto ``mesh1``, ``"12"`` the other
            way around. See `section_to_map`.
        as_precise_map : bool
            If True (default), wrap the image points into a ``densemaps`` precise
            map on the ORIGINAL triangulation. If False, return the raw
            ``(image_points, source, displ)`` triple.
        verbose : bool, optional
            Overrides the instance default.

        Returns
        -------
        densemaps precise map, or tuple
        """
        image_points, source, displ = section_to_map(
            self.pmesh,
            section,
            self.connections,
            self.curvatures,
            direction=direction,
            verbose=self._resolve(verbose, self.verbose),
        )

        if not as_precise_map:
            return image_points, source, displ

        import densemaps.numpy.maps as maps_np

        if direction == "21":
            return maps_np.EmbPreciseMap(self.mesh1.vertices, image_points, self.f1)
        return maps_np.EmbPreciseMap(self.mesh2.vertices, image_points, self.f2)

    # Everything at once

    def run(self, direction="21", **optimize_kwargs):
        """Run the four steps end to end and return the resulting correspondence.

        Uses a nearest-neighbor initialization if no input map has been set. Every
        intermediate quantity stays available on the instance afterwards.

        Parameters
        ----------
        direction : {"21", "12"}
            Direction of the returned map.
        **optimize_kwargs
            Forwarded to :meth:`optimize`.

        Returns
        -------
        densemaps precise map
        """
        if self._v2f_21 is None:
            self.set_initial_map_nn()

        self.compute_connections()
        initial_section = self.initialize_section()
        self.build_operators()
        section = self.optimize(initial_section=initial_section, **optimize_kwargs)
        return self.extract_map(section=section, direction=direction)

    # Internals

    def _require_connections(self):
        if self._connection1 is None:
            raise ValueError("No connections: call compute_connections() first.")

    @staticmethod
    def _resolve(value, default):
        """Instance default for arguments left as ``None``."""
        return default if value is None else value
