import numpy as np
from scipy import sparse


def wrap_to_pi(x):
    return np.angle(np.exp(1j * x))


def angle_diff(a, b):
    return np.angle(np.exp(1j * (a - b)))


def max_angle_diff(a, b):
    return np.max(np.abs(angle_diff(a, b)))


def barycentric_to_precise(faces, face_match, bary_coord, n_vertices=None):
    """
    Transforms set of barycentric coordinates into a precise map

    Parameters
    ----------------------------
    faces      :
        (m,3) - Set of faces defined by index of vertices.
    face_match :
        (n2,) - indices of the face assigned to each point
    bary_coord :
        (n2,3) - barycentric coordinates of each point within the face
    n_vertices : int
        number of vertices in the target mesh (on which faces are defined)

    Returns
    ----------------------------
    precise_map : scipy.sparse.csr_matrix
        (n2,n1) - precise point to point map
    """
    if n_vertices is None:
        n_vertices = 1 + faces.max()

    n_points = face_match.shape[0]

    v0 = faces[face_match, 0]  # (n2,)
    v1 = faces[face_match, 1]  # (n2,)
    v2 = faces[face_match, 2]  # (n2,)

    I = np.arange(n_points)  # (n2)

    In = np.concatenate([I, I, I])
    Jn = np.concatenate([v0, v1, v2])
    Sn = np.concatenate([bary_coord[:, 0], bary_coord[:, 1], bary_coord[:, 2]])

    precise_map = sparse.csr_matrix((Sn, (In, Jn)), shape=(n_points, n_vertices))
    return precise_map
