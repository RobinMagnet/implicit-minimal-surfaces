import numpy as np


def compute_face_areas_from_len(lengths: np.ndarray):
    """
    Compute per-face areas of a triangular mesh from lengths of edges.

    Parameters
    -----------------------------
    lengths : np.ndarray
        (m,3) array  of lengths of edges for each face

    Returns
    -----------------------------
    faces_areas : np.ndarray or float
        (m,) or float array of per-face areas
    """
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


def compute_tan_half_angle_at_c(len_a, len_b, len_c):

    test_ab = len_a >= len_b
    a = np.where(test_ab, len_a, len_b)
    b = np.where(test_ab, len_b, len_a)
    c = len_c

    mu = np.where(b >= c, c - (a - b), b - (a - c))

    tan_halfangle = np.sqrt((((a - b) + c) * mu) / ((a + (b + c)) * ((a - c) + b)))

    return tan_halfangle


def compute_angle_at_c(len_a, len_b, len_c):
    """
    Compute angle at vertex C given lengths of edges a,b,c using the stable law of cosines.

    Parameters
    ----------
    a : np.array
        (n_faces,) Array of lengths of edge opposite to vertex A
    b : np.array
        (n_faces,) Array of lengths of edge opposite to vertex B
    c : np.array
        (n_faces,) Array of lengths of edge opposite to vertex C

    Returns
    -------
    angle_C : np.array
        (n_faces,) Array of angles at vertex C
    """
    # test_ab = len_a >= len_b
    # a = np.where(test_ab, len_a, len_b)
    # b = np.where(test_ab, len_b, len_a)
    # c = len_c

    # mu = np.where(b >= c, c - (a - b), b - (a - c))

    # angle_C = 2 * np.arctan(
    #     np.sqrt((((a - b) + c) * mu) / ((a + (b + c)) * ((a - c) + b)))
    # )

    tan_halfangle = compute_tan_half_angle_at_c(len_a, len_b, len_c)

    angle_C = 2 * np.arctan(tan_halfangle)

    return angle_C


def compute_internal_angles_from_length_stable(length: np.ndarray):
    """
    Compute internal angles from lengths of edges.

    Parameters
    ----------
    length : np.array
        (n_faces,3) Array of lengths of edges (length of opposite edge of each vertex)

    Returns
    -------
    angles : np.array
        (n_faces,3) Array of internal angles
    """

    len_a = length[:, 0]
    len_b = length[:, 1]
    len_c = length[:, 2]

    angle_C = compute_angle_at_c(len_a, len_b, len_c)
    angle_B = compute_angle_at_c(len_a, len_c, len_b)
    angle_A = np.pi - angle_B - angle_C

    return np.stack([angle_A, angle_B, angle_C], axis=1)


def compute_cotan_weights_from_length(opposite_len: np.ndarray):
    """
    Compute the cotangent weights for each edge in the mesh from lengths of edges.

    Parameters
    --------------------------
    opposite_len : (m,3) array defining lengths of edges opposite to each vertex in each face

    Output
    --------------------------
    cotan_weights : (m,3) array of cotangent weights for each edge in each face
    """
    a_sq = np.square(opposite_len[:, 0])
    b_sq = np.square(opposite_len[:, 1])
    c_sq = np.square(opposite_len[:, 2])

    # Area
    face_areas = compute_face_areas_from_len(opposite_len)  # (m,)

    cotan0 = (b_sq + c_sq - a_sq) / (4 * face_areas)  # (m,)
    cotan1 = (c_sq + a_sq - b_sq) / (4 * face_areas)  # (m,)
    cotan2 = (a_sq + b_sq - c_sq) / (4 * face_areas)  # (m,)

    cotan_weights = np.stack([cotan0, cotan1, cotan2], axis=1)  # (m,3)

    return cotan_weights


def compute_flipped_edge_length(l12, l13, l14, l32, l43):
    r"""
    Compute the length of the flipped edge using Kahan's stable formula.

        Before Flip:                 After Flip:
        Shared Edge: (v1, v3)        Shared Edge: (v2, v4)

              v2                            v2
             /  \                         / | \
            / f0  \                      /  |  \
          v1------v3                    v1  |   v3
            \ f1 /                       \  |  /
             \  /                         \ | /
              v4                            v4

        Triangles:                   Triangles:
        (v1, v3, v2)                 (v1, v4, v2)  <- New Face A
        (v1, v4, v3)                 (v2, v4, v3)  <- New Face B

    """

    l12 = np.asanyarray(l12)
    l13 = np.asanyarray(l13)
    l14 = np.asanyarray(l14)
    l32 = np.asanyarray(l32)
    l43 = np.asanyarray(l43)
    # 1. Compute the two angles at vertex 1 adjacent to the shared edge l13
    # Angle in triangle (1,3,2) at vertex 1 (between l12 and l13)
    # The side opposite to vertex 1 is l32.
    theta_1 = compute_angle_at_c(l13, l12, l32)

    # Angle in triangle (1,4,3) at vertex 1 (between l14 and l13)
    # The side opposite to vertex 1 is l43.
    theta_2 = compute_angle_at_c(l14, l13, l43)

    # 2. Total angle at vertex 1 spanning the new diagonal
    theta_total = theta_1 + theta_2

    # 3. Compute l24 using Kahan's stable formula from Section 7
    # c = sqrt( (a-b)^2 + 4ab * sin^2(C/2) )
    # Here, 'a' and 'b' are l12 and l14, and 'C' is theta_total.

    term_diff = (l12 - l14) ** 2
    term_sin = 4 * l12 * l14 * (np.sin(theta_total / 2) ** 2)

    l24 = np.sqrt(term_diff + term_sin)

    if l24.ndim == 0:
        return l24.item()

    return l24
