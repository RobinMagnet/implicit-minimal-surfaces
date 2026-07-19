import numpy as np
import pyvista as pv


import numpy as np
import pyvista as pv

import os
import numpy as np

import pyvista as pv

pv.set_jupyter_backend("trame")


class ToyMesh:
    """
    A toy mesh class with minimal attributes to be compatible with the plotting functions.
    """

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    @property
    def faces_extrinsic(self):
        """Return the faces of the mesh in extrinsic coordinates."""
        return self.faces

    @property
    def n_faces(self):
        """Return the number of faces in the mesh."""
        return self.faces.shape[0]

    @property
    def n_vertices(self):
        """Return the number of vertices in the mesh."""
        return self.vertices.shape[0]


def normalize(f, vmin=0, vmax=1):
    """
    Normalize a function or a set of functions between vmin and vmax

    Parameters
    ----------------------------
    f : (n,) or (n,p) - one or multiple functions
    vmin : minimum value for the normalized function(s)
    vmax : maximum value for the normalized function(s)

    Output
    ---------------------------
    f_normalized : (n,) or (n,p) - normalized function(s)
    """
    if f.ndim == 1:
        f_norm = f - np.min(f)
        f_norm = vmin + (vmax - vmin) * f_norm / np.max(f_norm)

    else:
        f_norm = f - np.min(f, axis=0, keepdims=True)
        f_norm = vmin + (vmax - vmin) * f_norm / np.max(f_norm, axis=0, keepdims=True)

    return f_norm


def vertices_to_rgb_basic(vertices):
    """
    Transforms (x,y,z) coordinates into RGB values using normalization

    Parameters
    ----------------------------
    vertices : (n,3) - x,y,z coordinates of vertices

    Output
    ----------------------------
    cmap : (n,3) RGB value for each vertex
    """
    cmap = normalize(vertices, vmin=0, vmax=1)
    return cmap


def vertices_to_rgb_pretty(vertices, param=[-2, -1, 3]):
    """
    Transforms (x,y,z) coordinates into RGB values using a prettier procedure, parametrized by
    a rearangement of the array [1,2,3] up to sign flip and column reordering.
    Default parameter value is [-2, -1, 3].

    Parameters
    ----------------------------
    vertices : (n,3) - x,y,z coordinates of vertices
    param    : (3,) - rearangement of the [1,2,3] array up to sign flip and column reordering.

    Output
    ----------------------------
    cmap : (n,3) RGB value for each vertex
    """
    param = np.asarray(param)

    if np.any(np.sort(np.abs(param)) != np.arange(1, 4)):
        raise ValueError("'param' should use a reorganization of \
                          [1,2,3] up to sign flip and column switch")

    # Invert some colors and switch some channels
    cmap = np.sign(param)[None, :] * vertices
    cmap = cmap[:, np.abs(param) - 1]

    cmap = normalize(np.cos(normalize(cmap)))

    return cmap


def vert2rgb(vertices, pretty=False, param=[-2, -1, 3]):
    """
    Convert (x,y,z) coordinates of vertices to RGB values, using either `vertices_to_rgb_basic` or
    `vertices_to_rgb_basic`.

    If n_colors is specified as a positive integer, each channel only supports the given number of
    colors.

    Parameters
    ----------------------------
    vertices : (n,3) - x,y,z coordinates of vertices
    pretty   : bool - whether to use the 'pretty' procedure.
    param    : (3,) - rearangement of the [1,2,3] array up to sign flip and column reordering. Only
               used if `pretty` is set to "True".
    n_colors : int - If positive, limits the number of possible colors per channel

    Output
    ----------------------------
    cmap : (n,3) RGB value for each vertex
    """
    if pretty:
        cmap = vertices_to_rgb_pretty(vertices, param=param)
    else:
        cmap = vertices_to_rgb_basic(vertices)

    return cmap


def vert2_rgb_mesh(mesh, pretty=False, param=[-2, -1, 3]):
    """
    Similar to `vert2rgb` but takes a TriMesh as an input.
    """
    cmap = vert2rgb(mesh.vertices, pretty=pretty, param=param)
    return cmap


def load_texture(texture):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "data")

    if texture is None:
        texture = "texture_1.jpg"

    if os.path.isfile(texture):
        texture_path = texture

    elif os.path.isfile(os.path.join(data_dir, texture)):
        texture_path = os.path.join(data_dir, texture)

    else:
        raise ValueError(f"Texture file {texture} not found")

    return pv.read_texture(texture_path)


def triangles_to_cells(faces):
    """
    Convert list of faces to cells

    Parameters
    ------------------------------
    faces : np.ndarray or list
        (n,3) - list of faces

    Output
    ------------------------------
    cells : np.ndarray
        (n,4) - list of cells
    """
    if faces is None:
        return None
    cells = np.zeros((len(faces), 4), dtype=int)
    cells[:, 0] = 3
    cells[:, 1:] = faces

    return cells


def toPV(mesh=None, vertices=None, faces=None, cmap=None, vfield=None, uv=None):
    """
    Convert a pyFM.TriMesh object to a pyvista object

    Parameters
    ------------------------------
    mesh       : pyFM.TriMesh
        mesh object to convert
    cmap       : np.ndarray or list
        (m|n,) or (m|n, 3) - scalar or RGB values for each face or vertex
    vfield     : np.ndarray or list
        (m,3) or (n,3) - vector field for each face or vertex

    Output
    ------------------------------
    pv_mesh : pyvista.PolyData
        mesh object in pyvista format
    """
    if mesh is not None:
        mesh_pv = pv.PolyData(mesh.vertices, triangles_to_cells(mesh.faces_extrinsic))
    elif vertices is not None:
        if faces is None:
            mesh_pv = pv.PolyData(vertices)
        else:
            mesh_pv = pv.PolyData(vertices, triangles_to_cells(faces))
    else:
        raise ValueError("Either mesh or vertices must be provided")

    if cmap is not None:
        if cmap.shape[0] == mesh.n_vertices:
            mesh_pv.point_data["cmap"] = cmap  # (n,) or (n,3)
        else:
            assert cmap.shape[0] == mesh.n_faces
            mesh_pv.cell_data["cmap"] = cmap  # (m,)  or (m,3)

    if vfield is not None:
        if vfield.shape[0] == mesh.n_vertices:
            # mesh_pv.point_data['vfield'] = vfield  # (n,3)
            mesh_pv.point_data.set_vectors(vfield, name="vfield")
        else:
            assert vfield.shape[0] == mesh.n_faces
            # mesh_pv.cell_data['vfield'] = vfield  # (m,3)
            mesh_pv.cell_data.set_vectors(vfield, name="vfield")
        # mesh_pv.set_vectors(vfield, name='vfield')

        # mesh_pv['vfield'] = vfield

        # mesh_pv.set_active_vectors('vfield')
    if uv is not None:
        mesh_pv.active_texture_coordinates = uv

    return mesh_pv


def plot(
    mesh,
    cmap=None,
    wireframe=False,
    line_width=None,
    texture=None,
    interpolate_before_map=True,
    uv=None,
    show_colorbar=False,
    smooth=False,
    opacity=1,
    colormap="viridis",
    clim=None,
    pl=None,
    return_plot=False,
):
    """
    Plot a mesh with pyvista

    Parameters
    ------------------------------
    mesh       : pyFM.TriMesh - mesh object to plot
    point_size : float - size of the points
    cmap       : np.ndarray or list - (n,) or (n,3) - scalar or RGB values for each face or vertex
    wireframe  : bool - whether to show the mesh as a wireframe
    line_width : float - width of the wireframe
    show_colorbar : bool - whether to show the colorbar
    smooth     : bool - whether to use smooth shading
    colormap   : str - colormap to use
    pl         : meshplot.Plotter - plotter of meshplot
    return_plot : bool - whether to return the plotter

    Output
    ------------------------------
    Viewer - meshplot.Viewer class
    """
    is_pointcloud = mesh.n_faces == 0
    if uv is not None:
        if cmap is not None:
            print("WARNING: UV and cmap are both activated. Using UV only")
        cmap = None
    mesh_pv = toPV(mesh, cmap=cmap, uv=uv)

    if uv is not None:
        texture = load_texture(texture)

    scalars = None
    if cmap is not None:
        cmap = np.asarray(cmap)
        scalars = "cmap"
        if cmap.ndim == 1:
            is_rgb_cmap = False
        elif cmap.ndim == 2:
            is_rgb_cmap = True
        else:
            raise ValueError("cmap must be either (n,) or (n,3)")
    else:
        is_rgb_cmap = False

    show_plot = False
    if pl is None:
        show_plot = True if not return_plot else False
        pl = pv.Plotter()
    if is_pointcloud:
        pl.add_points(
            mesh_pv,
            point_size=None,
            scalars=scalars,
            cmap=colormap,
            rgb=is_rgb_cmap,
            render_points_as_spheres=True,
            style="points",
            show_scalar_bar=show_colorbar,
        )
    else:
        pl.add_mesh(
            mesh_pv,
            scalars=scalars,
            clim=clim,
            cmap=colormap,
            rgb=is_rgb_cmap,
            color="white",
            interpolate_before_map=interpolate_before_map,
            point_size=None,
            show_edges=wireframe,
            line_width=line_width,
            smooth_shading=smooth,
            texture=texture,
            show_scalar_bar=show_colorbar,
            opacity=opacity,
        )

    if show_plot:
        pl.show()
    else:
        return pl


def plot_p2p(
    mesh1,
    mesh2,
    p2p,
    pretty=True,
    n_colors=-1,
    param=[-2, -1, 3],
    uv1=None,
    texture=None,
    link_view=True,
    smooth=True,
):
    """
    Plot point-to-point correspondences between two meshes

    Parameters
    ------------------------------
    mesh1 : pyFM.TriMesh - source mesh
    mesh2 : pyFM.TriMesh - target mesh
    p2p   : np.ndarray - (n2,2) - point-to-point correspondences
    Output
    ------------------------------
    Viewer - meshplot.Viewer class
    """

    # plot the correspondences
    cmap1 = vert2rgb(mesh1.vertices, pretty=pretty, param=param, n_colors=n_colors)

    if uv1 is None:
        uv2 = None
    else:
        cmap1 = None

    if p2p.ndim == 1:
        if cmap1 is not None:
            cmap2 = cmap1[p2p]
        else:
            cmap2 = None
        if uv1 is not None:
            uv2 = uv1[p2p]
    else:
        assert p2p.shape == (
            mesh2.n_vertices,
            mesh1.n_vertices,
        ), "Pb with p2p dimension"
        if cmap1 is not None:
            cmap2 = p2p @ cmap1
        else:
            cmap2 = None
        if uv1 is not None:
            uv2 = p2p @ uv1

    # print(cmap1, uv1 is None, uv2 is None)

    # create a multiplot
    pl = pv.Plotter(shape=(1, 2))

    # plot the first mesh
    pl.subplot(0, 0)
    plot(
        mesh1,
        cmap=cmap1,
        uv=uv1,
        texture=texture,
        smooth=smooth,
        pl=pl,
        return_plot=True,
    )

    # plot the second mesh
    pl.subplot(0, 1)
    plot(
        mesh2,
        cmap=cmap2,
        uv=uv2,
        texture=texture,
        smooth=smooth,
        pl=pl,
        return_plot=True,
    )

    if link_view:
        pl.link_views()

    pl.show()
