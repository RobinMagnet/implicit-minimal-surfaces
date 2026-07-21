# Implicit Minimal Surfaces for Bijective Correspondences
![Toucan](https://www.yousufsoliman.com/img/implicit-minimal-surface-banner.png)

Official Python implementation of the paper:

[_Implicit Minimal Surfaces for Bijective Correspondences_](https://robinmagnet.github.io/SIGGRAPH26_MinimalSurfaces/MinimalSurfaces.pdf)\
Etienne Corman, Yousuf Soliman, Robin Magnet, and Mark Gillespie\
_ACM Transactions on Graphics_ 45 (4) | **SIGGRAPH 2026**

The algorithm computes an implicit representation of bijective correspondences of genus zero surfaces. Our implicit representation is based on discrete complex line bundles over products of triangulated surfaces.


## Installation

```
git clone --recurse-submodules https://github.com/RobinMagnet/implicit-minimal-surfaces
cd implicit-minimal-surfaces
pip install numpy scipy potpourri3d einops pykeops
```

If you cloned without `--recurse-submodules`, fetch the [`ScalableDenseMaps`](https://github.com/RobinMagnet/ScalableDenseMaps) dependency with

```
git submodule update --init --recursive
```

`ScalableDenseMaps` is not on PyPI, so it is added to the path rather than installed:

```python
import os, sys
sys.path.append(os.path.abspath("./ScalableDenseMaps/"))
```

Running `example_notebook.ipynb` additionally requires [`pyvista`](https://robinmagnet.github.io/SIGGRAPH26_MinimalSurfaces/MinimalSurfaces.pdf), but you can use your own visualization functions.

## Usage

An example script is provided in [`example_notebook.ipynb`](./example_notebook.ipynb)


The whole method is abstracted in a `Pipeline` class. The shortest complete run is

```python
from diffops import Pipeline

P21 = Pipeline.from_files("data/teddy/A.obj", "data/teddy/B.obj").run()
```

P21 is a `densemaps` precise map (vertex to barycentric coordinates). With `direction="21"` (the default) it sends the vertices of `B` onto the surface of `A`, so `P21 @ mesh_A.vertices` gives the image of every vertex of `B`. Using `direction="12"` gives the other direction.

### Step by step

`run` simply chains the four steps of the paper, each of which can be called on its own:

```python
pipe = Pipeline.from_files("data/teddy/A.obj", "data/teddy/B.obj")

pipe.set_initial_map_nn()            # input map (here: ambient nearest neighbor)
pipe.compute_connections()           # (1) Build CLB structure
section = pipe.initialize_section()  # (2) initial section, from the input map
pipe.build_operators()               #     product-space Laplacian and mass operators
section = pipe.optimize(section, lam=1e2)   # (3) Ginzburg-Landau minimization
P21 = pipe.extract_map(section, direction="21")    # (4) Zero set extracion
```

An arbitrary input map can be supplied instead of the nearest-neighbor one with
`pipe.set_initial_map(P21, P12)`, where both arguments are `densemaps` precise maps.

### Landmarks

Point and curve landmarks are supported as soft constraints.

Landmarks are read from a pair of `.pinned` files: line _k_ of each file describes one landmark, as a whitespace-separated
list of vertex indices. A single index is a point landmark, and multiple indices describe curve
landmark. The  `from_files` method find these files automatically
when they are saved next to the meshes (`A.obj` / `A.pinned`). Else, they can be loaded explicitly:

```python
landmarks = Pipeline.read_pinned("data/teddy/A.pinned", "data/teddy/B.pinned")
P21 = pipe.run(landmarks=landmarks, sigma=0.7)
```

where `sigma` sets the width of the Gaussian wells. The pinning potential is:

```python
V = pipe.compute_pin_matrix(landmarks, sigma=0.7)
```

### Main Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `lam` | `1e2` | Regularization parameter. Can use a list of values for annealing.
| `sigma` | `1/sqrt(2)` | Width of the Gaussian landmark wells. |
| `n_iter` | `1000` | Maximum number of L-BFGS iterations. |
| `load_pinned` | `True` | Auto-discover `.pinned` landmark files next to the meshes. |

The Ginzburg-Landau parameter `lam` is the main parameter of the method: `lam=1e2`
generally produces high quality results, while examples requiring significant untangling
benefit from the annealing schedule above.

### Notes

- Inputs must be **closed, genus zero** manifold triangle meshes. This is not checked.
- The section requires $|V_A| \times |V_B|$ storage.
- The connection Laplacian and mass matrices are never assembled in the product space; they are applied as operators factored across the two surfaces.

## Alternative Implementations
- **C++:** https://github.com/yousufmsoliman/implicit-minimal-surfaces
- **MATLAB:** https://github.com/etcorman/implicit-minimal-surfaces

## Citation

If our work contributes to your academic work, please consider cite the following paper:

```bib
@article{Corman:2026:IMS,
  title={Implicit Minimal Surfaces for Bijective Correspondences},
  author={Corman, Etienne and Soliman, Yousuf and Magnet, Robin and Gillespie, Mark},
  journal={ACM Transactions on Graphics},
  volume={45},
  number={4},
  year={2026},
  publisher={ACM}
}
```
