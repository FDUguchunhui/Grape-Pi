# Predict protein detection using protein-protein-interaction by semi-supervised graph neural networks for shotgun proteomics

## Introduction


## Usage

### Data preparation

### run






## Installation

PyG is available for Python 3.7 to Python 3.10.

### Anaconda

You can now install PyG via [Anaconda](https://anaconda.org/pyg/pyg) for all major OS/PyTorch/CUDA combinations 🤗
If you have not yet installed PyTorch, install it via `conda` as described in the [official PyTorch documentation](https://pytorch.org/get-started/locally/).
Given that you have PyTorch installed (`>=1.8.0`), simply run

```
conda install pyg -c pyg
```

### Pip Wheels

We alternatively provide pip wheels for all major OS/PyTorch/CUDA combinations, see [here](https://data.pyg.org/whl).

#### PyTorch 1.13

To install the binaries for PyTorch 1.13.0, simply run

```
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html
pip install torch_geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu116`, or `cu117` depending on your PyTorch installation.

|             | `cpu` | `cu116` | `cu117` |
|-------------|-------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      |
| **macOS**   | ✅    |         |         |

For additional but optional functionality, run

```
pip install torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html
```

#### PyTorch 1.12

To install the binaries for PyTorch 1.12.0, simply run

```
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
pip install torch_geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu116` depending on your PyTorch installation.

|             | `cpu` | `cu102` | `cu113` | `cu116` |
|-------------|-------|---------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      | ✅      |
| **Windows** | ✅    |         | ✅      | ✅      |
| **macOS**   | ✅    |         |         |         |

For additional but optional functionality, run

```
pip install torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
```

**Note:** Binaries of older versions are also provided for PyTorch 1.4.0, PyTorch 1.5.0, PyTorch 1.6.0, PyTorch 1.7.0/1.7.1, PyTorch 1.8.0/1.8.1, PyTorch 1.9.0, PyTorch 1.10.0/1.10.1/1.10.2, and PyTorch 1.11.0 (following the same procedure).
For older versions, you might need to explicitly specify the latest supported version number or install via `pip install --no-index` in order to prevent a manual installation from source.
You can look up the latest supported version number [here](https://data.pyg.org/whl).

### Nightly and Master

In case you want to experiment with the latest PyG features which are not fully released yet, ensure that `pyg_lib`, `torch_scatter` and `torch_sparse` are installed by [following the steps mentioned above](#pip-wheels), and install either the **nightly version** of PyG via

```
pip install pyg-nightly
```

or install PyG **from master** via

```
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

## Cite

Please cite [our paper](https://arxiv.org/abs/1903.02428) (and the respective papers of the methods used) if you use this code in your own work:

```
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
```

Feel free to [email us](mailto:matthias.fey@tu-dortmund.de) if you wish your work to be listed in the [external resources](https://pytorch-geometric.readthedocs.io/en/latest/external/resources.html).
If you notice anything unexpected, please open an [issue](https://github.com/pyg-team/pytorch_geometric/issues) and let us know.
If you have any questions or are missing a specific feature, feel free [to discuss them with us](https://github.com/pyg-team/pytorch_geometric/discussions).
We are motivated to constantly make PyG even better.
# graphgym
