# Predict protein detection using protein-protein-interaction by semi-supervised graph neural networks for shotgun proteomics

## Introduction
Protein-protein-interaction-Graph-neural-network (PPI-GNN) is a deep learning framework for predict protein existence based on
protein feature generated from Mass spectrometry (MS) instrument/analysis software and protein-protein-interaction (PPI)
network.

The main idea is to  promote proteins with medium evidence but are supported by protein-protein-interaction information
as existent. Unlike traditional network analysis, PPI information is used with strong assumptions and restricted to
specific sub-network structures (e.g. clique), PPI-GNN model is a fully data-driven model and can be much more versatile. 

The framework was built on top of torch-geometry and pytorch. GraphyGym is used for allowing model training with a
variety of hyperparameter combination with less computation and time using the idea of random-design-space searching. 

## installation
Due to special training infrastructure requirement, we modified part of code in the torch_geometric package. Therefore,
to run the code, the customized torch_geometric package must be used install of an official version of torch_geometric
package. We recommend use `conda` to create a virtual environment to avoid unexpected errors when running code that
requires a official version of torch_geometric package.

```
conda env create graph-pi
conda activate graph-pi
```

Then navigate to the root directory of the download folder
```
python setup.py install
```

## Usage

```
python main.py --cfg configs/example.yaml --repeat 3
```

### Data preparation
Graph-pi training framework was based on graphgym. Please check the  for details.


The PPI-GNN framework features a built-in module for easily load raw protein and PPI data into torch.geometry.dataset 
format that can be used for training model. The only things needed is to provide a root directory for where to
find the raw file, and the name of the raw protein data file and raw PPI data file in the configure file.

```ymal
dataset:
  format: PyG
  name: protein
  dir: ./data/single
  numeric_params:
    - protein probability
    - percent coverage
    - tot indep spectra
    - percent share of spectrum ids
  task: node
  task_type: classification
  transductive: false
  transform: none
  encoder: false
  node_encoder: false
  edge_encoder: false
```

The root folder should look like this, with a fold named "raw" and the "protein_filename" and "interaction_filename"
should be inside the raw folder.

```dir
root
    raw
        protein_filename
        interaction_filename
```

After the programming first running, a new "processed" folder will create under the root directory which stores the
converted torch_geometry.data.dataset format and additional processed files. This allows a one-time processing and the 
next time data the same data is used, the processed file will be loaded directly to save time.


### run


## Additional example

### modeling with metabolomics




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
