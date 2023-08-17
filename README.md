# Graph neural network using Protein-protein-interaction for Enhancing Protein Identification (Grape-Pi)
![img_2.png](img_2.png)
## Introduction
GRAph-neural-network using Protein-protein-interaction for Enhancing Protein Identification (Grape-Pi) is a deep 
learning framework for predict protein existence based on
protein feature generated from Mass spectrometry (MS) instrument/analysis software and protein-protein-interaction (PPI)
network.

The main idea is to  promote proteins with medium evidence but are supported by protein-protein-interaction information
as existent. Unlike traditional network analysis, PPI information is used with strong assumptions and restricted to
specific sub-network structures (e.g. clique), Grape-Pi model is a fully data-driven model and can be much more versatile. 

--------------------------------------------------------------------------------

The contribution of Grape-Pi comes in threefold. First, we developed a dataloader module designed for loading MS protein
data and protein-protein-interaction data into dataset format that can be readily used by torch-geometry.
Second, we customized the graphgym module for the purpose of unsupervised learning in proteomics data. Third, we explored
the design space and discussed caveats for training such a model for the best performance.


## installation

It consists of various methods for deep learning on graphs and other irregular structures, also known as *[geometric deep learning](http://geometricdeeplearning.com/)*, from a variety of published papers.
In addition, it consists of easy-to-use mini-batch loaders for operating on many small and single giant graphs, [multi GPU-support](https://github.com/pyg-team/pytorch_geometric/tree/master/examples/multi_gpu), [`DataPipe` support](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/datapipe.py), distributed graph learning via [Quiver](https://github.com/pyg-team/pytorch_geometric/tree/master/examples/quiver), a large number of common benchmark datasets (based on simple interfaces to create your own), the [GraphGym](https://pytorch-geometric.readthedocs.io/en/latest/advanced/graphgym.html) experiment manager, and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.
[Click here to join our Slack community!][slack-url]

Please refer to https://conda.io/projects/conda/en/latest/user-guide/install/index.html for how to install a miniconda
or anaconda in your local machine.

To create a virtual environment, for example, if using conda
```angular2html
conda create -n [Name-of-the-Virtual-Environment] python=3.7
conda activate [Name-of-the-Virtual-Environment]
```
Replace `[Name-of-the-Virtual-Environment]` with your preferred name.

The original torch-geometry support python 3.7-3.11 with pytorch 1.3.0-2.0.0. For illustration
purpose, we use python=3.7 and pytorch=1.13.0 here. 
For using and debugging with other python and pytorch version, please refer to https://github.com/pyg-team/pytorch_geometric for details

In case you have accidentally an official version of torch-geometric, run
```angular2html
pip uninstall torch-geometric
pip uninstall torch-geometric  # run this command twice
```

### Install pytorch
For mac
```angular2html
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
```
For Linux and Windows:
```angular2html
# CUDA 11.6
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# CUDA 11.7
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# CPU Only
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch
```

### Follow the installation instructions to install additional libraries to using Grape-Pi:
torch-scatter, torch-sparse, torch-cluster and torch-spline-conv (if you haven't already):
```angular2html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv 
-f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
If you are using pytorch=1.13.0 cpu only version, run
```angular2html
pip install torch_scatter torch_sparse torch_cluster 
torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
```

## If you follow the instruction above and use new virtual environment, you don't need to run this code chunk.
In case you have installed pyg-lib before from somewhere and run into unexpected error, try
uninstall pyg-lib to avoid that it conflicts with torch_sparse library.
```angular2html
pip uninstall pyg-lib
```


### Clone a copy of Grape-Pi from source and navigate to the root directory of the download folder
```angular2html
git clone https://github.com/FDUguchunhui/Grape-Pi.git
cd Grape-Pi
```

* The PyG **engine** utilizes the powerful PyTorch deep learning framework, as well as additions of efficient CUDA libraries for operating on sparse data, *e.g.*, [`pyg-lib`](https://github.com/pyg-team/pyg-lib), [`torch_scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch_sparse`](https://github.com/rusty1s/pytorch_sparse) and [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster).
* The PyG **storage** handles data processing, transformation and loading pipelines. It is capable of handling and processing large-scale graph datasets, and provides effective solutions for heterogeneous graphs. It further provides a variety of sampling solutions, which enable training of GNNs on large-scale graphs.
* The PyG **operators** bundle essential functionalities for implementing Graph Neural Networks. PyG supports important GNN building blocks that can be combined and applied to various parts of a GNN model, ensuring rich flexibility of GNN design.
* Finally, PyG provides an abundant set of GNN **models**, and examples that showcase GNN models on standard graph benchmarks. Thanks to its flexibility, users can easily build and modify custom GNN models to fit their specific needs.


## Data preparation
The Graph-Pi training framework was based on graphgym. Please check `Design Space for Graph Neural Networks 
https://arxiv.org/abs/2011.08843` for details.

The Graph-Pi framework features a built-in module for easily load raw protein and PPI data into torch.geometry.dataset 
format that can be used for training model. The only things needed is to provide a path to the dataset
folder.

The dataset folder structure should look like this, with a sub-folder named `raw`
inside the `raw` sub-folder, it should have three sub-folders: protein, interaction, reference.

The `protein` folder must contain only one csv or tsv file: the first column must be protein ID and other columns
contains additional protein features.

The `interaction` folder must contain only one csv or tsv file: the first and second columns must be same type protein ID,
refer to the two protein interactors, other columns can be additional features for the interaction relationship.

The `reference` folder must contain two txt files. 
`positive.txt` and `negative.txt` are two simple txt file with a protein ID in each line.
The protein in `positive.txt` will be used to create positive label in the final processed dataset
The protein in `negative.txt` will be used to create negative label in the final processed dataset.
Protein that is not found in either `positive.txt` and `negative.txt` will be treated as unlabelled.
Only labelled proteins will be used for calculating loss and then backward propagation to update model.

You could find such an example of such dataset `Graph-Pi/graphgym/data/yeast`

* **[Top-K Pooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.TopKPooling.html)** from Gao and Ji: [Graph U-Nets](https://arxiv.org/abs/1905.05178) (ICML 2019), Cangea *et al.*: [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287) (NeurIPS-W 2018) and Knyazev *et al.*: [Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850) (ICLR-W 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_topk_pool.py)]
* **[DiffPool](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.dense_diff_pool.html)** from Ying *et al.*: [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804) (NeurIPS 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py)]

<details>
<summary><b>Expand to see all implemented pooling layers...</b></summary>

* **[Attentional Aggregation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.AttentionalAggregation.html)** from Li *et al.*: [Graph Matching Networks for Learning the Similarity of Graph Structured Objects](https://arxiv.org/abs/1904.12787) (ICML 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/global_attention.py)]
* **[Set2Set](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.Set2Set.html)** from Vinyals *et al.*: [Order Matters: Sequence to Sequence for Sets](https://arxiv.org/abs/1511.06391) (ICLR 2016) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/set2set.py)]
* **[Sort Aggregation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.SortAggregation.html)** from Zhang *et al.*: [An End-to-End Deep Learning Architecture for Graph Classification](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf) (AAAI 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/sort_pool.py)]
* **[MinCut Pooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.dense_mincut_pool.html)** from Bianchi *et al.*: [Spectral Clustering with Graph Neural Networks for Graph Pooling](https://arxiv.org/abs/1907.00481) (ICML 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_mincut_pool.py)]
* **[DMoN Pooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.DMoNPooling.html)** from Tsitsulin *et al.*: [Graph Clustering with Graph Neural Networks](https://arxiv.org/abs/2006.16904) (CoRR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_dmon_pool.py)]
* **[Graclus Pooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.graclus.html)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_graclus.py)]
* **[Voxel Grid Pooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.voxel_grid.html)** from, *e.g.*, Simonovsky and Komodakis: [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_voxel_grid.py)]
* **[SAG Pooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.SAGPooling.html)** from Lee *et al.*: [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082) (ICML 2019) and Knyazev *et al.*: [Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850) (ICLR-W 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/sag_pool.py)]
* **[Edge Pooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.EdgePooling.html)** from Diehl *et al.*: [Towards Graph Pooling by Edge Contraction](https://graphreason.github.io/papers/17.pdf) (ICML-W 2019) and Diehl: [Edge Contraction Pooling for Graph Neural Networks](https://arxiv.org/abs/1905.10990) (CoRR 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/edge_pool.py)]
* **[ASAPooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.ASAPooling.html)** from Ranjan *et al.*: [ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations](https://arxiv.org/abs/1911.07979) (AAAI 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/asap.py)]
* **[PANPooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.PANPooling.html)** from Ma *et al.*: [Path Integral Based Convolution and Pooling for Graph Neural Networks](https://arxiv.org/abs/2006.16811) (NeurIPS 2020)
* **[MemPooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.MemPooling.html)** from Khasahmadi *et al.*: [Memory-Based Graph Networks](https://arxiv.org/abs/2002.09518) (ICLR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mem_pool.py)]
* **[Graph Multiset Transformer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.GraphMultisetTransformer.html)** from Baek *et al.*: [Accurate Learning of Graph Representations with Graph Multiset Pooling](https://arxiv.org/abs/2102.11533) (ICLR 2021) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_gmt.py)]
* **[Equilibrium Aggregation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.EquilibriumAggregation.html)** from Bartunov *et al.*: [](https://arxiv.org/abs/2202.12795) (UAI 2022) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/equilibrium_median.py)]
</details>

**GNN models:**
Our supported GNN models incorporate multiple message passing layers, and users can directly use these pre-defined models to make predictions on graphs.
Unlike simple stacking of GNN layers, these models could involve pre-processing, additional learnable parameters, skip connections, graph coarsening, etc.

* **[SchNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.SchNet.html)** from Schütt *et al.*: [SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions](https://arxiv.org/abs/1706.08566) (NIPS 2017) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_pretrained_schnet.py)]
* **[DimeNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DimeNet.html)** and **[DimeNetPlusPlus](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DimeNetPlusPlus.html)** from Klicpera *et al.*: [Directional Message Passing for Molecular Graphs](https://arxiv.org/abs/2003.03123) (ICLR 2020) and [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/abs/2011.14115) (NeurIPS-W 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_pretrained_dimenet.py)]
* **[Node2Vec](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.Node2Vec.html)** from Grover and Leskovec: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) (KDD 2016) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py)]
* **[Deep Graph Infomax](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGraphInfomax.html)** from Veličković *et al.*: [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) (ICLR 2019) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_transductive.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_inductive.py)]
* **Deep Multiplex Graph Infomax** from Park *et al.*: [Unsupervised Attributed Multiplex Network Embedding](https://arxiv.org/abs/1911.06750) (AAAI 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/dmgi_unsup.py)]
* **[Masked Label Prediction](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.MaskLabel.html)** from Shi *et al.*: [Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification](https://arxiv.org/abs/2009.03509) (CoRR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/unimp_arxiv.py)]

<details>
<summary><b>Expand to see all implemented GNN models...</b></summary>

* **[Jumping Knowledge](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.JumpingKnowledge.html)** from Xu *et al.*: [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536) (ICML 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gin.py#L54-L106)]
* A **[MetaLayer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.MetaLayer.html)** for building any kind of graph network similar to the [TensorFlow Graph Nets library](https://github.com/deepmind/graph_nets) from Battaglia *et al.*: [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) (CoRR 2018)
* **[MetaPath2Vec](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.MetaPath2Vec.html)** from Dong *et al.*: [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) (KDD 2017) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/metapath2vec.py)]
* All variants of **[Graph Autoencoders](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAE.html)** and **[Variational Autoencoders](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.VGAE.html)** from:
    * [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) from Kipf and Welling (NIPS-W 2016) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py)]
    * [Adversarially Regularized Graph Autoencoder for Graph Embedding](https://arxiv.org/abs/1802.04407) from Pan *et al.* (IJCAI 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/argva_node_clustering.py)]
    * [Simple and Effective Graph Autoencoders with One-Hop Linear Models](https://arxiv.org/abs/2001.07614) from Salha *et al.* (ECML 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py)]
* **[SEAL](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/seal_link_pred.py)** from Zhang and Chen: [Link Prediction Based on Graph Neural Networks](https://arxiv.org/pdf/1802.09691.pdf) (NeurIPS 2018) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/seal_link_pred.py)]
* **[RENet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.RENet.html)** from Jin *et al.*: [Recurrent Event Network for Reasoning over Temporal Knowledge Graphs](https://arxiv.org/abs/1904.05530) (ICLR-W 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/renet.py)]
* **[GraphUNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphUNet.html)** from Gao and Ji: [Graph U-Nets](https://arxiv.org/abs/1905.05178) (ICML 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_unet.py)]
* **[AttentiveFP](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.AttentiveFP.html)** from Xiong *et al.*: [Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959) (J. Med. Chem. 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/attentive_fp.py)]
* **[DeepGCN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGCNLayer.html)** and the **[GENConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GENConv.html)** from Li *et al.*: [DeepGCNs: Can GCNs Go as Deep as CNNs?](https://arxiv.org/abs/1904.03751) (ICCV 2019) and [DeeperGCN: All You Need to Train Deeper GCNs](https://arxiv.org/abs/2006.07739) (CoRR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py)]
* **[RECT](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.RECT_L.html)** from Wang *et al.*: [Network Embedding with Completely-imbalanced Labels](https://ieeexplore.ieee.org/document/8979355) (TKDE 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rect.py)]
* **[GNNExplainer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.explain.algorithm.GNNExplainer.html)** from Ying *et al.*: [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) (NeurIPS 2019) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gnn_explainer.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gnn_explainer_ba_shapes.py), [**Example3**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gnn_explainer_link_pred.py)]
* **Graph-less Neural Networks** from Zhang *et al.*: [Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation](https://arxiv.org/abs/2110.08727) (CoRR 2021) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/glnn.py)]
* **[LINKX](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.LINKX.html)** from Lim *et al.*: [Large Scale Learning on Non-Homophilous Graphs:
New Benchmarks and Strong Simple Methods](https://arxiv.org/abs/2110.14446) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/linkx.py)]
* **[RevGNN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GroupAddRev.html)** from Li *et al.*: [Training Graph Neural with 1000 Layers](https://arxiv.org/abs/2106.07476) (ICML 2021) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rev_gnn.py)]
* **[TransE](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.TransE.html)** from Bordes *et al.*: [Translating Embeddings for Modeling Multi-Relational Data](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf) (NIPS 2013) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/transe_fb15k_237.py)]
</details>

**GNN operators and utilities:**
PyG comes with a rich set of neural network operators that are commonly used in many GNN models.
They follow an extensible design: It is easy to apply these operators and graph utilities to existing GNN layers and models to further enhance model performance.

* **[DropEdge](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.dropout_edge)** from Rong *et al.*: [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://openreview.net/forum?id=Hkx1qkrKPr) (ICLR 2020)
* **[DropNode](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.dropout_node)**, **[MaskFeature](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.mask_feature)** and **[AddRandomEdge](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.add_random_edge)** from You *et al.*: [Graph Contrastive Learning with Augmentations](https://arxiv.org/abs/2010.13902) (NeurIPS 2020)
* **[DropPath](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.dropout_path)** from Li *et al.*: [MaskGAE: Masked Graph Modeling Meets Graph Autoencoders](https://arxiv.org/abs/2205.10053) (arXiv 2022)
* **[ShuffleNode](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.shuffle_node)** from Veličković *et al.*: [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) (ICLR 2019)
* **[GraphNorm](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.norm.GraphNorm.html)** from Cai *et al.*: [GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training](https://proceedings.mlr.press/v139/cai21e.html) (ICML 2021)
* **[GDC](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.GDC.html)** from Klicpera *et al.*: [Diffusion Improves Graph Learning](https://arxiv.org/abs/1911.05485) (NeurIPS 2019) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py)]

<details>
<summary><b>Expand to see all implemented GNN operators and utilities...</b></summary>

* **[GraphSizeNorm](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.norm.GraphSizeNorm.html)** from Dwivedi *et al.*: [Benchmarking Graph Neural Networks](https://arxiv.org/abs/2003.00982) (CoRR 2020)
* **[PairNorm](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.norm.PairNorm.html)** from Zhao and Akoglu: [PairNorm: Tackling Oversmoothing in GNNs](https://arxiv.org/abs/1909.12223) (ICLR 2020)
* **[MeanSubtractionNorm](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.norm.MeanSubtractionNorm.html)** from Yang *et al.*: [Revisiting "Over-smoothing" in Deep GCNs](https://arxiv.org/abs/2003.13663) (CoRR 2020)
* **[DiffGroupNorm](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.norm.DiffGroupNorm.html)** from Zhou *et al.*: [Towards Deeper Graph Neural Networks with Differentiable Group Normalization](https://arxiv.org/abs/2006.06972) (NeurIPS 2020)
* **[Tree Decomposition](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.tree_decomposition)** from Jin *et al.*: [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/abs/1802.04364) (ICML 2018)
* **[TGN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.TGNMemory.html)** from Rossi *et al.*: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637) (GRL+ 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py)]
* **[Weisfeiler Lehman Operator](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.WLConv.html)** from Weisfeiler and Lehman: [A Reduction of a Graph to a Canonical Form and an Algebra Arising During this Reduction](https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf) (Nauchno-Technicheskaya Informatsia 1968) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/wl_kernel.py)]
* **[Continuous Weisfeiler Lehman Operator](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.WLConvContinuous.html)** from Togninalli *et al.*: [Wasserstein Weisfeiler-Lehman Graph Kernels](https://arxiv.org/abs/1906.01277) (NeurIPS 2019)
* **[Label Propagation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.LabelPropagation.html)** from Zhu and Ghahramani: [Learning from Labeled and Unlabeled Data with Label Propagation](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf) (CMU-CALD 2002) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/label_prop.py)]
* **[Local Degree Profile](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.transforms.LocalDegreeProfile)** from Cai and Wang: [A Simple yet Effective Baseline for Non-attribute Graph Classification](https://arxiv.org/abs/1811.03508) (CoRR 2018)
* **[CorrectAndSmooth](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.CorrectAndSmooth.html)** from Huang *et al.*: [Combining Label Propagation And Simple Models Out-performs Graph Neural Networks](https://arxiv.org/abs/2010.13993) (CoRR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/correct_and_smooth.py)]
* **[Gini](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.functional.gini.html)** and **[BRO](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.functional.bro.html)** regularization from Henderson *et al.*: [Improving Molecular Graph Neural Network Explainability with Orthonormalization and Induced Sparsity](https://arxiv.org/abs/2105.04854) (ICML 2021)
* **[RootedEgoNets](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.transforms.RootedEgoNets)** and **[RootedRWSubgraph](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.transforms.RootedRWSubgraph)** from Zhao *et al.*: [From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness](https://arxiv.org/abs/2110.03753) (ICLR 2022)
* **[FeaturePropagation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.transforms.FeaturePropagation)** from Rossi *et al.*: [On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs with Missing Node Features](https://arxiv.org/abs/2111.12128) (CoRR 2021)
</details>

**Scalable GNNs:**
PyG supports the implementation of Graph Neural Networks that can scale to large-scale graphs.
Such application is challenging since the entire graph, its associated features and the GNN parameters cannot fit into GPU memory.
Many state-of-the-art scalability approaches tackle this challenge by sampling neighborhoods for mini-batch training, graph clustering and partitioning, or by using simplified GNN models.
These approaches have been implemented in PyG, and can benefit from the above GNN layers, operators and models.

* **[NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py), [**Example3**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_gat.py), [**Example4**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py)]
* **[ClusterGCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.ClusterLoader)** from Chiang *et al.*: [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953) (KDD 2019) [[**Example1**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py), [**Example2**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cluster_gcn_ppi.py)]
* **[GraphSAINT](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.GraphSAINTSampler)** from Zeng *et al.*: [GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/abs/1907.04931) (ICLR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_saint.py)]

<details>
<summary><b>Expand to see all implemented scalable GNNs...</b></summary>

* **[ShaDow](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.ShaDowKHopSampler)** from Zeng *et al.*: [Decoupling the Depth and Scope of Graph Neural Networks](https://arxiv.org/abs/2201.07858) (NeurIPS 2021) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/shadow.py)]
* **[SIGN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.SIGN.html)** from Rossi *et al.*: [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198) (CoRR 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/sign.py)]
* **[HGTLoader](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.loader.HGTLoader.html)** from Hu *et al.*: [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) (WWW 2020) [[**Example**](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py)]
</details>

## Installation

PyG is available for Python 3.7 to Python 3.10.

### Anaconda

You can now install PyG via [Anaconda](https://anaconda.org/pyg/pyg) for all major OS/PyTorch/CUDA combinations 🤗
If you have not yet installed PyTorch, install it via `conda` as described in the [official PyTorch documentation](https://pytorch.org/get-started/locally/).
Given that you have PyTorch installed (`>=1.8.0`), simply run

```

## Set up model configuration
Before you can train a model based on the provided data, a configuration file is needed to for some key information:
where to find the data, how to parse the data, what features to use, model structure, what loss function to use, how to 
update model weights, etc. 

You could find such an example of such config `Graph-Pi/graphgym/config/protein/protein-yeast-graphsage.yaml`

```ymal
out_dir: results # output will be put in graphgym/results
metric_best: auc # best model will be selected based on what
dataset: # need to delete the `processed` folder if change args under dataset category
  name: protein # specify what dataloader used
  dir: ./data/yeast
  numeric_params: # currently, only numeric protein features are allowed
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
  split: [0.8, 0.1, 0.1]
share:
  dim_in: -1      # `-1` means the input dim will be determined automatically
  dim_out: 2
train:
  loss_pos_weight: -1.0  # `-1` means sample will be automatically rebalanced using sklearn.utils.class_weight.compute_class_weight
  batch_size: 128 # number of nodes used in each iteration
  eval_period: 10 # evaluate model in val and test dataset every 10 epochs
  ckpt_period: 10 # save model after every 10 epochs training
  ckpt_clean: false
  sampler: neighbor # use `neighor` for sampler for layer_type=sageconv
  neighbor_sizes: # the number of nodes sampled in each neighor level
  - 100
  - 50
  - 25
  - 10
  epoch_resume: -1
model:
  type: gnn
  loss_fun: binary_cross_entropy_with_weight
gnn:
  layers_pre_mp: 1
  layers_mp: 2
  layers_post_mp: 1
  dim_inner: 8 # 2 * dim_in
  layer_type: sageconv
  stage_type: stack # other available: skipsum, skipconcat
  batchnorm: false
  act: relu
  dropout: 0.0
  normalize_adj: false
  head: protein
optim:
  optimizer: adam
  base_lr: 0.001
  weight_decay: 5e-4  # L2 penalty
  max_epoch: 100
  scheduler: none
```

### Pip Wheels

We alternatively provide pip wheels for all major OS/PyTorch/CUDA combinations, see [here](https://data.pyg.org/whl).

#### PyTorch 1.13

To install the binaries for PyTorch 1.13.0, simply run

```
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html
pip install torch_geometric
```

During the programming first running, a new "processed" folder will create under the provided data folder which stores the
converted torch_geometry.data.dataset format and additional processed files. This allows a one-time processing and the 
next time data the same data is used, the processed file will be loaded directly to save time.

**Caution**: In case you have updated the raw files, you need to manually deleted the entire `processed` folder to let the program
rebuild processed data from modified raw files.

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

CONFIG=${CONFIG:-protein-yeast-gcnconv}
GRID=${GRID:-protein-yeast-gcnconv}
REPEAT=${REPEAT:-3}
MAX_JOBS=${MAX_JOBS:-1}
SLEEP=${SLEEP:-0}
MAIN=${MAIN:-main}

In case you want to experiment with the latest PyG features which are not fully released yet, ensure that `pyg_lib`, `torch_scatter` and `torch_sparse` are installed by [following the steps mentioned above](#pip-wheels), and install either the **nightly version** of PyG via

```
The instruction above only aim to provide a start point for user to check how we did our experiment.
Please refer to https://github.com/snap-stanford/GraphGym for more details about how to config a batch experiment.

### run batch experiment
```angular2html
bash run_batch_yeast_gcnconv.sh
```


### Aggregate results
Run `bash run_batch_yeast_gcnconv.sh` should automatically aggregate batch experiment result into `agg` folder.
However, in case it is not generated automatically, you can manually aggregate the results by run
```angular2html 
python agg_batch.py --dir results/protein-yeast-graphsage_grid_protein-yeast-graphsage
```


## Post-training analysis





## Rebuild the best model based on analysis result from batch experiment for protein prediction application 
```angular2html

```

### ROCm Wheels

The external [`pyg-rocm-build` repository](https://github.com/Looong01/pyg-rocm-build) provides wheels and detailed instructions on how to install PyG for ROCm.
If you have any questions about it, please open an issue [here](https://github.com/Looong01/pyg-rocm-build/issues).

## Cite
Please cite the following papers if you use this code in your own work::
[Fast Graph Representation Learning with PyTorch Geometric


[Fast Graph Representation Learning with PyTorch Geometric
](https://arxiv.org/abs/1903.02428) 

```
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
```


Common issues:

1. If you have the following problem during processing the `raw` data into `processed` data
```angular2html
utf8' codec can't decode byte 0x80 in position 3131: invalid start byte
```
This is caused by a hidden `.DS_Store` file created by the Mac OS system
Use terminal enter the `protein` folder under the `raw` folder
```angular2html
ls -a # check if there is a .DS_Store file
rm .DS_Store # remove the file
rm -r ../../processed # remove the ill-created `processed` data
```


2. 