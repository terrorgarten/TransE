## A TransE graph model implementation using PyTorch and Torch Geometric libs.

Reference: https://proceedings.neurips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf
---
### Installation guide

- ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```
- ```pip install torch-geometric```
- ```pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html```

Please note that this install is configured for Linux and CPU training.
To configure your library downloads, please see the following links:
- https://pytorch.org/get-started/locally/
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
---
### Run
python3 ./transe.py
