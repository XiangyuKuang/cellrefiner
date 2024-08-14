# Reconstruct single-cell resolution from spatial transcriptomics with CellRefiner

Single-cell RNA sequencing (scRNA-seq) technologies profile the transcriptome of individual cells but lack the spatial context necessary for deeper understanding of interactions like cell-cell communications. On the other hand, most current spatial transcriptomic (ST) technologies lack cellular resolution, limiting their utility for downstream analysis. Here we present CellRefiner, a physical model-based method that uses a scRNA-seq dataset with a paired ST dataset to predict single-cell resolution ST data. CellRefiner models cells as particles connected by forces, and then optimizes cell locations with spatial proximity constraints, gene expression similarity, and ligand-receptor interactions between cells. We comprehensively benchmark CellRefiner over a variety of simulated and real datasets using Visium, MERFISH, seqFISH+, Slide-seqV2, and STARmap datasets to demonstrate its accuracy, robustness, and ability to recover spatial patterns of cells. We also demonstrate its utility for improving spatially dependent analysis over the original ST data for contact-based cell-cell communication on mouse cortex and lymph node tissues. Our results show CellRefiner is capable of reconstructing single-cell resolution from non-single-cell resolution ST data, allowing  downstream analysis that requires individual-cell resolution and spatial information.

# Installation
We recommend using the Anaconda Python distribution and creating an isolated environment to avoid conflicts with other packages. To create a virtual environment, run the following script in the command line:

```conda create -n cellrefiner_env python=3.7```

After creating the environment, activate it:

```conda activate cellrefiner_env```

Now install the CellRefiner package from github via the command line:

```pip install git+https://github.com/echang992000/cellrefiner.git```

If the installation is not successful, try installing the required packages in requirements.txt:

```pip install -r requirements.txt```

# Useage

For a quick tutorial, please see the jupyter notebook and associated data in the tutorial folder.

# Hardware Requirements

This requires only a standard computer with enough RAM to support the in-memory operations. Having a GPU is preferable but not necessary. Installation should take <10 minutes.

# Software Dependencies

Please see the requirements.txt for package versions. This package is supported for Windows and has been tested on Windows 10 and 11. Running the tutorial should take <2 min if using a GPU and affinity matrix provided.
