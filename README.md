# DeepTernary


SE(3)-Equivariant Ternary Complex Prediction Towards Target Protein Degradation


---

## Update
- **2025-07-21**: Release all ternary pdb files and support automatic download of ideal ligands.
- **2025-05-25**: Since the journal doesn't allow including thanks to anonymous referees, we would like to express our gratitude to the reviewer 3 for providing the RDKit codes to rigidly fix PROTAC chemical handles during conformational searches.

## Overview

DeepTernary is a deep learning-based method for predicting structures of ternary complexes induced by PROTACs and molecule glues (MG(D)). To fulfill our purpose, we first constructed a large-scale dataset of over 20k ternary structures by identifying high-quality complexes from the Protein Data Bank (PDB) that include a small molecule and two proteins. Using this curated data, we trained DeepTernary, an SE(3)-equivariant graph neural network designed for ternary structure prediction. DeepTernary is able to output the conformation of the small molecule and the docking poses of the complex. We evaluated it on existing PROTACs and MG(D)s, achieving DockQ scores of 0.65 and 0.23 with average inference times of about 7 seconds and 1 second, respectively, on the PROTAC and MG(D) benchmarks. The ability of DeepTernary to generalize from a non-PROTAC/MG PDB dataset to PROTAC/MG ternary structures demonstrates its capability to extract common interaction patterns and principles governing ternary structure formation induced by small molecules rather than merely memorizing existing structures. Furthermore, the calculated buried surface area (BSA) based on our predicted PROTAC structures aligns well with existing literature, suggesting high degradation potential when BSA ranges between 1100 and 1500. 

![Introduction](data/introduction.webp)

## TernaryDB

The collected **TernaryDB** dataset can be downloaded [here](https://github.com/youqingxiaozhua/DeepTernary/releases/download/v1.0.1/TernaryDB.tar.gz).

The PROTAC, the `val.csv` and `test.csv` have three columns, they represent complex_id, unbound structure for p1 and p2, respectively.

The similarity clustered results for the training examples are also included in the corresponding folder named `train_clusters.json`.


File structure:

```
TernaryDB
├── pdbs
│   ├── 1A2Y_A_C_PO4
│   │   └── gt_complex.pdb
│   ...
├── MGD
│   ├── test.txt
│   ├── train_clusters.json
│   ├── train.txt
│   └── val.txt
└── PROTAC
    ├── test.csv
    ├── train_clusters.json
    ├── train.txt
    └── val.csv
```

Please place the `TernaryDB` folder in the `data` folder of this codebase, otherwise, you need to modify the `DATA_BASE` variable in `deepternary/models/path.py` to the correct path.

## Usage

### Install

#### Hardware requirements

DeepTernary support inferencing on a standard computer with or without GPUs.

#### Software requirements

##### OS Requirements

This codebase is supported for Linux, macOS, and Microsoft, as long as it could install PyTorch.

##### Python Dependencies

1. Please follow the PyTorch document to install PyTorch: https://pytorch.org/get-started/locally/.
    For example, for PyTorch 2.3.1 with CUDA 12.1, you can use the following command:
    
    ```bash
    pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    ```

2. Install the dependencies by: `pip install -r requirements.txt`

The experiments are conducted under following envrionment, but other versions should also work with this codebase. It typically takes 10 minues to install all the dependecies.

- Red Hat Enterprise Linux release 8.8
- Python==3.10.3
- PyTorch==2.3.1+cu121
- mmengine==0.10.3



### Config Files

The model is defined in the following config file, please use the corresponding config file for the task.

| Task   | Config File                    |
|--------|-------------------------------------------------------|
| PROTAC | [deepternary/configs/protac.py](deepternary/configs/protac.py) |
| MGD    | [deepternary/configs/glue.py](deepternary/configs/glue.py)       |


### Prediction

To perform evaluation, follow these steps:
1. Download the pre-trained checkpoint and PROTAC unbound structures from [this link](https://github.com/youqingxiaozhua/DeepTernary/releases/download/v1.0.1/output.zip).
2. Unzip the folder and move it to the root directory of this codebase.
3. Execute the following command to evaluate the model:


```Bash
# download the pre-trained checkpoints and unbound structures
wget https://github.com/youqingxiaozhua/DeepTernary/releases/download/v1.0.1/output.zip

# unzip data
unzip output.zip

# evaluate on the PROTAC test set
python predict_cpu.py output/checkpoints/PROTAC

# MGD need to process the test data, see `TernaryDB/MGD/test.txt`
# python predict_cpu.py output/checkpoints/MGD
```
Typically, the results for every test sample should be shown in 5 minues.

For PROTAC prediction with different PROTACs, there is a simple API in `predict_btk.py`:

```Python

data_dir = 'output/protac_new'
name = '8QU8_A_F_WYL'
cfg.save_dir = f'output/pred_{name}'
predict_one_unbound(
    name,
    lig_path=f'{data_dir}/{name}/ligand.pdb',
    p1_path=f'{data_dir}/{name}/unbound_protein1.pdb', p2_path=f'{data_dir}/{name}/unbound_protein2.pdb',
    lig1_mask_path=f'{data_dir}/{name}/lig1_mask.pdb', lig2_mask_path=f'{data_dir}/{name}/unbound_lig2.pdb',
    unbound_lig1_path=f'{data_dir}/{name}/unbound_lig1.pdb', unbound_lig2_path=f'{data_dir}/{name}/unbound_lig2.pdb',
    cfg=cfg)

```

This will save the predicted ternary complex pdb file in the `cfg.save_dir` directory.

### Training

For training with 2 GPUs, use the command below:

```Bash
bash tools/dist_train.sh deepternary/configs/protac.py 2
```

Codes for model and dataset are located under `deepternary/models/`, their defination is in the config files under `deepternary/configs/`.


# Citing this work

If you find this codebase, the model useful, please cite our paper:

```BibTeX
@article{xueDeepTernary2025,
  title = {{{SE}}(3)-Equivariant Ternary Complex Prediction towards Target Protein Degradation},
  shorttitle = {{{DeepTernary}}},
  author = {Xue, Fanglei and Zhang, Meihan and Li, Shuqi and Gao, Xinyu and Wohlschlegel, James A. and Huang, Wenbing and Yang, Yi and Deng, Weixian},
  year = {2025},
  month = jul,
  journal = {Nature Communications},
  volume = {16},
  number = {1},
  pages = {5514},
  publisher = {Nature Publishing Group},
  issn = {2041-1723},
  doi = {10.1038/s41467-025-61272-5},
  urldate = {2025-07-22},
  langid = {english}
}
```


