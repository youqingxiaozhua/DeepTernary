"""preprocess the pdb files to graphs to save time during training"""

import os
from copy import deepcopy

import numpy as np
import torch
from mmengine.config import Config
from torch import multiprocessing
from tqdm import tqdm

from deepternary.models.path import IDEAL_PATH, PDB_PATH, PREPROCESSED_PATH
from deepternary.models.process_mols import (
    ShapeNotEqualError,
    get_geometry_graph_ring,
    get_lig_graph_revised,
    get_pocket_coords,
    get_rec_graph,
    get_receptor,
    read_molecule,
)

PDB_PATH = PDB_PATH
IDEAL_LIG_PATH = IDEAL_PATH


def find_file_by_suffix(path, suffixs):
    for suffix in suffixs:
        file_path = f"{path}.{suffix}"
        if os.path.exists(file_path):
            return file_path
    raise FileNotFoundError(f"No file found in {path} with suffixs: {suffixs}")


def preprocess_noHs(name):
    """take a name and save preprocessed file"""
    save_folder = PREPROCESSED_PATH
    config_file = "deepternary/configs/glue.py"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{name}.pth")
    if os.path.exists(save_path):
        return 0

    cfg = Config.fromfile(config_file)
    cfg = cfg.train_dataloader.dataset
    lig_path = find_file_by_suffix(
        os.path.join(PDB_PATH, name, "ligand"), ["sdf", "pdb"]
    )
    try:
        lig = read_molecule(lig_path, sanitize=True, remove_hs=cfg.remove_h)
    except:
        return 1

    if lig is None:
        return 1
    p1_path = find_file_by_suffix(
        os.path.join(PDB_PATH, name, "protein1"), ["cif", "pdb"]
    )
    p2_path = find_file_by_suffix(
        os.path.join(PDB_PATH, name, "protein2"), ["cif", "pdb"]
    )
    # p1_path = os.path.join(COMPLEX_PATH, name, f'protein1.cif')
    # p2_path = os.path.join(COMPLEX_PATH, name, f'protein2.cif')
    # if not os.path.exists(p1_path):
    #     p1_path = os.path.join(COMPLEX_PATH, name, 'protein1.pdb')
    #     p2_path = os.path.join(COMPLEX_PATH, name, 'protein2.pdb')

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor(
        p1_path,
        lig,
        cutoff=cfg.chain_radius,
    )
    p1_pocket = get_pocket_coords(
        lig,
        recs_coords,
        cutoff=cfg.pocket_cutoff,
        pocket_mode=cfg.pocket_mode,
    )
    p1_graph = get_rec_graph(
        recs,
        recs_coords,
        c_alpha_coords,
        n_coords,
        c_coords,
        use_rec_atoms=cfg.use_rec_atoms,
        rec_radius=cfg.rec_graph_radius,
        surface_max_neighbors=cfg.surface_max_neighbors,
        surface_graph_cutoff=cfg.surface_graph_cutoff,
        surface_mesh_cutoff=cfg.surface_mesh_cutoff,
        c_alpha_max_neighbors=cfg.c_alpha_max_neighbors,
    )

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor(
        p2_path,
        lig,
        cutoff=cfg.chain_radius,
    )
    p2_pocket = get_pocket_coords(
        lig,
        recs_coords,
        cutoff=cfg.pocket_cutoff,
        pocket_mode=cfg.pocket_mode,
    )
    p2_graph = get_rec_graph(
        recs,
        recs_coords,
        c_alpha_coords,
        n_coords,
        c_coords,
        use_rec_atoms=cfg.use_rec_atoms,
        rec_radius=cfg.rec_graph_radius,
        surface_max_neighbors=cfg.surface_max_neighbors,
        surface_graph_cutoff=cfg.surface_graph_cutoff,
        surface_mesh_cutoff=cfg.surface_mesh_cutoff,
        c_alpha_max_neighbors=cfg.c_alpha_max_neighbors,
    )

    lig_name = name.split("_")[-1]
    assert len(lig_name) == 3

    random_lig_num = 50
    lig_graphs = []
    geometry_graphs = []
    lig_origin = deepcopy(lig)
    for seed in range(random_lig_num):
        lig = deepcopy(lig_origin)
        try:
            lig, lig_graph = get_lig_graph_revised(
                lig,
                name,
                ideal_path=os.path.join(IDEAL_LIG_PATH, f"{lig_name}_ideal.sdf"),
                use_random_coords=False,
                seed=seed,
                max_neighbors=cfg.lig_max_neighbors,
                use_rdkit_coords=cfg.use_rdkit_coords,
                radius=cfg.lig_graph_radius,
            )
        except ShapeNotEqualError:
            return 1
        lig_graphs.append(lig_graph)
        assert cfg.geometry_regularization_ring is True
        geometry_graph = get_geometry_graph_ring(lig, coords=lig_graph.ndata["new_x"])
        geometry_graphs.append(geometry_graph)

    result = dict(
        name=name,
        lig_graphs=lig_graphs,
        p1_pocket=p1_pocket,
        p2_pocket=p2_pocket,
        p1_graph=p1_graph,
        p2_graph=p2_graph,
        geometry_graphs=geometry_graphs,
    )
    torch.save(result, save_path)
    return 0


def read_and_remove_hydrogen():
    from rdkit import Chem

    list_folder = "data/DeepTernary/pdb1223"
    names = os.listdir(list_folder)
    for name in tqdm(names):
        lig_path = os.path.join(list_folder, name, "ligand.pdb")
        lig = Chem.MolFromPDBFile(lig_path)
        if lig is None:
            print("load lig error: ", name)
            continue
        lig_noHs = Chem.RemoveHs(lig)
        Chem.MolToPDBFile(lig_noHs, os.path.join(list_folder, name, "ligand_noHs.pdb"))


def filter_complex(name):
    data = torch.load(f"/opt/data/DeepTernary/preprosessd/pdb2311_noHs/{name}.pth")
    if len(data["p1_pocket"]) < 3 or len(data["p2_pocket"]) < 3:
        return False
    return True


if __name__ == "__main__":
    complexes = os.listdir(PDB_PATH)
    with multiprocessing.Pool(os.cpu_count() // 2) as pool:
        status = list(tqdm(pool.imap(preprocess_noHs, complexes), total=len(complexes)))
    status = np.array(status)
    print("status: ", np.unique(status, return_counts=True))
    # status:  (array([0, 1, 2]), array([34321,  1505,  6616]))
