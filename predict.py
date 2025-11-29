"""User-friendly PROTAC/MGD inference script.

This script predicts ternary complex structures using a trained DeepTernary model.
It focuses on easy usage for non-CS users and reuses the core utilities from the
existing codebase without re-implementing modeling logic.

Basic usage examples
--------------------

PROTAC prediction (unbound setting):

    python predict.py \
        --task PROTAC \
        --lig /path/to/ligand.pdb \
        --p1 /path/to/unbound_protein1.pdb \
        --p2 /path/to/unbound_protein2.pdb \
        --unbound-lig1 /path/to/unbound_lig1.pdb \
        --unbound-lig2 /path/to/unbound_lig2.pdb \
        --lig1-mask /path/to/lig1_mask.pdb \
        --lig2-mask /path/to/lig2_mask.pdb \
        --outdir ./results/protac_case

MGD prediction (bound setting):

    python predict.py \
        --task MGD \
        --lig /path/to/ligand.pdb \
        --p1 /path/to/protein1.pdb \
        --p2 /path/to/protein2.pdb \
        --outdir ./results/mgd_case

Notes
-----
- The script auto-selects config and checkpoint based on --task:
  - PROTAC: deepternary/configs/protac.py, checkpoints under output/checkpoints/PROTAC
  - MGD:    deepternary/configs/glue.py,   checkpoints under output/checkpoints/MGD
  If the primary checkpoint directory is missing, it falls back to output_Old/checkpoints/<TASK>.
- Seeds default to 40 for PROTAC and 1 for MGD. You can change with --seeds.
- Device defaults to CUDA if available; pass --device cpu to force CPU.
- Ground-truth based metrics (e.g., DockQ) are NOT required nor computed here.
"""

import argparse
import csv
import os
import tempfile
from copy import deepcopy
from functools import partial
from typing import List, Tuple

import dgl
import numpy as np
import scipy.spatial as spa
import torch
from biopandas.pdb import PandasPdb
from mmengine.config import Config
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.DistanceGeometry.DistGeom import DoTriangleSmoothing
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

from deepternary.models.correct import correct_ligand
from deepternary.models.geometry_utils import random_rotation_translation
from deepternary.models.path import IDEAL_PATH
from deepternary.models.pdb_utils import (
    merge_pdbs,
    set_new_chain,
    set_new_coords,
    write_pdb,
)
from deepternary.models.process_mols import (
    distance_featurizer,
    get_geometry_graph_ring,
    get_lig_graph_revised,
    get_rdkit_coords_v2,
    get_rec_graph,
    get_receptor_inference,
    lig_atom_featurizer,
    read_molecule,
    rigid_transform_Kabsch_3D,
)
from deepternary.models.rotate_utils import kabsch, rotate_and_translate
from deepternary.models.ternary_pdb import get_pocket_and_mask
from mmpretrain.apis.model import get_model


def auto_download_ideal_sdf(lig_name):
    """Auto-download ideal SDF from RCSB if not present locally."""
    import requests
    ideal_sdf_path = f'{IDEAL_PATH}/{lig_name}_ideal.sdf'
    if not os.path.exists(ideal_sdf_path):
        os.makedirs(IDEAL_PATH, exist_ok=True)
        url = f'https://files.rcsb.org/ligands/download/{lig_name}_ideal.sdf'
        response = requests.get(url)
        with open(ideal_sdf_path, 'wb') as f:
            f.write(response.content)
    return ideal_sdf_path


def adjust_bounds(bounds_mat, atom_map, mol):
    """
    Codes from Reviewer #3.
    Adjusts the bounds matrix based on the distance between atoms in the given map.

    Args:
        bounds_mat: The bounds matrix to adjust.
        atom_map: A list of atom indices.
        mol: The molecule object.

    Returns:
        A tuple containing the adjusted bounds matrix and the result of triangle smoothing.
    """
    for i in range(len(atom_map)):
        atom_i = atom_map[i]
        for j in range(i + 1, len(atom_map)):
            atom_j = atom_map[j]
            a = min(atom_i, atom_j)
            b = max(atom_i, atom_j)
            pos1 = mol.GetConformer().GetAtomPosition(a)
            pos2 = mol.GetConformer().GetAtomPosition(b)
            distance = pos1.Distance(pos2)
            bounds_mat[a, b] = distance + 0.1
            bounds_mat[b, a] = distance - 0.1
    check = DoTriangleSmoothing(bounds_mat)

    return bounds_mat, check


def get_rdkit_coords_with_fixed_ends(ligand, lig1, lig2, seed=None):
    """
    Codes from Reviewer #3.
    Fixed two ends, only use rdkit to generate the rest of the coordinates"""
    params = rdDistGeom.ETKDGv3()
    ch1_map = ligand.GetSubstructMatch(lig1)
    ch2_map = ligand.GetSubstructMatch(lig2)
    if not (len(ch1_map) and len(ch2_map)):
        # fall back to the original method
        conf_id = rdDistGeom.EmbedMolecule(ligand, params)
    else:
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(ligand)
        bounds, check1 = adjust_bounds(bounds, ch1_map, ligand)
        bounds, check2 = adjust_bounds(bounds, ch2_map, ligand)
        params.SetBoundsMat(bounds)
        if seed is not None:
            params.randomSeed = seed
        conf_ids = rdDistGeom.EmbedMultipleConfs(ligand, 1, params)
    ligand = Chem.RemoveHs(ligand)
    conf = ligand.GetConformer()
    lig_coords = conf.GetPositions()
    return ligand, lig_coords


embedding_failed_names = set()
def get_lig_graph_protac(
    mol,
    name,
    lig1=None,
    lig2=None,
    radius=20,
    max_neighbors=None,
    use_rdkit_coords=False,
    use_random_coords=True,
    seed=None,
):
    """Build ligand graph with FIX_TWO_ENDS approach (Reviewer #3 method)."""
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    if use_rdkit_coords:
        if name in embedding_failed_names:
            rdkit_coords = None
        else:
            try:
                mol, rdkit_coords = get_rdkit_coords_with_fixed_ends(mol, lig1, lig2, seed=seed)
            except Exception as e:
                print(f"Error in getting RDKit coordinates: {e}")
                rdkit_coords = None
        if rdkit_coords is None:
            print(f"{name} RDKit coordinate generation failed. Using ideal.sdf")
            embedding_failed_names.add(name)
            ideal_path = auto_download_ideal_sdf(name.split('_')[-1])
            ideal_mol = read_molecule(ideal_path, sanitize=True, remove_hs=True)
            if ideal_mol is None:
                raise ValueError(f"ideal_mol is None for {ideal_path}")

            try:
                mol, rdkit_coords = get_rdkit_coords_with_fixed_ends(deepcopy(ideal_mol), lig1, lig2, seed=seed)
            except Exception as e:
                mol, rdkit_coords = get_rdkit_coords_v2(deepcopy(ideal_mol), seed=seed, use_random_coords=use_random_coords)

            if mol is None:
                mol = ideal_mol
                try:
                    rdkit_coords = ideal_mol.GetConformer().GetPositions()
                except Exception as e:
                    raise RuntimeError(f"ideal_mol {ideal_path} RDKit coordinate generation failed: {e}")
            if rdkit_coords.shape != true_lig_coords.shape:
                raise RuntimeError(
                    f"{name}, rdkit_coords.shape = {rdkit_coords.shape}, \
                true_lig_coords.shape = {true_lig_coords.shape}"
                )

        R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
        lig_coords = (R @ (rdkit_coords).T).T + t.squeeze()
    else:
        lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)

    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[
                1 : max_neighbors + 1
            ]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[
                1:2
            ]  # closest would be the index i itself > self loop
            print(
                f"The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}"
            )
        assert i not in dst
        assert dst != []
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distance[i, dst]
        sigma = np.array([1.0, 2.0, 5.0, 10.0, 30.0]).reshape((-1, 1))
        weights = softmax(
            -valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1
        )  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = lig_coords[src, :] - lig_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)

        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph(
        (torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32
    )

    graph.ndata["feat"] = lig_atom_featurizer(mol)
    graph.edata["feat"] = distance_featurizer(
        dist_list, 0.75
    )  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata["x"] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata["mu_r_norm"] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    if use_rdkit_coords:
        graph.ndata["new_x"] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    return mol, graph


def replace_to_unbound_coords(whole_coords, part_coords):
    """
    Replace PROTAC coordinates with unbound coordinates for matching atoms.

    whole_coords: the whole PROTAC
    part_coords: unbound lig1 or unbound lig2
    """
    # First, find need to update whole atoms
    cdist = torch.cdist(whole_coords, part_coords)
    update_mask = cdist.min(dim=1)[0] < 1
    # Update to the closed unbound atom coords
    whole_coords[update_mask] = part_coords[cdist.argmin(dim=1)[update_mask]]

    # check
    cdist = torch.cdist(whole_coords, part_coords)
    assert cdist.min(dim=1)[0][update_mask].max() < 1

    return whole_coords, update_mask


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_task_paths(task: str) -> Tuple[str, str]:
    """Return (config_path, checkpoint_path) for a given task.

    """
    task = task.upper()
    if task == "PROTAC":
        config_path = "deepternary/configs/protac.py"
        ckpt_dir = "output/checkpoints/PROTAC"
    elif task in ("MGD", "MOLECULARGLUE", "GLUE"):
        config_path = "deepternary/configs/glue.py"
        ckpt_dir = "output/checkpoints/MGD"
    else:
        raise ValueError("--task must be one of: PROTAC, MGD")

    last_ckpt = os.path.join(ckpt_dir, "last_checkpoint")
    if os.path.exists(last_ckpt):
        with open(last_ckpt, "r") as f:
            ckpt_path = f.read().strip()
        if os.path.exists(ckpt_path):
            return config_path, ckpt_path
    raise FileNotFoundError(
        f"Could not find checkpoint for task {task}. Expected last_checkpoint under: "
        f"{ckpt_dir}"
    )


def _load_model(config_path: str, checkpoint_path: str, device: str):
    cfg = Config.fromfile(config_path)
    if hasattr(cfg, "model") and isinstance(cfg.model, dict):
        if "noise_initial" in cfg.model:
            cfg.model["noise_initial"] = 0
        else:
            try:
                cfg.model.noise_initial = 0
            except Exception:
                pass
    cfg.device = device
    if hasattr(cfg, "model") and isinstance(cfg.model, dict):
        cfg.model["device"] = device
    try:
        model: torch.nn.Module = get_model(
            cfg, pretrained=checkpoint_path, device=device
        )
    except TypeError:
        model: torch.nn.Module = get_model(cfg, pretrained=checkpoint_path)
        model.to(device)
    model.eval()
    cfg.nn_model = model
    return cfg, model


def _prepare_protac_inputs(
    name: str,
    lig_path: str,
    p1_path: str,
    p2_path: str,
    lig1_mask_path: str,
    lig2_mask_path: str,
    unbound_lig1_path: str,
    unbound_lig2_path: str,
    cfg,
    use_fix_two_ends: bool = True,
):
    """Prepare graphs, pocket masks and initial ligand graph for PROTAC.

    Args:
        use_fix_two_ends: If True, use get_lig_graph_protac with fixed two ends (Reviewer #3 method).
                         If False, use get_lig_graph_revised.
    """
    ds_cfg = cfg.test_dataloader.dataset

    lig_origin = read_molecule(lig_path, sanitize=True, remove_hs=ds_cfg.remove_h)

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(
        p1_path
    )
    p1_graph_origin = get_rec_graph(
        recs,
        recs_coords,
        c_alpha_coords,
        n_coords,
        c_coords,
        use_rec_atoms=ds_cfg.use_rec_atoms,
        rec_radius=ds_cfg.rec_graph_radius,
        surface_max_neighbors=ds_cfg.surface_max_neighbors,
        surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
        surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
        c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,
    )

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(
        p2_path
    )
    p2_graph_origin = get_rec_graph(
        recs,
        recs_coords,
        c_alpha_coords,
        n_coords,
        c_coords,
        use_rec_atoms=ds_cfg.use_rec_atoms,
        rec_radius=ds_cfg.rec_graph_radius,
        surface_max_neighbors=ds_cfg.surface_max_neighbors,
        surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
        surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
        c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,
    )

    # Read complete unbound ligands (not just masks)
    unbound_lig1 = read_molecule(unbound_lig1_path, sanitize=True, remove_hs=True)
    unbound_lig2 = read_molecule(unbound_lig2_path, sanitize=True, remove_hs=True)
    lig1_coords = torch.from_numpy(unbound_lig1.GetConformer().GetPositions()).float()
    lig2_coords = torch.from_numpy(unbound_lig2.GetConformer().GetPositions()).float()

    # Build ligand graph using the same approach as predict_cpu.py
    lig = deepcopy(lig_origin)
    if use_fix_two_ends:
        # Use get_lig_graph_protac with fixed two ends
        lig1_mask_mol = read_molecule(lig1_mask_path, sanitize=True, remove_hs=True)
        lig2_mask_mol = read_molecule(lig2_mask_path, sanitize=True, remove_hs=True)
        graph_func = partial(get_lig_graph_protac, lig1=lig1_mask_mol, lig2=lig2_mask_mol)
    else:
        graph_func = get_lig_graph_revised

    lig, lig_graph = graph_func(
        lig,
        name=name,
        max_neighbors=ds_cfg.lig_max_neighbors,
        radius=ds_cfg.lig_graph_radius,
        use_rdkit_coords=True,
        use_random_coords=False,
        seed=0,
    )
    geometry_graph = get_geometry_graph_ring(lig)

    # Get ground truth ligand coords and replace with unbound coords (like predict_cpu.py)
    lig_coords_gt = deepcopy(lig_graph.ndata['x'])
    lig_coords_gt, update_mask1 = replace_to_unbound_coords(lig_coords_gt, lig1_coords)
    lig_coords_gt, update_mask2 = replace_to_unbound_coords(lig_coords_gt, lig2_coords)
    pocket_mask = update_mask1 | update_mask2

    # Compute pocket masks using the updated coords
    p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = get_pocket_and_mask(
        lig_coords_gt, p1_graph_origin.ndata["x"], cutoff=ds_cfg.pocket_cutoff
    )
    p2lig_lig_pocket_coords_origin, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = get_pocket_and_mask(
        lig_coords_gt, p2_graph_origin.ndata["x"], cutoff=ds_cfg.pocket_cutoff
    )

    # Filter pocket masks to only include unbound atoms
    p1lig_lig_pocket_mask = p1lig_lig_pocket_mask & pocket_mask
    p1lig_lig_pocket_coords = lig_coords_gt[p1lig_lig_pocket_mask]

    p2lig_lig_pocket_mask = p2lig_lig_pocket_mask & pocket_mask
    p2lig_lig_pocket_coords = lig_coords_gt[p2lig_lig_pocket_mask]

    return dict(
        lig_origin=lig_origin,
        lig_graph=lig_graph,
        geometry_graph=geometry_graph,
        p1_graph_origin=p1_graph_origin,
        p2_graph_origin=p2_graph_origin,
        p1lig_lig_pocket_mask=p1lig_lig_pocket_mask,
        p1lig_lig_pocket_coords=p1lig_lig_pocket_coords,
        p2lig_lig_pocket_mask=p2lig_lig_pocket_mask,
        p2lig_lig_pocket_coords=p2lig_lig_pocket_coords,
        p1lig_p1_pocket_mask=p1lig_p1_pocket_mask,
        p2lig_p2_pocket_mask=p2lig_p2_pocket_mask,
        lig1_mask_mol=read_molecule(lig1_mask_path, sanitize=True, remove_hs=True) if use_fix_two_ends else None,
        lig2_mask_mol=read_molecule(lig2_mask_path, sanitize=True, remove_hs=True) if use_fix_two_ends else None,
    )


def _run_protac(
    name: str,
    lig_path: str,
    p1_path: str,
    p2_path: str,
    lig1_mask_path: str,
    lig2_mask_path: str,
    unbound_lig1_path: str,
    unbound_lig2_path: str,
    outdir: str,
    cfg,
    num_seeds: int,
    apply_correction: bool,
) -> str:
    """Run PROTAC inference and save results. Returns the path to summary.csv."""
    _ensure_dir(outdir)
    prepared = _prepare_protac_inputs(
        name,
        lig_path,
        p1_path,
        p2_path,
        lig1_mask_path,
        lig2_mask_path,
        unbound_lig1_path,
        unbound_lig2_path,
        cfg,
    )

    lig_origin = prepared["lig_origin"]
    p1_graph_origin = prepared["p1_graph_origin"]
    p2_graph_origin = prepared["p2_graph_origin"]
    p1lig_p1_pocket_mask = prepared["p1lig_p1_pocket_mask"]
    p2lig_p2_pocket_mask = prepared["p2lig_p2_pocket_mask"]
    lig1_mask_mol = prepared["lig1_mask_mol"]
    lig2_mask_mol = prepared["lig2_mask_mol"]

    ds_cfg = cfg.test_dataloader.dataset

    # Read complete unbound ligands for pocket replacement
    unbound_lig1 = read_molecule(unbound_lig1_path, sanitize=True, remove_hs=True)
    unbound_lig2 = read_molecule(unbound_lig2_path, sanitize=True, remove_hs=True)
    lig1_coords = torch.from_numpy(unbound_lig1.GetConformer().GetPositions()).float()
    lig2_coords = torch.from_numpy(unbound_lig2.GetConformer().GetPositions()).float()

    results = []
    seed = 0
    while len(results) < num_seeds:
        print(f"Generating PROTAC conformer {name} with seed {seed}...")

        # Regenerate ligand graph for each seed (same as predict_cpu.py)
        lig = deepcopy(lig_origin)
        p1_graph = deepcopy(p1_graph_origin)
        p2_graph = deepcopy(p2_graph_origin)

        if lig1_mask_mol is not None and lig2_mask_mol is not None:
            # Use get_lig_graph_protac with fixed two ends
            graph_func = partial(get_lig_graph_protac, lig1=lig1_mask_mol, lig2=lig2_mask_mol)
        else:
            graph_func = get_lig_graph_revised

        lig, lig_graph = graph_func(
            lig,
            name=name,
            max_neighbors=ds_cfg.lig_max_neighbors,
            radius=ds_cfg.lig_graph_radius,
            use_rdkit_coords=True,
            use_random_coords=False,
            seed=seed,
        )
        geometry_graph = get_geometry_graph_ring(lig)

        # Get ground truth ligand coords and replace with unbound coords
        lig_coords_gt = deepcopy(lig_graph.ndata['x'])
        lig_coords_gt, update_mask1 = replace_to_unbound_coords(lig_coords_gt, lig1_coords)
        lig_coords_gt, update_mask2 = replace_to_unbound_coords(lig_coords_gt, lig2_coords)
        pocket_mask = update_mask1 | update_mask2

        # Compute pocket masks
        p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, _ = get_pocket_and_mask(
            lig_coords_gt, p1_graph.ndata["x"], cutoff=ds_cfg.pocket_cutoff
        )
        p2lig_lig_pocket_coords_origin, p2lig_lig_pocket_mask, _ = get_pocket_and_mask(
            lig_coords_gt, p2_graph.ndata["x"], cutoff=ds_cfg.pocket_cutoff
        )

        # Filter pocket masks to only include unbound atoms
        p1lig_lig_pocket_mask = p1lig_lig_pocket_mask & pocket_mask
        p1lig_lig_pocket_coords = lig_coords_gt[p1lig_lig_pocket_mask]
        p1lig_p1_pocket_coords = deepcopy(p1lig_lig_pocket_coords)

        p2lig_lig_pocket_mask = p2lig_lig_pocket_mask & pocket_mask
        p2lig_lig_pocket_coords = lig_coords_gt[p2lig_lig_pocket_mask]

        # Store ground truth p2 coords before random transformation
        protein2_coords_gt = deepcopy(p2_graph.ndata["x"])

        data = dict(
            lig_graph=lig_graph,
            rec_graph=p1_graph,
            rec2_graph=p2_graph,
            geometry_graph=geometry_graph,
            complex_name=[name],
            rec2_coords=[protein2_coords_gt],
            rec2_coords_input=[p2_graph.ndata["x"]],
            p1lig_p1_pocket_mask=[p1lig_p1_pocket_mask],
            p1lig_p1_pocket_coords=[p1lig_p1_pocket_coords],
            p1lig_lig_pocket_mask=[p1lig_lig_pocket_mask],
            p1lig_lig_pocket_coords=[p1lig_lig_pocket_coords],
            p2lig_p2_pocket_mask=[p2lig_p2_pocket_mask],
            p2lig_p2_pocket_coords=[p2lig_lig_pocket_coords],
            p2lig_lig_pocket_mask=[p2lig_lig_pocket_mask],
            p2lig_lig_pocket_coords=[p2lig_lig_pocket_coords],
        )

        # Apply random transformations to ligand and p2
        rot_T, rot_b = random_rotation_translation(translation_distance=5)
        lig_coords_to_move = data['lig_graph'].ndata['new_x']
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        data['lig_graph'].ndata['new_x'] = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        data['p1lig_lig_pocket_coords'][0] = (rot_T @ (data['p1lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b
        data['p2lig_lig_pocket_coords'][0] = (rot_T @ (data['p2lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b

        # Random move p2
        protein2_rot_T, protein2_rot_b = random_rotation_translation(translation_distance=5)
        protein2_coord_to_move = data['rec2_graph'].ndata['x']
        protein2_mean_to_remove = protein2_coord_to_move.mean(dim=0, keepdims=True)
        data['rec2_graph'].ndata['x'] = (protein2_rot_T @ (protein2_coord_to_move - protein2_mean_to_remove).T).T + protein2_rot_b
        data['p2lig_p2_pocket_coords'][0] = (protein2_rot_T @ (data['p2lig_p2_pocket_coords'][0] - protein2_mean_to_remove).T).T + protein2_rot_b
        data['rec2_coords_input'] = [data['rec2_graph'].ndata['x']]

        with torch.no_grad():
            outputs = cfg.nn_model(**data, mode="predict")[0]

        pred_lig_coords = outputs["ligs_coords_pred"]
        rot2 = outputs["rotation_2"]
        trans2 = outputs["translation_2"]
        pred_p2_rmsd = outputs["p2_rmsd_pred"].item()
        pred_p2_coords = rotate_and_translate(data['rec2_coords_input'][0], rot2, trans2)
        # Optional ligand correction
        if apply_correction:
            pred_lig_coords = correct_ligand(
                pred_lig_coords, lig_graph.ndata["new_x"], lig_origin
            )
            pred_lig_coords = torch.from_numpy(pred_lig_coords).float()

        # Clash estimation between p1 and predicted p2
        cdist = torch.cdist(p1_graph.ndata["x"], pred_p2_coords)
        clash_ratio = (cdist.min(dim=1).values < 3.8).sum() / cdist.shape[0]

        # Save outputs (only complex_pred). Use temp files for intermediate ligand and p2.
        with tempfile.TemporaryDirectory() as tmpdir:
            lig_pred_path = os.path.join(tmpdir, f"lig_pred_{name}_{len(results)}.pdb")
            pred_lig = set_new_coords(
                lig_path, pred_lig_coords.cpu().numpy(), is_ligand=True
            )
            write_pdb(pred_lig, lig_pred_path)

            # Prepare p1 with chain A
            p1_pred_file = os.path.join(tmpdir, f"p1_pred_{name}_{len(results)}.pdb")
            p1_pdb = PandasPdb().read_pdb(p1_path)
            p1_pdb.df["ATOM"]["chain_id"] = "A"
            p1_pdb.to_pdb(p1_pred_file, records=["ATOM"], gz=False)

            # Prepare p2 with chain B
            p2_pred_file = os.path.join(tmpdir, f"p2_pred_{name}_{len(results)}.pdb")
            p2_pdb = PandasPdb().read_pdb(p2_path)
            p2_gt_coords = p2_pdb.df["ATOM"][
                ["x_coord", "y_coord", "z_coord"]
            ].to_numpy()
            # Compute transform on graph nodes using GT coords, then apply to atom-level coords
            p2_rot, p2_trans, _ = kabsch(protein2_coords_gt, pred_p2_coords)
            p2_pred_coords = rotate_and_translate(
                p2_gt_coords, p2_rot.cpu().numpy(), p2_trans.cpu().numpy()
            )
            p2_pdb.df["ATOM"][["x_coord", "y_coord", "z_coord"]] = p2_pred_coords
            p2_pdb.df["ATOM"]["chain_id"] = "B"
            p2_pdb.to_pdb(p2_pred_file, records=["ATOM"], gz=False)

            complex_path = os.path.join(
                outdir, f"complex_pred_{name}_{len(results)}.pdb"
            )
            merge_pdbs(
                pdb_files=[p1_pred_file, p2_pred_file, lig_pred_path], path=complex_path
            )

        results.append(
            dict(
                seed=len(results),
                pred_p2_rmsd=float(pred_p2_rmsd),
                clash_ratio=float(clash_ratio),
                complex_pred=complex_path,
            )
        )
        seed += 1

    # Sort by surrogate score (lower predicted P2 RMSD is better)
    results = sorted(results, key=lambda x: x["pred_p2_rmsd"])

    # Write a summary CSV
    summary_csv = os.path.join(outdir, f"summary_{name}.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "pred_p2_rmsd", "clash_ratio", "complex_pred"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    k: row[k]
                    for k in ["seed", "pred_p2_rmsd", "clash_ratio", "complex_pred"]
                }
            )

    print(f"Saved results to: {outdir}")
    print(f"Top-1 predicted P2 RMSD (surrogate): {results[0]['pred_p2_rmsd']:.4f}")
    return summary_csv


def _run_mgd(
    name: str,
    lig_path: str,
    p1_path: str,
    p2_path: str,
    outdir: str,
    cfg,
    apply_correction: bool,
) -> str:
    """Minimal MGD inference: build graphs and predict a single seed pose."""
    _ensure_dir(outdir)
    ds_cfg = cfg.test_dataloader.dataset

    lig_origin = read_molecule(lig_path, sanitize=True, remove_hs=ds_cfg.remove_h)
    lig, lig_graph = get_lig_graph_revised(
        lig_origin,
        name=name,
        max_neighbors=ds_cfg.lig_max_neighbors,
        ideal_path=None,
        use_random_coords=False,
        seed=0,
        use_rdkit_coords=True,
        radius=ds_cfg.lig_graph_radius,
    )
    geometry_graph = get_geometry_graph_ring(lig_origin)

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(
        p1_path
    )
    p1_graph = get_rec_graph(
        recs,
        recs_coords,
        c_alpha_coords,
        n_coords,
        c_coords,
        use_rec_atoms=ds_cfg.use_rec_atoms,
        rec_radius=ds_cfg.rec_graph_radius,
        surface_max_neighbors=ds_cfg.surface_max_neighbors,
        surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
        surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
        c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,
    )

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(
        p2_path
    )
    p2_graph = get_rec_graph(
        recs,
        recs_coords,
        c_alpha_coords,
        n_coords,
        c_coords,
        use_rec_atoms=ds_cfg.use_rec_atoms,
        rec_radius=ds_cfg.rec_graph_radius,
        surface_max_neighbors=ds_cfg.surface_max_neighbors,
        surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
        surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
        c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,
    )

    # Store ground truth p2 coords (for MGD/bound, no random transform applied)
    protein2_coords_gt = deepcopy(p2_graph.ndata["x"])

    data = dict(
        lig_graph=lig_graph,
        rec_graph=p1_graph,
        rec2_graph=p2_graph,
        geometry_graph=geometry_graph,
        complex_name=[name],
        rec2_coords=[protein2_coords_gt],
        rec2_coords_input=[p2_graph.ndata["x"]],
        p1lig_p1_pocket_mask=[
            torch.zeros(lig_graph.ndata["x"].shape[0], dtype=torch.bool)
        ],
        p1lig_p1_pocket_coords=[lig_graph.ndata["x"][0:0]],
        p1lig_lig_pocket_mask=[
            torch.zeros(lig_graph.ndata["x"].shape[0], dtype=torch.bool)
        ],
        p1lig_lig_pocket_coords=[lig_graph.ndata["x"][0:0]],
        p2lig_p2_pocket_mask=[
            torch.zeros(lig_graph.ndata["x"].shape[0], dtype=torch.bool)
        ],
        p2lig_p2_pocket_coords=[lig_graph.ndata["x"][0:0]],
        p2lig_lig_pocket_mask=[
            torch.zeros(lig_graph.ndata["x"].shape[0], dtype=torch.bool)
        ],
        p2lig_lig_pocket_coords=[lig_graph.ndata["x"][0:0]],
    )

    with torch.no_grad():
        outputs = cfg.nn_model(**data, mode="predict")[0]

    pred_lig_coords = outputs["ligs_coords_pred"]
    rot2 = outputs["rotation_2"]
    trans2 = outputs["translation_2"]
    pred_p2_coords = rotate_and_translate(p2_graph.ndata["x"], rot2, trans2)

    if apply_correction:
        pred_lig_coords = correct_ligand(
            pred_lig_coords, lig_graph.ndata["x"], lig_origin
        )
        pred_lig_coords = torch.from_numpy(pred_lig_coords).float()

    # Only output complex_pred using temp files for intermediates
    with tempfile.TemporaryDirectory() as tmpdir:
        lig_pred_path = os.path.join(tmpdir, f"lig_pred_{name}_0.pdb")
        pred_lig = set_new_coords(
            lig_path, pred_lig_coords.cpu().numpy(), is_ligand=True
        )
        write_pdb(pred_lig, lig_pred_path)

        # Prepare p1 with chain A
        p1_pred_file = os.path.join(tmpdir, f"p1_pred_{name}_0.pdb")
        p1_pdb = PandasPdb().read_pdb(p1_path)
        p1_pdb.df["ATOM"]["chain_id"] = "A"
        p1_pdb.to_pdb(p1_pred_file, records=["ATOM"], gz=False)

        # Prepare p2 with chain B
        p2_pred_file = os.path.join(tmpdir, f"p2_pred_{name}_0.pdb")
        p2_pdb = PandasPdb().read_pdb(p2_path)
        p2_gt_coords = p2_pdb.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy()
        # Compute transform on graph nodes using GT coords, then apply to atom-level coords
        p2_rot, p2_trans, _ = kabsch(protein2_coords_gt, pred_p2_coords)
        p2_pred_coords = rotate_and_translate(
            p2_gt_coords, p2_rot.cpu().numpy(), p2_trans.cpu().numpy()
        )
        p2_pdb.df["ATOM"][["x_coord", "y_coord", "z_coord"]] = p2_pred_coords
        p2_pdb.df["ATOM"]["chain_id"] = "B"
        p2_pdb.to_pdb(p2_pred_file, records=["ATOM"], gz=False)

        complex_path = os.path.join(outdir, f"complex_pred_{name}_0.pdb")
        merge_pdbs(pdb_files=[p1_pred_file, p2_pred_file, lig_pred_path], path=complex_path)

    summary_csv = os.path.join(outdir, f"summary_{name}.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "complex_pred"])
        writer.writeheader()
        writer.writerow(dict(seed=0, complex_pred=complex_path))
    print(f"Saved results to: {outdir}")
    return summary_csv


def _parse_args():
    p = argparse.ArgumentParser(description="DeepTernary inference (user-friendly)")
    p.add_argument(
        "--task",
        required=True,
        choices=["PROTAC", "MGD"],
        help="Task type: PROTAC or MGD",
    )
    p.add_argument("--lig", required=True, help="Ligand file (.pdb or .sdf)")
    p.add_argument("--p1", required=True, help="Protein 1 PDB (unbound for PROTAC)")
    p.add_argument("--p2", required=True, help="Protein 2 PDB (unbound for PROTAC)")
    p.add_argument(
        "--outdir", required=True, help="Directory to save predicted outputs"
    )
    p.add_argument(
        "--name", default=None, help="Sample name. Defaults to ligand filename stem"
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds (PROTAC default 40, MGD default 1)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda or cpu (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--no-correct", action="store_true", help="Disable ligand correction (EquiBind)"
    )

    # PROTAC-specific
    p.add_argument(
        "--unbound-lig1", dest="unbound_lig1", help="Unbound lig1 PDB (PROTAC)"
    )
    p.add_argument(
        "--unbound-lig2", dest="unbound_lig2", help="Unbound lig2 PDB (PROTAC)"
    )
    p.add_argument(
        "--lig1-mask", dest="lig1_mask", help="Lig1 mask PDB (subset for alignment)"
    )
    p.add_argument(
        "--lig2-mask", dest="lig2_mask", help="Lig2 mask PDB (subset for alignment)"
    )

    # Advanced overrides (optional)
    p.add_argument("--config", default=None, help=argparse.SUPPRESS)
    p.add_argument("--checkpoint", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = _parse_args()

    # Resolve defaults
    if args.name is None:
        args.name = os.path.splitext(os.path.basename(args.lig))[0]
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seeds is None:
        args.seeds = 40 if args.task.upper() == "PROTAC" else 1

    # Resolve config/checkpoint
    if args.config is None or args.checkpoint is None:
        try:
            auto_config, auto_ckpt = _resolve_task_paths(args.task)
        except Exception as e:
            # If auto-detection fails and user supplied overrides, continue; else raise
            if args.config is None or args.checkpoint is None:
                raise
            auto_config, auto_ckpt = args.config, args.checkpoint
    else:
        auto_config, auto_ckpt = args.config, args.checkpoint

    print(f"Loading model for task {args.task}...")
    cfg, model = _load_model(auto_config, auto_ckpt, args.device)

    print(f"Predicting {args.task} structure for sample: {args.name}")
    if args.task.upper() == "PROTAC":
        # Validate required PROTAC inputs
        required = [
            args.unbound_lig1,
            args.unbound_lig2,
            args.lig1_mask,
            args.lig2_mask,
        ]
        names = ["--unbound-lig1", "--unbound-lig2", "--lig1-mask", "--lig2-mask"]
        for v, n in zip(required, names):
            if not v:
                raise ValueError(f"Missing required argument for PROTAC: {n}")
        _run_protac(
            name=args.name,
            lig_path=args.lig,
            p1_path=args.p1,
            p2_path=args.p2,
            lig1_mask_path=args.lig1_mask,
            lig2_mask_path=args.lig2_mask,
            unbound_lig1_path=args.unbound_lig1,
            unbound_lig2_path=args.unbound_lig2,
            outdir=args.outdir,
            cfg=cfg,
            num_seeds=args.seeds,
            apply_correction=not args.no_correct,
        )
    else:
        _run_mgd(
            name=args.name,
            lig_path=args.lig,
            p1_path=args.p1,
            p2_path=args.p2,
            outdir=args.outdir,
            cfg=cfg,
            apply_correction=not args.no_correct,
        )


if __name__ == "__main__":
    main()
