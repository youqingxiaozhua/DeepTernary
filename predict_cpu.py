"""use preprocessed dataset as input"""
import argparse
import os
import shutil
from functools import partial
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from typing import List

import dgl
import numpy as np
import scipy.spatial as spa
import torch
from biopandas.pdb import PandasPdb
from mmengine.config import Config
from prettytable import PrettyTable
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.DistanceGeometry.DistGeom import DoTriangleSmoothing
from scipy.special import softmax
from torch import multiprocessing
from torch.multiprocessing import set_start_method
from tqdm import tqdm

from mmpretrain.apis.model import get_model

try:
    set_start_method('spawn')
except:
    pass

from deepternary.models.correct import correct_ligand
from deepternary.models.geometry_utils import random_rotation_translation
from deepternary.models.logger import log
from deepternary.models.path import IDEAL_PATH, PDB_PATH
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
    rigid_transform_Kabsch_3D
)
from deepternary.models.rotate_utils import kabsch, rotate_and_translate
from deepternary.models.ternary_pdb import get_pocket_and_mask

use_rdkit_coords = True

THREAD_NUM = 5
FIX_TWO_ENDS = False  # whether to fix the two ends of the PROTAC
CORRECT_LIGAND = True   # run EquiBind correction for ligand


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
    fixed two ends, only use rdkit to generate the rest of the coordinates"""
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
    ideal_path=None,
    seed=None,
):
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
            if isinstance(ideal_path, str):
                ideal_mol = read_molecule(ideal_path, sanitize=True, remove_hs=True)
            else:
                ideal_mol = ideal_path
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
            # assert rdkit_coords is not None, f'ideal_mol {ideal_path} RDKit coordinate generation failed'
            # rdkit_coords = ideal_mol.GetConformer().GetPositions()
            if rdkit_coords.shape != true_lig_coords.shape:
                raise RuntimeError(
                    f"{name}, rdkit_coords.shape = {rdkit_coords.shape}, \
                true_lig_coords.shape = {true_lig_coords.shape}"
                )

            # mol = ideal_mol
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
            log(
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




def save_mol_to_pdb(mol, path):
    mol = deepcopy(mol)
    Chem.MolToPDBFile(mol, path)


def RMSD(pred, gt):
    diff = pred - gt
    if isinstance(pred, torch.Tensor):
        return torch.sqrt(torch.sum(diff ** 2, dim=1).mean()).item()
    elif isinstance(pred, np.ndarray):
        return np.sqrt(np.sum(diff ** 2, axis=1).mean())
    else:
        raise NotImplementedError


def get_lig_coords(lig_path):
    lig = read_molecule(lig_path, sanitize=True, remove_hs=True)
    conf = lig.GetConformer()
    true_lig_coords = conf.GetPositions()
    return torch.from_numpy(true_lig_coords).float()


def classify_prediction(f_nat, l_rms, i_rms):
    # High-quality prediction
    if f_nat >= 0.5 and (l_rms <= 1.0 or i_rms <= 1.0):
        return 3  # High
    # Medium-quality prediction
    elif f_nat >= 0.3 and ((l_rms > 1.0 and l_rms <= 5.0) or (i_rms > 1.0 and i_rms <= 2.0)):
        return 2  # Medium
    # Acceptable-quality prediction
    elif f_nat >= 0.1 and ((l_rms > 5.0 and l_rms <= 10.0) or (i_rms > 2.0 and i_rms <= 4.0)):
        return 1  # Acceptable
    # Incorrect prediction
    else:
        return 0  # Incorrect


def replace_to_unbound_coords(whole_coords, part_coords):
    """
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


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('work_dir', help='saved model directory')
    p.add_argument('--config', help='test config file path', default=None)
    p.add_argument('--checkpoint', help='checkpoint file', default=None)
    args = p.parse_args()
    return args


@torch.no_grad()
def predict_one_unbound(name, cfg=None, seed_num=40):
    UNBOUND = True
    print(f'--{name}--')
    model = cfg.nn_model
    ds_cfg = cfg.test_dataloader.dataset

    UNBOUND_FOLDER = 'output/protac22'
    # UNBOUND_FOLDER = 'output/protac_new'
    lig_path = os.path.join(UNBOUND_FOLDER, name, 'ligand.pdb')
    p1_path = os.path.join(UNBOUND_FOLDER, name, 'unbound_protein1.pdb')
    p2_path = os.path.join(UNBOUND_FOLDER, name, 'unbound_protein2.pdb')

    lig_origin = read_molecule(lig_path, sanitize=True, remove_hs=ds_cfg.remove_h)

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(
        p1_path)
    p1_graph_origin = get_rec_graph(recs, recs_coords, c_alpha_coords, n_coords, c_coords,
                            use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
                            surface_max_neighbors=ds_cfg.surface_max_neighbors,
                            surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
                            surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
                            c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,)

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(
        p2_path)
    p2_graph_origin = get_rec_graph(recs, recs_coords, c_alpha_coords, n_coords, c_coords,
                            use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
                            surface_max_neighbors=ds_cfg.surface_max_neighbors,
                            surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
                            surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
                            c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,)
    protein2_coords_gt = deepcopy(p2_graph_origin.ndata['x'])

    if FIX_TWO_ENDS:
        lig1 = read_molecule(os.path.join(UNBOUND_FOLDER, name, 'unbound_lig1.pdb'), sanitize=True, remove_hs=True)
        lig2 = read_molecule(os.path.join(UNBOUND_FOLDER, name, 'unbound_lig2.pdb'), sanitize=True, remove_hs=True)

    results = []
    for seed in range(seed_num):
        lig = deepcopy(lig_origin)
        p1_graph = deepcopy(p1_graph_origin)
        p2_graph = deepcopy(p2_graph_origin)
        if FIX_TWO_ENDS:
            graph_func =  partial(get_lig_graph_protac, lig1=lig1, lig2=lig2)
        else:
            graph_func = get_lig_graph_revised
        lig, lig_graph = graph_func(
            lig, name=name, max_neighbors=ds_cfg.lig_max_neighbors,
            ideal_path=f'{IDEAL_PATH}/{name[-3:]}_ideal.sdf',
            # ideal_path=None,
            use_random_coords=False,
            seed=seed,
            use_rdkit_coords=True, radius=ds_cfg.lig_graph_radius)
        geometry_graph = get_geometry_graph_ring(lig)

        # find unbound pocket
        lig_coods_gt = deepcopy(lig_graph.ndata['x'])

        if UNBOUND:
            # unbound anchor and warhead
            lig1_coords = get_lig_coords(os.path.join(UNBOUND_FOLDER, name, 'unbound_lig1.pdb'))
            lig2_coords = get_lig_coords(os.path.join(UNBOUND_FOLDER, name, 'unbound_lig2.pdb'))
            lig_coods_gt, update_mask1 = replace_to_unbound_coords(lig_coods_gt, lig1_coords)
            lig_coods_gt, update_mask2 = replace_to_unbound_coords(lig_coods_gt, lig2_coords)
            pocket_mask = update_mask1 | update_mask2

        p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = get_pocket_and_mask(
            lig_coods_gt, p1_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
        p2lig_lig_pocket_coords_origin, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = get_pocket_and_mask(
            lig_coods_gt, p2_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
        assert torch.allclose(p1lig_lig_pocket_coords, lig_coods_gt[p1lig_lig_pocket_mask])

        if UNBOUND:
            p1lig_lig_pocket_mask = p1lig_lig_pocket_mask & pocket_mask
        p1lig_lig_pocket_coords = lig_coods_gt[p1lig_lig_pocket_mask]
        p1lig_p1_pocket_coords = deepcopy(p1lig_lig_pocket_coords)

        if UNBOUND:
            p2lig_lig_pocket_mask = p2lig_lig_pocket_mask & pocket_mask
        p2lig_lig_pocket_coords_origin = lig_coods_gt[p2lig_lig_pocket_mask]

        if UNBOUND:
            pocket_mask = p1lig_lig_pocket_mask | p2lig_lig_pocket_mask
            pockt_rdkit_coords = lig_graph.ndata['new_x'][pocket_mask]
            pocket_gt_coords = lig_graph.ndata['x'][pocket_mask]
            print('Pocket RMSD: ', RMSD(pockt_rdkit_coords, pocket_gt_coords))

        data = dict(
            lig_graph=lig_graph,
            rec_graph=p1_graph,
            rec2_graph=p2_graph,
            geometry_graph=geometry_graph,
            complex_name=[name],
            rec2_coords=[protein2_coords_gt],
            p1lig_p1_pocket_mask=[p1lig_p1_pocket_mask],
            p1lig_p1_pocket_coords=[p1lig_p1_pocket_coords],
            p1lig_lig_pocket_mask=[p1lig_lig_pocket_mask],
            p1lig_lig_pocket_coords=[p1lig_lig_pocket_coords],
            p2lig_p2_pocket_mask=[p2lig_p2_pocket_mask],
            p2lig_p2_pocket_coords=[p2lig_lig_pocket_coords_origin],
            p2lig_lig_pocket_mask=[p2lig_lig_pocket_mask],
            p2lig_lig_pocket_coords=[p2lig_lig_pocket_coords_origin],
        )

        # random move lig
        rot_T, rot_b = random_rotation_translation(translation_distance=5)
        if use_rdkit_coords:
            lig_coords_to_move = data['lig_graph'].ndata['new_x']
        else:
            lig_coords_to_move = data['lig_graph'].ndata['x']
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        data['lig_graph'].ndata['new_x'] = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        data['p1lig_lig_pocket_coords'][0] = (rot_T @ (data['p1lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b
        data['p2lig_lig_pocket_coords'][0] = (rot_T @ (data['p2lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b

        # random move p2
        protein2_rot_T, protein2_rot_b = random_rotation_translation(translation_distance=5)
        protein2_coord_to_move = data['rec2_graph'].ndata['x']
        protein2_mean_to_remove = protein2_coord_to_move.mean(dim=0, keepdims=True)
        data['rec2_graph'].ndata['x'] = (protein2_rot_T @ (protein2_coord_to_move - protein2_mean_to_remove).T).T + protein2_rot_b
        data['p2lig_p2_pocket_coords'][0] = (protein2_rot_T @ (data['p2lig_p2_pocket_coords'][0] - protein2_mean_to_remove).T).T + protein2_rot_b
        data['rec2_coords_input'] = [data['rec2_graph'].ndata['x']]

        with torch.no_grad():
            model_outputs = model(**data, mode='predict')[0]

        prediction = model_outputs['ligs_coords_pred']
        p2_to_p1_rotation = model_outputs['rotation_2']
        p2_to_p1_translation = model_outputs['translation_2']
        p2_rmsd_pred = model_outputs['p2_rmsd_pred'].item()
        lig_coords_gt = data['lig_graph'].ndata['x'].to(prediction.device)

        # rot, trans, is_nan = kabsch(prediction, lig_coords_gt)
        # kabsch_lig_pred = rotate_and_translate(prediction, rot, trans)
        # print('Kabsch RMSD: ', f'{RMSD(kabsch_lig_pred, lig_coords_gt):.2f}')
        # # save predict ligand
        # lig_path = os.path.join(PDB_PATH, name, 'ligand.pdb')
        # lig_pred_file = os.path.join(tmp_dir.name, f'lig_pred_{name}.pdb')
        # lig_pdb = PandasPdb().read_pdb(lig_path)
        # lig_pdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']] = prediction.cpu().numpy()
        # lig_pdb.to_pdb(lig_pred_file, records=['HETATM'], gz=False)

        # run correction
        if CORRECT_LIGAND:
            prediction = correct_ligand(prediction, lig_graph.ndata['new_x'], lig)
            prediction = torch.from_numpy(prediction).float()

        lig_pred_path = os.path.join(cfg.tmp_dir.name, f'lig_pred_{name}_{seed}.pdb')
        pred_lig = set_new_coords(lig_path, prediction.cpu().numpy(), is_ligand=True)
        pred_lig = set_new_chain(pred_lig, 'A', is_ligand=True)
        write_pdb(pred_lig, lig_pred_path)

        # calculate the rotate and translation to move p2 from gt to pred
        p2_gt = data['rec2_coords'][0].to(prediction.device)
        p2_pred = rotate_and_translate(data['rec2_coords_input'][0].to(prediction.device),
                                        p2_to_p1_rotation, p2_to_p1_translation)
        p2_rmsd_gt = RMSD(p2_gt, p2_pred)
        p1_gt = p1_graph.ndata['x']
        overall_rmsd = RMSD(torch.cat((p1_gt, p2_gt), dim=0), torch.cat((p1_gt, p2_pred), dim=0))
        p2_rot, p2_trans, _ = kabsch(p2_gt, p2_pred)

        # save predict complex
        p1_path = os.path.join(UNBOUND_FOLDER, name, 'protein1.pdb')
        p2_path = os.path.join(UNBOUND_FOLDER, name, 'protein2.pdb')
        p2_pred_file = os.path.join(cfg.tmp_dir.name, f'p2_pred_{name}_{seed}.pdb')
        p2_pdb = PandasPdb().read_pdb(p2_path)
        p2_gt_coords = p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
        p2_pred_coords = rotate_and_translate(p2_gt_coords, p2_rot.cpu().numpy(), p2_trans.cpu().numpy())
        p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = p2_pred_coords
        p2_pdb.to_pdb(p2_pred_file, records=['ATOM'], gz=False)
        complex_save_path = os.path.join(cfg.tmp_dir.name, f'complex_pred_{name}_{seed}.pdb')
        merge_pdbs(
            pdb_files=[p1_path, p2_pred_file, lig_pred_path],
            path=complex_save_path,
        )

        lig_rmsd = RMSD(prediction, lig_coords_gt)
        print('Ligand RMSD: ', f'{lig_rmsd:.2f}')
        pocket_mask = p1lig_lig_pocket_mask | p2lig_lig_pocket_mask
        sm_rmsd = RMSD(prediction[pocket_mask], lig_coords_gt[pocket_mask])
        # compute DockQ score
        from DockQ.dockq_util import cal_dockq
        try:
            fnat, irms, Lrms, DockQ = cal_dockq(
                complex_save_path,
                os.path.join(UNBOUND_FOLDER, name, 'gt_complex.pdb'))
        except Exception as e:
            print(e)
            DockQ = 0.
        # print(f'Predicted Aligned Error: {p2_rmsd_pred:.2f}, GT Aligned Error: {p2_rmsd_gt:.2f}, DockQ: {DockQ:.2f}')
        results.append(dict(
            fnat=fnat,
            irms=irms,
            Lrms=Lrms,
            dockq=DockQ,
            pred_p2_rmsd=p2_rmsd_pred,
            gt_p2_rmsd=p2_rmsd_gt,
            lig_rmsd=lig_rmsd,
            sm_rmsd=sm_rmsd,
            overall_rmsd=overall_rmsd,
        ))
        # # save results
        # results_save_folder = 'output/pred_protac_noGT_noCorrect'
        # os.makedirs(results_save_folder, exist_ok=True)
        # shutil.copyfile(complex_save_path, os.path.join(results_save_folder, f'{name}_DockQ{DockQ*100:.2f}.pdb'))

    # sort results by pred_p2_rmsd
    results = sorted(results, key=lambda x: x['pred_p2_rmsd'])
    # results = results[:30]
    return results



@torch.no_grad()
def predict_one_bound(name, cfg=None, seed_num=1):
    UNBOUND = False
    print(f'--{name}--')
    model = cfg.nn_model
    ds_cfg = cfg.test_dataloader.dataset

    lig_path = os.path.join(PDB_PATH, name, 'ligand.sdf')
    p1_path = os.path.join(PDB_PATH, name, 'protein1.cif')
    p2_path = os.path.join(PDB_PATH, name, 'protein2.pdb')

    # lig_path = f'data/PDBBind/{name}/{name}_ligand.mol2'
    # p1_path = f'data/PDBBind/{name}/{name}_protein_processed.pdb'
    # p2_path = f'data/PDBBind/{name}/{name}_protein_processed.pdb'

    lig_origin = read_molecule(lig_path, sanitize=True, remove_hs=ds_cfg.remove_h)
    assert lig_origin is not None, f'{name} ligand is None'
    # if lig_origin is None:
    #     return [{'fnat': 0, 'irms': 0, 'Lrms': 0, 'dockq': 0, 'pred_p2_rmsd': 0, 'gt_p2_rmsd': 0, 'lig_rmsd': 0, 'sm_rmsd': 0, 'overall_rmsd': 0}]

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(
        p1_path)
    p1_graph_origin = get_rec_graph(recs, recs_coords, c_alpha_coords, n_coords, c_coords,
                            use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
                            surface_max_neighbors=ds_cfg.surface_max_neighbors,
                            surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
                            surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
                            c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,)

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(
        p2_path)
    p2_graph_origin = get_rec_graph(recs, recs_coords, c_alpha_coords, n_coords, c_coords,
                            use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
                            surface_max_neighbors=ds_cfg.surface_max_neighbors,
                            surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
                            surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
                            c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,)
    protein2_coords_gt = deepcopy(p2_graph_origin.ndata['x'])

    results = []
    for seed in range(seed_num):
        lig = deepcopy(lig_origin)
        p1_graph = deepcopy(p1_graph_origin)
        p2_graph = deepcopy(p2_graph_origin)
        lig_id = name.split('_')[-1]
        assert len(lig_id) == 3
        lig, lig_graph = get_lig_graph_revised(
            lig, name=name, max_neighbors=ds_cfg.lig_max_neighbors,
            ideal_path=f'{IDEAL_PATH}/{lig_id}_ideal.sdf',
            # ideal_path=None,
            use_random_coords=False,
            seed=seed,
            use_rdkit_coords=True, radius=ds_cfg.lig_graph_radius)
        geometry_graph = get_geometry_graph_ring(lig)

        # find unbound pocket
        lig_coods_gt = deepcopy(lig_graph.ndata['x'])

        p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = get_pocket_and_mask(
            lig_coods_gt, p1_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
        p2lig_lig_pocket_coords_origin, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = get_pocket_and_mask(
            lig_coods_gt, p2_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
        assert torch.allclose(p1lig_lig_pocket_coords, lig_coods_gt[p1lig_lig_pocket_mask])

        p1lig_lig_pocket_coords = lig_coods_gt[p1lig_lig_pocket_mask]
        p1lig_p1_pocket_coords = deepcopy(p1lig_lig_pocket_coords)

        p2lig_lig_pocket_coords_origin = lig_coods_gt[p2lig_lig_pocket_mask]

        data = dict(
            lig_graph=lig_graph,
            rec_graph=p1_graph,
            rec2_graph=p2_graph,
            geometry_graph=geometry_graph,
            complex_name=[name],
            rec2_coords=[protein2_coords_gt],
            p1lig_p1_pocket_mask=[p1lig_p1_pocket_mask],
            p1lig_p1_pocket_coords=[p1lig_p1_pocket_coords],
            p1lig_lig_pocket_mask=[p1lig_lig_pocket_mask],
            p1lig_lig_pocket_coords=[p1lig_lig_pocket_coords],
            p2lig_p2_pocket_mask=[p2lig_p2_pocket_mask],
            p2lig_p2_pocket_coords=[p2lig_lig_pocket_coords_origin],
            p2lig_lig_pocket_mask=[p2lig_lig_pocket_mask],
            p2lig_lig_pocket_coords=[p2lig_lig_pocket_coords_origin],
            # for PDBBind
            lig_coords=[lig_coods_gt],
            rec_coords=[p1_graph.ndata['x']],
            lig_pocket_coords=[p1lig_lig_pocket_coords],
            rec_pocket_coords=[p1lig_p1_pocket_coords],
        )

        # random move lig
        rot_T, rot_b = random_rotation_translation(translation_distance=5)
        if use_rdkit_coords:
            lig_coords_to_move = data['lig_graph'].ndata['new_x']
        else:
            lig_coords_to_move = data['lig_graph'].ndata['x']
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        data['lig_graph'].ndata['new_x'] = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        data['p1lig_lig_pocket_coords'][0] = (rot_T @ (data['p1lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b
        data['p2lig_lig_pocket_coords'][0] = (rot_T @ (data['p2lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b

        # random move p2
        protein2_rot_T, protein2_rot_b = random_rotation_translation(translation_distance=5)
        protein2_coord_to_move = data['rec2_graph'].ndata['x']
        protein2_mean_to_remove = protein2_coord_to_move.mean(dim=0, keepdims=True)
        data['rec2_graph'].ndata['x'] = (protein2_rot_T @ (protein2_coord_to_move - protein2_mean_to_remove).T).T + protein2_rot_b
        data['p2lig_p2_pocket_coords'][0] = (protein2_rot_T @ (data['p2lig_p2_pocket_coords'][0] - protein2_mean_to_remove).T).T + protein2_rot_b
        data['rec2_coords_input'] = [data['rec2_graph'].ndata['x']]

        with torch.no_grad():
            model_outputs = model(**data, mode='predict')[0]

        prediction = model_outputs['ligs_coords_pred']
        p2_to_p1_rotation = model_outputs['rotation_2']
        p2_to_p1_translation = model_outputs['translation_2']
        p2_rmsd_pred = model_outputs['p2_rmsd_pred'].item()
        lig_coords_gt = data['lig_graph'].ndata['x'].to(prediction.device)

        # rot, trans, is_nan = kabsch(prediction, lig_coords_gt)
        # kabsch_lig_pred = rotate_and_translate(prediction, rot, trans)
        # print('Kabsch RMSD: ', f'{RMSD(kabsch_lig_pred, lig_coords_gt):.2f}')

        # run correction
        if CORRECT_LIGAND:
            prediction = correct_ligand(prediction, lig_graph.ndata['new_x'], lig)
            prediction = torch.from_numpy(prediction).float()

        # save predict ligand
        lig_path = os.path.join(PDB_PATH, name, 'ligand.pdb')
        lig_pred_path = os.path.join(cfg.tmp_dir.name, f'lig_pred_{name}_{seed}.pdb')
        pred_lig = set_new_coords(lig_path, prediction.cpu().numpy(), is_ligand=True)
        pred_lig = set_new_chain(pred_lig, 'A', is_ligand=True)
        write_pdb(pred_lig, lig_pred_path)

        # calculate the rotate and translation to move p2 from gt to pred
        p2_gt = data['rec2_coords'][0].to(prediction.device)
        p2_pred = rotate_and_translate(data['rec2_coords_input'][0].to(prediction.device),
                                        p2_to_p1_rotation, p2_to_p1_translation)
        p1_gt = p1_graph.ndata['x']
        overall_rmsd = RMSD(torch.cat((p1_gt, p2_gt), dim=0), torch.cat((p1_gt, p2_pred), dim=0))
        p2_rmsd_gt = RMSD(p2_gt, p2_pred)
        p2_rot, p2_trans, _ = kabsch(p2_gt, p2_pred)

        # save predict complex
        p1_path = os.path.join(PDB_PATH, name, 'protein1.pdb')
        p2_path = os.path.join(PDB_PATH, name, 'protein2.pdb')
        p2_pred_file = os.path.join(cfg.tmp_dir.name, f'p2_pred_{name}_{seed}.pdb')
        p2_pdb = PandasPdb().read_pdb(p2_path)
        p2_gt_coords = p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
        p2_pred_coords = rotate_and_translate(p2_gt_coords, p2_rot.cpu().numpy(), p2_trans.cpu().numpy())
        p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = p2_pred_coords
        p2_pdb.to_pdb(p2_pred_file, records=['ATOM'], gz=False)
        complex_save_path = os.path.join(cfg.tmp_dir.name, f'complex_pred_{name}_{seed}.pdb')
        merge_pdbs(
            pdb_files=[p1_path, p2_pred_file, lig_pred_path],
            path=complex_save_path,
        )

        lig_rmsd = RMSD(prediction, lig_coords_gt)
        print('Ligand RMSD: ', f'{lig_rmsd:.2f}')
        pocket_mask = p1lig_lig_pocket_mask | p2lig_lig_pocket_mask
        sm_rmsd = RMSD(prediction[pocket_mask], lig_coords_gt[pocket_mask])
        # compute DockQ score
        from DockQ.dockq_util import cal_dockq
        try:
            fnat, irms, Lrms, DockQ = cal_dockq(
                complex_save_path,
                os.path.join(PDB_PATH, name, 'gt_complex.pdb'))
        except Exception as e:
            print(e)
            DockQ = 0.
        results.append(dict(
            fnat=fnat,
            irms=irms,
            Lrms=Lrms,
            dockq=DockQ,
            pred_p2_rmsd=p2_rmsd_pred,
            gt_p2_rmsd=p2_rmsd_gt,
            lig_rmsd=lig_rmsd,
            sm_rmsd=sm_rmsd,
            overall_rmsd=overall_rmsd,
        ))
        # # save results
        # results_save_folder = 'output/pred_mg_noGT_Correct'
        # os.makedirs(results_save_folder, exist_ok=True)
        # shutil.copyfile(complex_save_path, os.path.join(results_save_folder, f'{name}_DockQ{DockQ*100:.2f}.pdb'))

    # sort results by pred_p2_rmsd
    results = sorted(results, key=lambda x: x['pred_p2_rmsd'])
    return results


if __name__ == '__main__':
    args = parse_arguments()
    if args.config is None:
        files = os.listdir(args.work_dir)
        files = [i for i in files if i.endswith('.py')]
        assert len(files) == 1, 'more than one config file found'
        args.config = os.path.join(args.work_dir, files[0])
    if args.checkpoint is None:
        with open(os.path.join(args.work_dir, 'last_checkpoint'), 'r') as f:
            args.checkpoint = f.read().strip()
    cfg = Config.fromfile(args.config)
    cfg.model.noise_initial = 0  # disable noise for test
    cfg_str = cfg.pretty_text
    cfg.tmp_dir = tempfile.TemporaryDirectory()

    device = 'cuda'
    cfg.device = device
    cfg.model.device = device
    cfg.checkpoint = args.checkpoint
    cfg.nn_model = None    # will be created after loading the ligand
    model: torch.nn.Module = get_model(cfg, pretrained=cfg.checkpoint, device=cfg.device)
    model.to(cfg.device)
    model.device = device
    model.eval()
    cfg.nn_model = model

    test_file = cfg.test_dataloader.dataset.complex_names_path
    UNBOUND = True
    SEED_NUM = 40
    if 'MolecularGlue' in test_file:
        UNBOUND = False
        SEED_NUM = 1
        # print('Only test representative for efficiency.')
        # test_file = 'data/MolecularGlue/test_representative.txt'
    # test_file = 'data/timesplit_no_lig_or_rec_overlap_val_sdfAvaliable'
    # test_file = 'data/PROTAC/protac_new.txt'
    with open(test_file, 'r') as f:
        complex_names = f.read().splitlines()
    # complex_names = [i for i in complex_names if i.endswith('RAP')]
    # complex_names = [
    #     '6BOY_B_C_RN6',
    #     # '6HR2_B_A_FWZ',
    #     # '6HAY_F_E_FX8',
    #     # '6W7O_C_A_TL7',
    #     # '7KHH_C_D_WEP',
    #     ]

    tt = time.time()
    if THREAD_NUM <= 1:
        if UNBOUND:
            results = [predict_one_unbound(name, cfg, seed_num=SEED_NUM) for name in complex_names]
        else:
            results = [predict_one_bound(name, cfg, seed_num=SEED_NUM) for name in complex_names]
    else:
        with multiprocessing.Pool(THREAD_NUM) as pool:
            if UNBOUND:
                results = pool.starmap(predict_one_unbound, [(name, cfg, SEED_NUM) for name in complex_names])
            else:
                results = pool.starmap(predict_one_bound, [(name, cfg, SEED_NUM) for name in complex_names])
    tt = time.time() - tt
    print('average time:', tt / len(complex_names))

    pred_dockq_scores = []
    best_dockq_scores = []
    best_ranks = []
    best_sm_rmsds = []
    sm_rmsd_ranks = []
    dockqs = []
    p2_rmsds = []
    sm_rmsds = []
    p2_rmsds_gt = []
    for result in results:
        result = sorted(result, key=lambda x: x['pred_p2_rmsd'])
        dockqs.append([i['dockq'] for i in result])
        p2_rmsds.append([i['pred_p2_rmsd'] for i in result])
        sm_rmsds.append([i['sm_rmsd'] for i in result])
        p2_rmsds_gt.append([i['gt_p2_rmsd'] for i in result])

    p2_rmsds = np.array(p2_rmsds)
    result_shape = p2_rmsds.shape
    p2_rmsds = p2_rmsds.flatten()
    dockqs = np.array(dockqs).flatten()
    p2_rmsds_gt = np.array(p2_rmsds_gt).flatten()

    # antique hit rate
    hit_num_list = []
    hit_rate_list = []
    for name, result in zip(complex_names, results):
        fnat = np.array([i['fnat'] for i in result])
        irms = np.array([i['irms'] for i in result])
        Lrms = np.array([i['Lrms'] for i in result])
        labels = [classify_prediction(x, y, z) for x,y,z in zip(fnat, irms, Lrms)]
        labels = np.array(labels)
        acc = (labels == 1).sum()
        medium = (labels == 2).sum()
        high = (labels == 3).sum()
        hit_num_list.append(f'{acc}/{medium}/{high}')
        hit_all = (labels != 0).sum()
        hit_rate = hit_all / len(labels)
        hit_rate_list.append(hit_rate)

    # generate result table
    rmsd_acc_rate_list = []
    for name, result in zip(complex_names, results):
        dockqs = np.array([i['dockq'] for i in result])
        pred_dockq_scores.append(dockqs[0])  # alread sorted
        best_dockq_scores.append(np.max(dockqs))
        curr_sm_rmsds = [i['sm_rmsd'] for i in result]
        sm_rmsd_ranks.append(np.argmin(curr_sm_rmsds) + 1)
        best_sm_rmsds.append(np.min(curr_sm_rmsds))
        if np.max(dockqs) < 0.23:
            best_rank = None
        else:
            best_rank = np.argmax(dockqs >= 0.23) + 1   # start from 1
        best_ranks.append(best_rank)
        acc_count = ((np.array([i['overall_rmsd'] for i in result])) < 10.).sum()
        rmsd_acc_rate_list.append(acc_count / len(result))

    table = PrettyTable()
    table.field_names = ['pred dockq', 'best dockq', 'best rank', 'best SmRMSD', 'sm rank', 'RMSD<10', 'hit_count', 'hit_rate']
    for pred_dockq, best_dockq, best_rank, best_sm_rmsd, sm_rank, rmsd_acc_rate, hit_count, hit_rate in zip(
        pred_dockq_scores, best_dockq_scores, best_ranks, best_sm_rmsds, sm_rmsd_ranks, rmsd_acc_rate_list, hit_num_list, hit_rate_list):
        table.add_row([f'{pred_dockq:.4f}', f'{best_dockq:.4f}', best_rank, f'{best_sm_rmsd:.4f}', sm_rank,
        f'{rmsd_acc_rate:.4f}', hit_count, f'{hit_rate:.4f}'])
    # add split and mean of each column
    table.add_row(['--mean--'] * 8)
    best_ranks_avaliable = [i for i in best_ranks if i is not None]
    table.add_row([f'{np.mean(pred_dockq_scores):.4f}', f'{np.mean(best_dockq_scores):.4f}', f'{np.mean(best_ranks_avaliable):.4f}',
                   f'{np.mean(best_sm_rmsds):.4f}', f'{np.mean(sm_rmsd_ranks):.4f}',
                   f'{np.mean(rmsd_acc_rate_list):.4f}', '', f'{np.mean(hit_rate_list):.4f}'])
    print(table)
    # print('mean best dockq score:', np.mean(best_dockq_scores))
    acceptable_rate = (np.array(best_dockq_scores) > 0.23).sum() / len(best_dockq_scores)
    print('Acceptable Rate: ', acceptable_rate)

    # save result to target_folder
    result_file = os.path.join(args.work_dir, f'result_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt')
    with open(result_file, 'w') as f:
        f.write(f'UNBOUND: {UNBOUND}\n')
        f.write(f'SEED_NUM: {SEED_NUM}\n')
        for k, v in args._get_kwargs():
            f.write(f'{k}: {v}\n')
        f.write('\n')
        f.write(str(table))
        f.write('\n')
        f.write(f'Acceptable Rate: {acceptable_rate}\n')
        f.write(cfg_str)

    cfg.tmp_dir.cleanup()
