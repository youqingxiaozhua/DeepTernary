import argparse
from typing import List
from copy import deepcopy
from itertools import repeat
import os
import tempfile
import threading
import time
from datetime import datetime

from prettytable import PrettyTable
import numpy as np
import torch
import dgl
from torch import multiprocessing
from torch.multiprocessing import set_start_method
from rdkit import Chem
from tqdm import tqdm
from biopandas.pdb import PandasPdb
from mmengine.config import Config
from mmengine.runner import Runner
from mmpretrain.apis.model import get_model
from mmpretrain.registry import DATASETS

from rdkit import Chem
from rdkit.Chem import MolFromPDBFile, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


from projects.equibind.models.geometry_utils import rigid_transform_Kabsch_3D, get_torsions, get_dihedral_vonMises, apply_changes, \
    random_rotation_translation
from projects.equibind.models.process_mols import read_molecule, get_lig_graph_revised, \
    get_rec_graph, get_geometry_graph, get_geometry_graph_ring, \
    get_receptor_inference, get_multiple_lig_graph_revised
from projects.equibind.models.rotate_utils import kabsch, reverse_rotate_and_translate, rotate_and_translate
from projects.equibind.models.path import PDB_PATH

from projects.equibind.models.ternary_pdb import TernaryPreprocessedDataset, get_pocket_and_mask
from projects.equibind.models.utils import combine_multiple_pdb, RMSD

SEED_NUM = 5



def get_lig_coords(lig_path):
    lig = read_molecule(lig_path, sanitize=True, remove_hs=True)
    conf = lig.GetConformer()
    true_lig_coords = conf.GetPositions()
    return torch.from_numpy(true_lig_coords).float()


def replace_to_unbound_coords(whole_coords, part_coords, realign=False):
    """
    whole_coords: the whole PROTAC
    part_coords: unbound lig1 or unbound lig2
    realign: realign part_coords to whole_coords
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



def embed_ligand():
    smiles = 'CC(C)S(=O)(=O)C1=CC=CC=C1NC1=NC(NC2=CC=C(N3CCN(CCOCCOCCNC(=O)CNC4=CC=CC5=C4C(=O)N(C4CCC(=O)NC4=O)C5=O)CC3)C=C2)=NC=C1Cl'
    standard_smiles = rdMolStandardize.StandardizeSmiles(smiles)
    mol = Chem.MolFromSmiles(standard_smiles)

    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)

    AllChem.MolToPDBFile(mol, 'tl12186_embed.pdb')


@torch.no_grad()
def predict_one_unbound(name, cfg=None):
    print(f'--{name}--')
    model = cfg.nn_model
    ds_cfg = cfg.test_dataloader.dataset


    UNBOUND_FOLDER = 'output/tl12186'
    # lig_path = os.path.join(UNBOUND_FOLDER, 'tl12186_embed.pdb')
    lig_path = os.path.join('output/tl12186/tl12186_embed_GUI.pdb')
    p1_path = os.path.join(UNBOUND_FOLDER, '4TZ4_C_protein.pdb')
    p2_path = os.path.join(UNBOUND_FOLDER, f'{name}_protein.pdb')

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

    tmp_dir = tempfile.TemporaryDirectory()
    result = []
    for seed in range(SEED_NUM):
        lig = deepcopy(lig_origin)
        p1_graph = deepcopy(p1_graph_origin)
        p2_graph = deepcopy(p2_graph_origin)
        lig, lig_graph = get_lig_graph_revised(
            lig, name=name, max_neighbors=ds_cfg.lig_max_neighbors,
            # ideal_path=f'output/protac22/{name}/ligand_ideal.sdf',
            ideal_path=None,
            use_random_coords=False,
            seed=seed,
            use_rdkit_coords=True, radius=ds_cfg.lig_graph_radius)
        geometry_graph = get_geometry_graph_ring(lig)

        # find unbound pocket
        lig_coods_gt = deepcopy(lig_graph.ndata['x'])


        # unbound anchor and warhead
        lig1_coords = get_lig_coords(os.path.join(UNBOUND_FOLDER, '4TZ4_C_lig.pdb'))
        lig2_coords = get_lig_coords(os.path.join(UNBOUND_FOLDER, f'{name}_lig.pdb'))
        lig_coods_gt, update_mask1 = replace_to_unbound_coords(lig_coods_gt, lig1_coords)
        lig_coods_gt, update_mask2 = replace_to_unbound_coords(lig_coods_gt, lig2_coords)
        pocket_mask = update_mask1 | update_mask2

        p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = get_pocket_and_mask(
            lig_coods_gt, p1_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
        p2lig_lig_pocket_coords_origin, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = get_pocket_and_mask(
            lig_coods_gt, p2_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
        assert torch.allclose(p1lig_lig_pocket_coords, lig_coods_gt[p1lig_lig_pocket_mask])

        p1lig_lig_pocket_mask = p1lig_lig_pocket_mask & pocket_mask
        p1lig_lig_pocket_coords = lig_coods_gt[p1lig_lig_pocket_mask]
        p1lig_p1_pocket_coords = deepcopy(p1lig_lig_pocket_coords)

        p2lig_lig_pocket_mask = p2lig_lig_pocket_mask & pocket_mask
        # for lig2, we align to rdkit coords to avoid reveal gt
        r, t, _ = kabsch(lig_coods_gt[p2lig_lig_pocket_mask], lig_graph.ndata['new_x'][p2lig_lig_pocket_mask])
        p2lig_lig_pocket_coords_origin = rotate_and_translate(lig_coods_gt[p2lig_lig_pocket_mask], r, t)


        pocket_mask = p1lig_lig_pocket_mask | p2lig_lig_pocket_mask
        pockt_rdkit_coords = lig_graph.ndata['new_x'][pocket_mask]
        pocket_gt_coords = lig_graph.ndata['x'][pocket_mask]
        # print('pocket rdkit vs gt: ', RMSD(pockt_rdkit_coords, pocket_gt_coords))


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
        lig_coords_to_move = data['lig_graph'].ndata['new_x']
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
        result.append({
            'data': data,
            'p2_rmsd_pred': p2_rmsd_pred,
            'prediction': prediction,
            'p2_to_p1_rotation': p2_to_p1_rotation,
            'p2_to_p1_translation': p2_to_p1_translation,
        })
    
    # sort result accroding to p2_rmsd_pred
    result.sort(key=lambda x: x['p2_rmsd_pred'])

    print('p2 rmsd pred', [i['p2_rmsd_pred'] for i in result])

    data = result[0]['data']
    prediction = result[0]['prediction']
    p2_to_p1_rotation = result[0]['p2_to_p1_rotation']
    p2_to_p1_translation = result[0]['p2_to_p1_translation']

    p2_gt = data['rec2_coords'][0].to(prediction.device)
    p2_pred = rotate_and_translate(data['rec2_coords_input'][0].to(prediction.device),
                                    p2_to_p1_rotation, p2_to_p1_translation)
    # p2_rmsd_gt = RMSD(p2_gt, p2_pred)
    p2_rot, p2_trans, _ = kabsch(p2_gt, p2_pred)

    # save predict complex
    tmp_dir = tempfile.TemporaryDirectory()
    lig_pred_file = os.path.join(tmp_dir.name, f'lig_pred_{name}.pdb')
    p2_pred_file = os.path.join(tmp_dir.name, f'p2_pred_{name}.pdb')

    lig_pdb = PandasPdb().read_pdb(lig_path)
    lig_pdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']] = prediction.cpu().numpy()
    lig_pdb.to_pdb(lig_pred_file, records=['HETATM'], gz=False)

    p2_pdb = PandasPdb().read_pdb(p2_path)
    p2_gt_coords = p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
    p2_pred_coords = rotate_and_translate(p2_gt_coords, p2_rot.cpu().numpy(), p2_trans.cpu().numpy())
    p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = p2_pred_coords
    p2_pdb.to_pdb(p2_pred_file, records=['ATOM'], gz=False)

    combine_multiple_pdb(
        pdb_files=[p1_path, lig_pred_file, p2_pred_file],
        path=os.path.join('output/tl12186_pred', f'complex_pred_{name}.pdb'),
    )

    tmp_dir.cleanup()


if __name__ == '__main__':
    config = 'output/PP_lay8_catCoord_GtVal/protac.py'
    checkpoint = 'output/PP_lay8_catCoord_GtVal/epoch_1000.pth'
    cfg = Config.fromfile(config)
    cfg.model.noise_initial = 0  # disable noise for test

    device = 'cuda'
    cfg.device = device
    cfg.model.device = device
    cfg.checkpoint = checkpoint
    cfg.nn_model = None    # will be created after loading the ligand
    model: torch.nn.Module = get_model(cfg, pretrained=checkpoint, device=cfg.device)
    model.to(cfg.device)
    model.device = device
    model.eval()
    cfg.nn_model = model

    # predict_one_unbound('ABL2_3HMI_A', cfg=cfg)
    predict_one_unbound('7SJ3_CDK4_A', cfg=cfg)
