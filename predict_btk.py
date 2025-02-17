"""predict CRBN and BTK under different length of linkers"""
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
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from biopandas.pdb import PandasPdb
from mmengine.config import Config
from mmengine.runner import Runner
from mmpretrain.apis.model import get_model
from scipy.optimize import linear_sum_assignment


from deepternary.models.geometry_utils import rigid_transform_Kabsch_3D, get_torsions, get_dihedral_vonMises, \
    apply_changes, \
    random_rotation_translation
from deepternary.models.process_mols import read_molecule, get_lig_graph_revised, \
    get_rec_graph, get_geometry_graph, get_geometry_graph_ring, \
    get_receptor_inference, get_multiple_lig_graph_revised
from deepternary.models.rotate_utils import kabsch, reverse_rotate_and_translate, rotate_and_translate
from deepternary.models.path import PDB_PATH

from deepternary.models.ternary_pdb import TernaryPreprocessedDataset, get_pocket_and_mask
from deepternary.models.pdb_utils import get_pdb_coords, combine_multiple_pdb, set_new_coords, write_pdb, merge_pdbs
from deepternary.models.correct import correct_ligand, fix_clashes

run_corrections = False
realign_after_correction = False  # 效果不好
FIX_CLASHES = False

SEED_NUM = 40


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('work_dir', help='saved model directory')
    p.add_argument('--config', help='test config file path', default=None)
    p.add_argument('--checkpoint', help='checkpoint file', default=None)
    args = p.parse_args()
    return args


def get_unbound_matches(protac_path, lig1_path, lig2_path):
    """
    generate rdkit coords with unbound pocket constrained
    :param name:
    :return: protac_lig1_mask, protac_lig2_mask, lig1_mask, lig2_mask
        protac_lig1_matches: [(protac_idx, lig1_idx)]
    """
    protac_coords = get_pdb_coords(protac_path, is_ligand=True)
    lig1_coords = get_pdb_coords(lig1_path, is_ligand=True)
    lig2_coords = get_pdb_coords(lig2_path, is_ligand=True)

    def get_matches(A, B, threshold=1):
        distance = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
        row_ind, col_ind = linear_sum_assignment(distance)
        matches = [(i, j) for i, j in zip(row_ind, col_ind) if distance[i, j] < threshold]
        return matches

    protac_lig1_matches = get_matches(protac_coords, lig1_coords)
    protac_lig2_matches = get_matches(protac_coords, lig2_coords)

    return protac_lig1_matches, protac_lig2_matches


def get_rdkit_matches(protac_path, lig1_path, lig2_path):
    """Use rdkit to get matches"""
    protac = read_molecule(protac_path, sanitize=True, remove_hs=True)
    lig1 = read_molecule(lig1_path, sanitize=True, remove_hs=True)
    lig2 = read_molecule(lig2_path, sanitize=True, remove_hs=True)

    lig1_matches = protac.GetSubstructMatch(lig1)
    protac_lig1_matches = [(p, l) for l, p in enumerate(lig1_matches)]

    lig2_matches = protac.GetSubstructMatch(lig2)
    protac_lig2_matches = [(p, l) for l, p in enumerate(lig2_matches)]

    return protac_lig1_matches, protac_lig2_matches


def predict_one_unbound(name, lig_path, p1_path, p2_path, lig1_mask_path, lig2_mask_path,
        unbound_lig1_path, unbound_lig2_path, cfg):
    """

    :param lig1_mask_path: pdb file to rigid align with protac, could delete some atoms,
        but no more modify individual atom coords
    :param unbound_lig1_path: unbound lig1, extract lig1 coords
    :return:
    """
    print(f'--{name}--')
    os.makedirs(cfg.save_dir, exist_ok=True)
    model = cfg.nn_model
    ds_cfg = cfg.test_dataloader.dataset

    lig_origin = read_molecule(lig_path, sanitize=True, remove_hs=ds_cfg.remove_h)
    conf = lig_origin.GetConformer()
    protac_coords = torch.from_numpy(conf.GetPositions()).float()

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(
        p1_path)
    p1_graph_origin = get_rec_graph(recs, recs_coords, c_alpha_coords, n_coords, c_coords,
                                    use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
                                    surface_max_neighbors=ds_cfg.surface_max_neighbors,
                                    surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
                                    surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
                                    c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors, )

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(
        p2_path)
    p2_graph_origin = get_rec_graph(recs, recs_coords, c_alpha_coords, n_coords, c_coords,
                                    use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
                                    surface_max_neighbors=ds_cfg.surface_max_neighbors,
                                    surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
                                    surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
                                    c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors, )

    protac_lig1_matches, protac_lig2_matches = get_rdkit_matches(lig_path, lig1_mask_path, lig2_mask_path)
    assert len(protac_lig1_matches) and len(protac_lig2_matches), f'no matches found for {name}'
    # ATTENTION: lig1_mask and unbound_lig1 may have different number of atoms!!! (lig1_mask may be a subset of unbound_lig1)
    # for lig1
    unbound_lig1 = read_molecule(unbound_lig1_path, sanitize=True, remove_hs=True)
    lig1_mask = read_molecule(lig1_mask_path, sanitize=True, remove_hs=True)
    matches = unbound_lig1.GetSubstructMatch(lig1_mask)
    lig1_mask_coords = torch.from_numpy(unbound_lig1.GetConformer().GetPositions()[matches, :]).float()
    unbd_lig1_coords = lig1_mask_coords
    # for lig2
    unbound_lig2 = read_molecule(unbound_lig2_path, sanitize=True, remove_hs=True)
    lig2_mask = read_molecule(lig2_mask_path, sanitize=True, remove_hs=True)
    matches = unbound_lig2.GetSubstructMatch(lig2_mask)
    lig2_mask_coords = torch.from_numpy(unbound_lig2.GetConformer().GetPositions()[matches, :]).float()
    unbd_lig2_coords = lig2_mask_coords
    """
    lig1_pocket_coords: lig1 order
    lig1_pocket_mask: length = lig1_len
    """
    lig1_pocket_coords, lig1_pocket_mask, p1lig_p1_pocket_mask = get_pocket_and_mask(
        unbd_lig1_coords, p1_graph_origin.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
    lig2_pocket_coords, lig2_pocket_mask, p2lig_p2_pocket_mask = get_pocket_and_mask(
        unbd_lig2_coords, p2_graph_origin.ndata['x'], cutoff=ds_cfg.pocket_cutoff)

    # convert lig1_pocket_coords to PROTAC order (p1lig_lig_pocket_coords, p1lig_lig_pocket_mask
    protac_pocket_coords1 = torch.zeros_like(protac_coords)
    p1lig_lig_pocket_mask = torch.zeros_like(protac_coords[:, 0], dtype=torch.bool)
    for (protac_idx, lig1_idx) in protac_lig1_matches:
        protac_pocket_coords1[protac_idx] = unbd_lig1_coords[lig1_idx]
        p1lig_lig_pocket_mask[protac_idx] = True
    p1lig_lig_pocket_coords = protac_pocket_coords1[p1lig_lig_pocket_mask]

    protac_pocket_coords2 = torch.zeros_like(protac_coords)
    p2lig_lig_pocket_mask = torch.zeros_like(protac_coords[:, 0], dtype=torch.bool)
    for (protac_idx, lig2_idx) in protac_lig2_matches:
        protac_pocket_coords2[protac_idx] = unbd_lig2_coords[lig2_idx]
        p2lig_lig_pocket_mask[protac_idx] = True
    p2lig_lig_pocket_coords = protac_pocket_coords2[p2lig_lig_pocket_mask]

    assert p1lig_lig_pocket_mask.sum() > 3
    assert p2lig_lig_pocket_mask.sum() > 3

    lig = deepcopy(lig_origin)
    lig, lig_graph = get_lig_graph_revised(
        lig, name=name, max_neighbors=ds_cfg.lig_max_neighbors,
        ideal_path=None,
        # ideal_path=None,
        use_random_coords=False,
        seed=0,
        use_rdkit_coords=True, radius=ds_cfg.lig_graph_radius)
    geometry_graph = get_geometry_graph_ring(lig)
    lig1 = read_molecule(lig1_mask_path, sanitize=True, remove_hs=True)

    all_results = []
    unclash_results = []
    seed = 0
    while len(all_results) < SEED_NUM:
        print(f'Generating PROTAC conformer {name} with seed {seed}...')
        protac = deepcopy(lig_origin)
        # generate constraint rdkit coords
        # protac = AllChem.AddHs(lig)
        try:
            protac = AllChem.ConstrainedEmbed(protac, lig1, randomseed=seed)
        except ValueError as error:
            assert 'Could not embed molecule' in str(error)
            print(f'ConstrainedEmbed failed: {name}')
            protac = AllChem.AddHs(protac)
            id = AllChem.EmbedMolecule(protac, randomSeed=seed)
            if id < 0:
                print(f'EmbedMolecule failed with Hs: {name}')
                protac = AllChem.RemoveHs(protac)
                id = AllChem.EmbedMolecule(protac, randomSeed=seed)
                if id < 0: 
                    seed += 1
                    continue
            protac = AllChem.RemoveHs(protac)
            conf = protac.GetConformer()
            protac_coords = conf.GetPositions()
            protac_coords = torch.tensor(protac_coords, dtype=torch.float32)
            # align to pocket
            r, t, _ = kabsch(protac_coords[p1lig_lig_pocket_mask], p1lig_lig_pocket_coords)
            protac_coords = rotate_and_translate(protac_coords, r, t)
        else:
            conf = protac.GetConformer()
            protac_coords = conf.GetPositions()
            protac_coords = torch.tensor(protac_coords, dtype=torch.float32)

        lig_graph.ndata['new_x'] = protac_coords

        p1_graph = deepcopy(p1_graph_origin)
        p2_graph = deepcopy(p2_graph_origin)

        data = dict(
            lig_graph=lig_graph,
            rec_graph=p1_graph,
            rec2_graph=p2_graph,
            geometry_graph=geometry_graph,
            complex_name=[name],
            rec2_coords=[p2_graph.ndata['x']],
            rec2_coords_input=[p2_graph.ndata['x']],
            p1lig_p1_pocket_mask=[p1lig_p1_pocket_mask],
            p1lig_p1_pocket_coords=[p1lig_lig_pocket_coords],
            p1lig_lig_pocket_mask=[p1lig_lig_pocket_mask],
            p1lig_lig_pocket_coords=[p1lig_lig_pocket_coords],
            p2lig_p2_pocket_mask=[p2lig_p2_pocket_mask],
            p2lig_p2_pocket_coords=[p2lig_lig_pocket_coords],
            p2lig_lig_pocket_mask=[p2lig_lig_pocket_mask],
            p2lig_lig_pocket_coords=[p2lig_lig_pocket_coords],
        )

        # there is no need to random remote ligand and p2

        with torch.no_grad():
            model_outputs = model(**data, mode='predict')[0]

        prediction = model_outputs['ligs_coords_pred']
        p2_to_p1_rotation = model_outputs['rotation_2']
        p2_to_p1_translation = model_outputs['translation_2']
        p2_prediction = rotate_and_translate(p2_graph.ndata['x'], p2_to_p1_rotation, p2_to_p1_translation)
        p2_rmsd_pred = model_outputs['p2_rmsd_pred'].item()
        print(f'p2_rmsd_pred: {p2_rmsd_pred:.4f}')
        if run_corrections:
            prediction = correct_ligand(prediction, lig_graph.ndata['new_x'], lig)
            prediction = torch.from_numpy(prediction).float()
            from predict_cpu import RMSD
            print(f'correction changed: {RMSD(model_outputs["ligs_coords_pred"], prediction)}')
            if realign_after_correction:
                # align lig to p1
                r, t, _ = kabsch(prediction[p1lig_lig_pocket_mask], p1lig_lig_pocket_coords)
                prediction = rotate_and_translate(prediction, r, t)
                # align p2 to lig
                curr_p2_poc = rotate_and_translate(p2lig_lig_pocket_coords, p2_to_p1_rotation, p2_to_p1_translation)
                r, t, _ = kabsch(curr_p2_poc, prediction[p2lig_lig_pocket_mask])
                p2_prediction = rotate_and_translate(p2_prediction, r, t)

        # exclude clash
        cdist = torch.cdist(p1_graph.ndata['x'], p2_prediction)
        clash_level = (cdist.min(dim=1).values < 3.8).sum() / cdist.shape[0]

        curr_result = dict(
            clash_level=clash_level,
            prediction=prediction,
            # p2_to_p1_rotation=p2_to_p1_rotation,
            # p2_to_p1_translation=p2_to_p1_translation,
            p2_prediction=p2_prediction,
            rec2_coords=data['rec2_coords'][0],
            pred_p2_rmsd=p2_rmsd_pred,
        )
        if clash_level == 0.:
            unclash_results.append(curr_result)
        all_results.append(curr_result)
        seed += 1

    assert len(all_results)  == SEED_NUM
    # sort results by pred_p2_rmsd
    # if len(unclash_results) > 0:
    #     results = unclash_results[:5]
    # else:
    all_results = sorted(all_results, key=lambda x: x['clash_level'])
    results = all_results
    results = sorted(results, key=lambda x: x['pred_p2_rmsd'])
    print(f'clash levels: {[i["clash_level"].item() for i in results]}')
    print(f'pred_p2_rmsd: {[i["pred_p2_rmsd"] for i in results]}')

    # save the top results
    for i, res in enumerate(results):
        prediction = res['prediction']
        p2_prediction = res['p2_prediction']
        rec2_coords = res['rec2_coords']

        # calculate the rotate and translation to move p2 from gt to pred
        p2_input = rec2_coords.to(prediction.device)
        p2_rot, p2_trans, _ = kabsch(p2_input, p2_prediction)

        # save predict complex
        # for 8QU8
        p1_path = 'output/protac_new/8QU8_A_F_WYL/protein1.pdb'
        p2_path = 'output/protac_new/8QU8_A_F_WYL/protein2.pdb'

        lig_pred_path = os.path.join(cfg.save_dir, f'lig_pred_{name}_{i}.pdb')
        pred_lig = set_new_coords(lig_path, prediction.cpu().numpy(), is_ligand=True)
        write_pdb(pred_lig, lig_pred_path)
        p2_pred_file = os.path.join(cfg.save_dir, f'p2_pred_{name}_{i}.pdb')
        p2_pdb = PandasPdb().read_pdb(p2_path)
        p2_gt_coords = p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
        p2_pred_coords = rotate_and_translate(p2_gt_coords, p2_rot.cpu().numpy(), p2_trans.cpu().numpy())
        p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = p2_pred_coords
        p2_pdb.to_pdb(p2_pred_file, records=['ATOM'], gz=False)
        merge_pdbs(
            pdb_files=[p1_path, p2_pred_file, lig_pred_path],
            path=os.path.join(cfg.save_dir, f'complex_pred_{name}_{i}.pdb'),
        )
        if FIX_CLASHES and results[0]['clash_level'] > 0:
            p1_coords = get_pdb_coords(p1_path, is_ligand=False)
            p1_coords = torch.from_numpy(p1_coords).float().cuda()
            p2_coords = torch.from_numpy(p2_pred_coords).float().cuda()
            p2_pred_coords, _, _ = fix_clashes(p1_coords, p2_coords)
            p2_pred_coords = p2_pred_coords.cpu().numpy()
            p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = p2_pred_coords
            p2_pdb.to_pdb(p2_pred_file, records=['ATOM'], gz=False)
            # TODO: move ligand atoms

            merge_pdbs(
                pdb_files=[p1_path, p2_pred_file, lig_pred_path],
                path=os.path.join(cfg.save_dir, f'complex_pred_{name}_{i}_fix.pdb'),
            )


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

    device = 'cuda'
    cfg.device = device
    cfg.model.device = device
    cfg.checkpoint = args.checkpoint
    cfg.nn_model = None  # will be created after loading the ligand
    model: torch.nn.Module = get_model(cfg, pretrained=cfg.checkpoint, device=cfg.device)
    model.to(cfg.device)
    model.device = device
    model.eval()
    cfg.nn_model = model

    # BTK
    # cfg.save_dir = 'output/btk'
    # complex_names = map(str, [7])
    # # complex_names = map(str, [i+1 for i in range(11)])
    # for name in complex_names:
    #     predict_one_unbound(
    #         name, lig_path=f'protacs/{name}.pdb',
    #         p1_path='unbound_CRBN_4CI3_B.pdb', p2_path='unbound_BTK_5P9J_A.pdb',
    #         lig1_mask_path=f'lig_mask/{name}_lig1.pdb', lig2_mask_path=f'lig_mask/{name}_lig2.pdb',
    #         unbound_lig1_path='Y70.pdb', unbound_lig2_path='8E8.pdb',
    #         cfg=cfg)
    
    # # Affinity
    # data_dir = 'data/Affinity'
    # cfg.save_dir = 'output/affinity'
    # test_list = np.loadtxt('data/Affinity/test_list.csv', dtype=str, delimiter=',', skiprows=1)
    # for item in test_list[15:]:
    #     name, lig_path, p1_path, p2_path, lig1_mask, lig2_mask, unbound_lig1, unbound_lig2 = item
    #     predict_one_unbound(
    #         name, lig_path=f'{data_dir}/{lig_path}',
    #         p1_path=f'{data_dir}/{p1_path}', p2_path=f'{data_dir}/{p2_path}',
    #         lig1_mask_path=f'{data_dir}/{lig1_mask}', lig2_mask_path=f'{data_dir}/{lig2_mask}',
    #         unbound_lig1_path=f'{data_dir}/{unbound_lig1}', unbound_lig2_path=f'{data_dir}/{unbound_lig2}',
    #         cfg=cfg)
    
    # # PROTAC
    # data_dir = 'output/protac22'
    # cfg.save_dir = 'output/test'
    # with open('data/PROTAC/protac22.txt', 'r') as f:
    #     complex_names = f.read().splitlines()
    # for name in complex_names:
    #     predict_one_unbound(
    #         name, lig_path=f'{data_dir}/{name}/ligand.pdb',
    #         p1_path=f'{data_dir}/{name}/unbound_protein1.pdb', p2_path=f'{data_dir}/{name}/unbound_protein2.pdb',
    #         lig1_mask_path=f'{data_dir}/{name}/unbound_lig1.pdb', lig2_mask_path=f'{data_dir}/{name}/unbound_lig2.pdb',
    #         unbound_lig1_path=f'{data_dir}/{name}/unbound_lig1.pdb', unbound_lig2_path=f'{data_dir}/{name}/unbound_lig2.pdb',
    #         cfg=cfg)

    # 8QU8
    data_dir = 'output/protac_new'
    name = '8QU8_A_F_WYL'
    cfg.save_dir = f'output/pred_{name}'
    predict_one_unbound(
        name, lig_path=f'{data_dir}/{name}/ligand.pdb',
        p1_path=f'{data_dir}/{name}/unbound_protein1.pdb', p2_path=f'{data_dir}/{name}/unbound_protein2.pdb',
        lig1_mask_path=f'{data_dir}/{name}/lig1_mask.pdb', lig2_mask_path=f'{data_dir}/{name}/unbound_lig2.pdb',
        unbound_lig1_path=f'{data_dir}/{name}/unbound_lig1.pdb', unbound_lig2_path=f'{data_dir}/{name}/unbound_lig2.pdb',
        cfg=cfg)
