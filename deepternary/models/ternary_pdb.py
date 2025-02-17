
import os
import random
import json
from copy import deepcopy

import numpy as np
import torch

from torch.utils.data import Dataset
import mmengine

from mmpretrain.registry import DATASETS

from .geometry_utils import random_rotation_translation, rigid_transform_Kabsch_3D_torch
from .process_mols import get_rdkit_coords, get_receptor, get_pocket_coords, \
    read_molecule, get_rec_graph, get_lig_graph_revised, get_receptor_atom_subgraph, get_lig_structure_graph, \
    get_geometry_graph, get_lig_graph_multiple_conformer, get_geometry_graph_ring, get_pocket_and_mask
from .utils import pmap_multi, read_strings_from_txt
from .path import PDB_PATH, PREPROCESSED_PATH
from .rotate_utils import kabsch


def get_rec_pocket_mask(lig_pocket: torch.Tensor, rec_coords: torch.Tensor, cutoff=8.0):
    """
    Return:
        rec_coords_mask: Bool
    """
    if len(lig_pocket) == 0:
        return torch.zeros(rec_coords.shape[0], dtype=torch.bool)
    dist = torch.cdist(lig_pocket, rec_coords)
    min_dist = dist.min(dim=0)[0]
    return min_dist <= cutoff


def swap_variable(v1, v2):
    return v2, v1


@DATASETS.register_module()
class TernaryPreprocessedDataset(Dataset):
    """Use preprocessed folder, each item is stored as a dict."""
    def __init__(
        self,
        complex_names_path='data/',
        preprocess_sub_path='noHs',
        translation_distance=5.0,
        use_rdkit_coords=True,
        freeze='protein1',
        is_train_data=True,
        min_lig_atoms=10,
        min_pocket_atoms=3,
        use_gt_pocket=False,
        random_flip_proteins=False,
        unrepresent_aug_rate=0.,
        # below is inherit from PDBBind,
        remove_h=False,
        lig_max_neighbors=20,
        lig_graph_radius=30,
        geometry_regularization= False,
        geometry_regularization_ring= False,
        pocket_cutoff=6.0,
        pocket_cutoff_p12=8.0,
        **kwargs
    ):
        super().__init__()
        self.preprocess_sub_path = preprocess_sub_path
        self.translation_distance = translation_distance
        self.use_rdkit_coords = use_rdkit_coords
        assert freeze in ('ligand', 'protein1', 'protein2')
        self.freeze = freeze
        self.use_gt_pocket = use_gt_pocket
        self.random_flip_proteins = random_flip_proteins
        self.is_train_data = is_train_data
        # self.complex_names = [i for i in complex_names if len(i) == len('5T35_DA_D_759')]
        self.unrepresent_aug_rate = unrepresent_aug_rate
        if is_train_data:
            with open(complex_names_path, 'r') as f:
                self.complex_names = json.load(f)
            print(f'There are {len(self.complex_names)} clusters for training.')
        else:
            # self.complex_names = read_strings_from_txt(complex_names_path)
            # support csv file
            self.complex_names = np.loadtxt(complex_names_path, dtype=str, delimiter=',')
            if len(self.complex_names.shape) == 2:
                self.complex_names = self.complex_names[:, 0]
            assert len(self.complex_names.shape) == 1
        # below is inherit from PDBBind
        self.remove_h = remove_h
        self.lig_max_neighbors = lig_max_neighbors
        self.lig_graph_radius = lig_graph_radius
        self.geometry_regularization = geometry_regularization
        self.geometry_regularization_ring = geometry_regularization_ring
        self.pocket_cutoff = pocket_cutoff
        self.pocket_cutoff_p12 = pocket_cutoff_p12
        self.regenerate_rdkit = False

    def __len__(self):
        return len(self.complex_names)
    
    def __getitem__(self, index):
        name = self.complex_names[index]    # 6NWV_C_L_CRS
        if self.is_train_data:
            represent = name['representative']
            all = name['items']

            if self.unrepresent_aug_rate > 0. and random.random() < self.unrepresent_aug_rate:
                name = random.choice(all)
            else:
                if len(represent):
                    name = random.choice(represent)
                else:
                    name = random.choice(all)

        preprocessed_file = os.path.join(PREPROCESSED_PATH, self.preprocess_sub_path, f'{name}.pth')
        data = torch.load(preprocessed_file)
        assert data['name'] == name
        rec_graph = data['p1_graph']
        protein2_graph = data['p2_graph']
        # generate new rdkit conformation every time
        # lig_graph = data['lig_graph']
        # geometry_graph = data['geometry_graph_ring']
        if self.regenerate_rdkit:
            lig_path = os.path.join(PDB_PATH, name, 'ligand_noHs.pdb')
            if not os.path.exists(lig_path):
                lig_path = os.path.join(PDB_PATH, name, 'ligand.pdb')
            lig = read_molecule(lig_path, sanitize=True, remove_hs=self.remove_h)
            lig, lig_graph = get_lig_graph_revised(
                lig, name=name, max_neighbors=self.lig_max_neighbors,
                ideal_path=os.path.join(PDB_PATH, name, 'ligand_ideal.sdf'),
                useRandomCoords=False,
                seed=None,
                use_rdkit_coords=True, radius=self.lig_graph_radius)
            lig_coords_gt = deepcopy(lig_graph.ndata['x'])
            if self.geometry_regularization:
                geometry_graph = get_geometry_graph(lig)
            elif self.geometry_regularization_ring:
                geometry_graph = get_geometry_graph_ring(lig)
            else:
                geometry_graph = None
        else:
            # read from preprocessed file
            # rdkit_data = torch.load(os.path.join(PREPROCESSED_PATH, self.preprocess_sub_path, f'{name}_multiLigand.pth'))
            lig_graphs = data['lig_graphs']
            geometry_graphs = data['geometry_graphs']
            assert len(lig_graphs) == len(geometry_graphs)
            if len(lig_graphs) == 0:
                print(name)
                exit()
            if self.is_train_data:
                i = random.randint(0, len(lig_graphs) - 1)
            else:
                i = 0
            lig_graph = lig_graphs[i]
            geometry_graph = geometry_graphs[i]
        
        protein1_pocket_coords = data['p1_pocket']
        protein2_pocket_coords = data['p2_pocket']

        if self.random_flip_proteins and self.is_train_data and random.random() > 0.5:
        # if True:
            # swap protein1 and protein2
            rec_graph, protein2_graph = protein2_graph, rec_graph
            protein1_pocket_coords, protein2_pocket_coords = protein2_pocket_coords, protein1_pocket_coords
        lig_protein_cutoff = self.pocket_cutoff + 1 + 2  # 1 for ligand hydrogen, 2 for Ca to surface
        p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = get_pocket_and_mask(
            lig_graph.ndata['x'], rec_graph.ndata['x'], cutoff=lig_protein_cutoff)
        p1lig_p1_pocket_coords = deepcopy(p1lig_lig_pocket_coords)
        p2lig_lig_pocket_coords_origin, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = get_pocket_and_mask(
            lig_graph.ndata['x'], protein2_graph.ndata['x'], cutoff=lig_protein_cutoff)
        assert torch.allclose(p1lig_lig_pocket_coords, lig_graph.ndata['x'][p1lig_lig_pocket_mask])
        # assert torch.allclose(protein1_pocket_coords, p1lig_lig_pocket_coords)

        p12_p1_pocket_coords, p12_p1_pocket_mask, p12_p2_pocket_mask = get_pocket_and_mask(
            rec_graph.ndata['x'], protein2_graph.ndata['x'], cutoff=self.pocket_cutoff_p12)


        lig_coords_gt: torch.Tensor = deepcopy(lig_graph.ndata['x'])
        rec_coords_gt = deepcopy(rec_graph.ndata['x'])
        protein2_coords_gt = deepcopy(protein2_graph.ndata['x'])

        # randomly rotate and translate the ligand.
        if self.freeze != 'ligand':
            rot_T, rot_b = random_rotation_translation(translation_distance=self.translation_distance)
            if self.use_rdkit_coords:
                lig_coords_to_move = lig_graph.ndata['new_x']
            else:
                lig_coords_to_move = lig_graph.ndata['x']
            mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
            lig_graph.ndata['new_x'] = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
            # protein1_pocket_ligand = (rot_T @ (protein1_pocket_coords - mean_to_remove).T).T + rot_b
            # protein2_pocket_ligand = (rot_T @ (protein2_pocket_coords - mean_to_remove).T).T + rot_b
            p1lig_lig_pocket_coords = (rot_T @ (p1lig_lig_pocket_coords - mean_to_remove).T).T + rot_b
            p2lig_lig_pocket_coords = (rot_T @ (p2lig_lig_pocket_coords_origin - mean_to_remove).T).T + rot_b
        else:
            raise ValueError('freeze ligand not implemented')
            protein1_pocket_ligand = protein1_pocket_coords
            protein2_pocket_ligand = protein2_pocket_coords

        # Randomly rotate and translate the protein1.
        if self.freeze != 'protein1':
            raise ValueError('p1 need to be freezed')
            rot_T, rot_b = random_rotation_translation(translation_distance=self.translation_distance)
            coords_to_move = rec_graph.ndata['x']
            mean_to_remove = coords_to_move.mean(dim=0, keepdim=True)
            rec_graph.ndata['x'] = (rot_T @ (coords_to_move - mean_to_remove).T).T + rot_b
            protein1_pocket_protein = (rot_T @ (protein1_pocket_coords - mean_to_remove).T).T + rot_b
            # 同时移动 ligand 和 p1，需要 ligand 的 gt 再移动到p1新位置
            # rec_coords_gt = (rot_T @ (rec_coords_gt - mean_to_remove).T).T + rot_b
            # lig_coords_gt = (rot_T @ (lig_coords_gt - mean_to_remove).T).T + rot_b
        else:
            pass
            # protein1_pocket_protein = protein1_pocket_coords
            # p12_pocket_p1 = p12_pocket

        if self.freeze != 'protein2':
            # Randomly rotate and translate the protein2
            protein2_rot_T, protein2_rot_b = random_rotation_translation(translation_distance=self.translation_distance)
            protein2_coord_to_move = protein2_graph.ndata['x']
            protein2_mean_to_remove = protein2_coord_to_move.mean(dim=0, keepdims=True)
            protein2_graph.ndata['x'] = (protein2_rot_T @ (protein2_coord_to_move - protein2_mean_to_remove).T).T + protein2_rot_b
            # protein2_pocket_protein = (protein2_rot_T @ (protein2_pocket_coords - protein2_mean_to_remove).T).T + protein2_rot_b
            # dock ligand to protein2
            # lig_to_p2_coords_gt = (protein2_rot_T @ (lig_coords_gt - protein2_mean_to_remove).T).T + protein2_rot_b
            p2lig_p2_pocket_coords = (protein2_rot_T @ (p2lig_lig_pocket_coords_origin - protein2_mean_to_remove).T).T + protein2_rot_b
            p12_p2_pocket_coords = (protein2_rot_T @ (p12_p1_pocket_coords - protein2_mean_to_remove).T).T + protein2_rot_b
            # p12_pocket_p2 = (protein2_rot_T @ (p12_pocket - protein2_mean_to_remove).T).T + protein2_rot_b
        else:
            raise ValueError('p2 can not be freezed')
            lig_to_p2_coords_gt = lig_coords_gt
            protein2_pocket_protein = protein2_pocket_coords
        
        if self.use_gt_pocket:
            raise NotImplementedError
            lig_graph.ndata['new_x'][ligand1_pocket_mask] = protein1_pocket_ligand
            rdkit_lig2_pocket = lig_graph.ndata['new_x'][ligand2_pocket_mask]
            rot, trans, _ = kabsch(protein2_pocket_ligand, rdkit_lig2_pocket)
            lig_graph.ndata['new_x'][ligand2_pocket_mask] = (rot @ protein2_pocket_ligand.t()).t() + trans

        return {
            'lig_graph': lig_graph,
            'rec_graph': rec_graph,
            'rec2_graph': protein2_graph,
            'lig_coords': lig_coords_gt,
            'rec_coords': rec_coords_gt,
            'rec2_coords': protein2_coords_gt,
            'rec2_coords_input': protein2_graph.ndata['x'],
            # p1lig pocket
            'p1lig_p1_pocket_coords': p1lig_p1_pocket_coords,
            'p1lig_lig_pocket_coords': p1lig_lig_pocket_coords,
            # p2lig pocket
            'p2lig_p2_pocket_coords': p2lig_p2_pocket_coords,
            'p2lig_lig_pocket_coords': p2lig_lig_pocket_coords,
            # p12 pocket
            'p12_p1_pocket_coords': p12_p1_pocket_coords,
            'p12_p2_pocket_coords': p12_p2_pocket_coords,

            'p1lig_p1_pocket_mask': p1lig_p1_pocket_mask,
            'p1lig_lig_pocket_mask': p1lig_lig_pocket_mask,
            'p2lig_lig_pocket_mask': p2lig_lig_pocket_mask,
            'p2lig_p2_pocket_mask': p2lig_p2_pocket_mask,
            'lig_pocket_coords': p1lig_lig_pocket_coords,    # for EquiBind
            'rec_pocket_coords': p1lig_p1_pocket_coords,   # for EquiBind
            # 'lig2_pocket_coords': protein2_pocket_ligand,
            # 'rec2_pocket_coords': protein2_pocket_protein,
            'geometry_graph': geometry_graph,
            'complex_name': name,
        }

