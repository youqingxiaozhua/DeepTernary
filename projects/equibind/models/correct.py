import os
from copy import deepcopy

import numpy as np
import torch
from rdkit.Chem import RemoveHs
from rdkit.Geometry import Point3D

from projects.equibind.models.geometry_utils import (
    apply_changes,
    get_dihedral_vonMises,
    get_torsions,
    rigid_transform_Kabsch_3D,
)
from projects.equibind.models.losses import compute_body_intersection_loss
from projects.equibind.models.pdb_utils import (
    get_pdb_coords,
    read_pdb,
    set_new_coords,
    write_pdb,
)


def correct_ligand(prediction, rdkit_coords, lig):
    """Correct model predict ligand atom coords.
    Args:
        prediction (torch.Tensor): Predicted ligand atom coords
        rdkit_coords (np.ndarray): Ligand atom coords from rdkit
        lig_keypts (torch.Tensor): Ligand keypoint coords
        rec_keypts (torch.Tensor): Receptor keypoint coords
        name (str): Complex name
    """

    lig_rdkit = deepcopy(lig)
    conf = lig_rdkit.GetConformer()
    for i in range(lig_rdkit.GetNumAtoms()):
        x, y, z = rdkit_coords[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    lig_rdkit = RemoveHs(lig_rdkit)

    lig = RemoveHs(lig)
    lig_equibind = deepcopy(lig)
    conf = lig_equibind.GetConformer()
    for i in range(lig_equibind.GetNumAtoms()):
        x, y, z = prediction[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    coords_pred = lig_equibind.GetConformer().GetPositions()

    Z_pt_cloud = coords_pred
    rotable_bonds = get_torsions([lig_rdkit])
    new_dihedrals = np.zeros(len(rotable_bonds))
    for idx, r in enumerate(rotable_bonds):
        new_dihedrals[idx] = get_dihedral_vonMises(lig_rdkit, lig_rdkit.GetConformer(), r,
                                                   Z_pt_cloud)
    optimized_mol = apply_changes(lig_rdkit, new_dihedrals, rotable_bonds)

    coords_pred_optimized = optimized_mol.GetConformer().GetPositions()
    try:
        R, t = rigid_transform_Kabsch_3D(coords_pred_optimized.T, coords_pred.T)
    except Exception as e:
        print(e)
        return prediction.numpy()
    coords_pred_optimized = (R @ (coords_pred_optimized).T).T + t.squeeze()

    return coords_pred_optimized


def fix_clashes(rec_coords, lig_coords):
    """rigid move lig_coords to avoid clashes with rec_coords."""
    assert isinstance(rec_coords, torch.Tensor)
    assert isinstance(lig_coords, torch.Tensor)

    def get_rot_mat(euler_angles):
        roll = euler_angles[0]
        yaw = euler_angles[1]
        pitch = euler_angles[2]

        tensor_0 = torch.zeros([], device=euler_angles.device)
        tensor_1 = torch.ones([], device=euler_angles.device)
        cos = torch.cos
        sin = torch.sin

        RX = torch.stack([
            torch.stack([tensor_1, tensor_0, tensor_0]),
            torch.stack([tensor_0, cos(roll), -sin(roll)]),
            torch.stack([tensor_0, sin(roll), cos(roll)])
        ]).reshape(3, 3)

        RY = torch.stack([
            torch.stack([cos(pitch), tensor_0, sin(pitch)]),
            torch.stack([tensor_0, tensor_1, tensor_0]),
            torch.stack([-sin(pitch), tensor_0, cos(pitch)])
        ]).reshape(3, 3)

        RZ = torch.stack([
            torch.stack([cos(yaw), -sin(yaw), tensor_0]),
            torch.stack([sin(yaw), cos(yaw), tensor_0]),
            torch.stack([tensor_0, tensor_0, tensor_1])
        ]).reshape(3, 3)

        R = torch.mm(RZ, RY)
        R = torch.mm(R, RX)
        return R

    euler_angles_finetune = torch.zeros([3], requires_grad=True, device=rec_coords.device)
    translation_finetune = torch.zeros([3], requires_grad=True, device=rec_coords.device)
    ligand_th = (get_rot_mat(euler_angles_finetune) @ lig_coords.T).T + translation_finetune

    ## Optimize the non-intersection loss:
    non_int_loss_item = 100.
    it = 0
    while non_int_loss_item > 0.5 and it < 2000:
        non_int_loss = compute_body_intersection_loss(ligand_th, rec_coords, sigma=8, surface_ct=8)
        non_int_loss_item = non_int_loss.item()
        eta = 1e-3
        if non_int_loss < 2.:
            eta = 1e-4
        if it > 1500:
            eta = 1e-2
        if it % 100 == 0:
            print(it, ' ', non_int_loss_item)
        non_int_loss.backward()
        translation_finetune = translation_finetune - eta * translation_finetune.grad.detach()
        # translation_finetune = torch.tensor(translation_finetune, requires_grad=True)
        translation_finetune = translation_finetune.clone().detach().requires_grad_(True)

        euler_angles_finetune = euler_angles_finetune - eta * euler_angles_finetune.grad.detach()
        # euler_angles_finetune = torch.tensor(euler_angles_finetune, requires_grad=True)
        euler_angles_finetune = euler_angles_finetune.clone().detach().requires_grad_(True)

        rot = get_rot_mat(euler_angles_finetune)
        ligand_th = (rot @ lig_coords.T).T + translation_finetune

        it += 1

    return ligand_th.detach(), rot, translation_finetune


if __name__ == "__main__":
    # name = "4YTM_F_G_F6A"

    # pdb = read_pdb('output/lig_visualize_4YTM_F_G_F6A/layer_7.pdb')
    # prediction = get_pdb_coords(pdb, is_ligand=True)
    # rdkit_coords = get_pdb_coords('output/lig_visualize_4YTM_F_G_F6A/rdkit.pdb', is_ligand=True)

    # corrected = correct_ligand(prediction, rdkit_coords, name)

    # pdb = set_new_coords(pdb, corrected, is_ligand=True)
    # write_pdb(pdb, 'output/lig_visualize_4YTM_F_G_F6A/corrected.pdb')

    # test fix_clashes
    pdb = read_pdb('output/btk/complex_pred_9.pdb')
    rec = pdb.df['ATOM'][pdb.df['ATOM']["chain_id"] =="A"]
    rec_coords = rec[['x_coord', 'y_coord', 'z_coord']]
    lig = pdb.df['ATOM'][pdb.df['ATOM']["chain_id"] =="B"]
    lig_coords = lig[['x_coord', 'y_coord', 'z_coord']]
    rec_coords = torch.from_numpy(rec_coords.to_numpy()).float().cuda()
    lig_coords = torch.from_numpy(lig_coords.to_numpy()).float().cuda()
    lig_new_coords, r, t = fix_clashes(rec_coords, lig_coords)

    pdb.df['ATOM'].loc[
        pdb.df['ATOM']["chain_id"] =="B", ['x_coord', 'y_coord', 'z_coord']
        ] = lig_new_coords.cpu().numpy()

    write_pdb(pdb, 'output/btk/complex_pred_9_fix.pdb')
    print()
