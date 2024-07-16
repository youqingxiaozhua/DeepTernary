"""Use coords directly to calculate a approximate DockQ score."""
import math
from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor


def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    if isinstance(A, Tensor):
        A = A.cpu().numpy()
    if isinstance(B, Tensor):
        B = B.cpu().numpy()
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t


def RMSD(lig_coords_pred, lig_coords):
    return torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1)))


def find_pocket(lig_coords, rec_coords, cutoff=5):
    dist = torch.cdist(lig_coords, rec_coords, p=2)
    lig_poc_id = torch.nonzero(torch.min(dist, dim=1).values <= cutoff).squeeze()
    rec_poc_id = torch.nonzero(torch.min(dist, dim=0).values <= cutoff).squeeze()
    return lig_poc_id, rec_poc_id


def calculate_fnat(native_pocket_lig_ids, native_pocket_rec_ids, model_pocket_lig_ids, model_pocket_rec_ids):
    native_pockets = []
    for i in native_pocket_lig_ids:
        native_pockets.append(f'{i}_lig')
    for i in native_pocket_rec_ids:
        native_pockets.append(f'{i}_rec')
    model_pockets = []
    for i in model_pocket_lig_ids:
        model_pockets.append(f'{i}_lig')
    for i in model_pocket_rec_ids:
        model_pockets.append(f'{i}_rec')
    fnat = len(set(native_pockets) & set(model_pockets)) / len(set(native_pockets))
    return fnat


def cal_approx_dockq(
        native_coords: Tuple[Tensor, Tensor, Tensor],   # lig, p1, p2
        model_coords: Tuple[Tensor, Tensor, Tensor],):
    """
    We assume p1 is the receptor, p2 is the ligand, and p1 is already aligned.
    We also assume lig and p2 are in the same chain, because both of them are predicted
        from the model.
    """
    gt_lig, gt_p1, gt_p2 = native_coords
    pred_lig, pred_p1, pred_p2 = model_coords

    gt_ligand = torch.cat([gt_lig, gt_p2], dim=0)
    gt_rec = gt_p1
    pred_ligand = torch.cat([pred_lig, pred_p2], dim=0)
    pred_rec = pred_p1

    # find pocket
    gt_lig_poc_id, gt_rec_poc_id = find_pocket(gt_ligand, gt_rec)
    pred_lig_poc_id, pred_rec_poc_id = find_pocket(pred_ligand, pred_rec)

    # calculate fnat
    fnat = calculate_fnat(gt_lig_poc_id, gt_rec_poc_id, pred_lig_poc_id, pred_rec_poc_id)

    # calculate lrmsd
    lrmsd = RMSD(gt_ligand, pred_ligand)

    # calculate irmsd
    native_interface = torch.cat([gt_ligand[gt_lig_poc_id], gt_rec[gt_rec_poc_id]], dim=0)
    model_corre_interface = torch.cat([pred_ligand[gt_lig_poc_id], pred_rec[gt_rec_poc_id]], dim=0)
    R, t = rigid_transform_Kabsch_3D(model_corre_interface.T, native_interface.T)
    aligned_coords = ((R @ (model_corre_interface.numpy()).T).T + t.squeeze())
    irmsd = RMSD(native_interface, torch.from_numpy(aligned_coords))

    dockq = (fnat + 1/(1+(irmsd/1.5)*(irmsd/1.5)) + 1/(1+(lrmsd/8.5)*(lrmsd/8.5)))/3
    return fnat, lrmsd, irmsd, dockq
