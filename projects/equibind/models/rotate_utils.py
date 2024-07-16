import torch


def kabsch(from_points: torch.Tensor, dst_points: torch.Tensor):
    is_nan = False
    from_mean = from_points.mean(dim=0, keepdim=True)  # (1,3)
    dst_mean = dst_points.mean(dim=0, keepdim=True)
    A = (dst_points - dst_mean).T @ (from_points - from_mean)

    # if torch.isnan(A).any():
    #     log(complex_names[idx], 'complex_names where Nan encountered')
    if torch.isnan(A).any():
        # print('max_values: ', torch.max(dst_points), torch.max(from_points))
        is_nan = True
    # assert not torch.isnan(A).any()
    if torch.isinf(A).any():
        # log(complex_names[idx], 'complex_names where inf encountered')
        is_nan = True
    # assert not torch.isinf(A).any()
    if is_nan:  # fake result
        rotation = dst_mean * 0.
        translation = dst_mean * 0.
        return rotation, translation, is_nan

    U, S, Vt = torch.linalg.svd(A)
    num_it = 0
    while torch.min(S) < 1e-3 or torch.min(
            torch.abs((S ** 2).view(1, 3) - (S ** 2).view(3, 1) + torch.eye(3).to(A.device))) < 1e-2:
        # if self.debug: log('S inside loop ', num_it, ' is ', S, ' and A = ', A)
        A = A + torch.rand(3, 3).to(A.device) * torch.eye(3).to(A.device)
        U, S, Vt = torch.linalg.svd(A)
        num_it += 1
        if num_it > 10: raise Exception('SVD was consitantly unstable')

    corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=A.device))
    rotation = (U @ corr_mat) @ Vt

    translation = dst_mean - torch.t(rotation @ from_mean.t())
    return rotation, translation, is_nan


def rotate_and_translate(points: torch.Tensor, rot:torch.Tensor, trans:torch.Tensor):
    assert points.shape[1] == 3
    assert rot.shape == (3, 3)
    assert trans.shape == (1, 3)
    return (rot @ points.T).T + trans


def reverse_rotate_and_translate(points: torch.Tensor, rot:torch.Tensor, trans:torch.Tensor):
    """
    This is used to reverse the translation cause by `rot` and `trans`.
    >>> rot_T, rot_b = random_rotation_translation(translation_distance=5)
    >>> A = torch.rand(32, 3)
    >>> B = (rot_T @ A.T).T + rot_b
    >>> A_reverse = (rot_T.T @ (B-rot_b).T).T
    >>> (A - A_reverse).mean()
    tensor(3.8805e-11)
    """
    return rotate_and_translate(points - trans, rot=rot.T, trans=trans * 0.)


def model_kabsch(from_points, dst_points, num_att_heads=1, complex_names='', device=None):
    """
    migrate from equibind model to simplify the code.
    move lig to rec
    """
    assert len(from_points) > 0
    assert len(dst_points) > 0
    if device is None:
        device = from_points.device
    rec_keypts = dst_points
    lig_keypts = from_points
    ## Apply Kabsch algorithm
    rec_keypts_mean = rec_keypts.mean(dim=0, keepdim=True)  # (1,3)
    lig_keypts_mean = lig_keypts.mean(dim=0, keepdim=True)  # (1,3)

    A = (rec_keypts - rec_keypts_mean).transpose(0, 1) @ (lig_keypts - lig_keypts_mean) / float(
        num_att_heads)  # 3, 3
    if torch.isnan(A).any():
        print(complex_names, 'complex_names where Nan encountered')
    assert not torch.isnan(A).any()
    if torch.isinf(A).any():
        print(complex_names, 'complex_names where inf encountered')
    assert not torch.isinf(A).any()

    U, S, Vt = torch.linalg.svd(A)
    num_it = 0
    while torch.min(S) < 1e-3 or torch.min(
            torch.abs((S ** 2).view(1, 3) - (S ** 2).view(3, 1) + torch.eye(3).to(device))) < 1e-2:
        A = A + torch.rand(3, 3).to(device) * torch.eye(3).to(device)
        U, S, Vt = torch.linalg.svd(A)
        num_it += 1
        if num_it > 10: raise Exception('SVD was consitantly unstable')

    corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=device))
    rotation = (U @ corr_mat) @ Vt

    translation = rec_keypts_mean - torch.t(rotation @ lig_keypts_mean.t())  # (1,3)
    return rotation, translation


def rigid_align_batch_pockets(old_batch_all_coords, batch_pocket_masks: list, batch_pocket_coords):
    """
    Used to align the pockets of a batch of molecules.
    Args:
        old_batch_all_coords: the original coordinates of the whole molecule.
        batch_pocket_masks: a list of masks, each mask is a list of bools, indicating whether the atom is in the pocket.
        batch_pocket_coords: a list of coordinates, each coordinate is a tensor of shape (num_pocket_atoms, 3).
    Output:
        new_batch_all_coords: the new coordinates of the whole molecule.
    ```
    R, t = model_kabsch(from_points=p2lig_lig_pocket_coords, dst_points=lig_graph.ndata['new_x'][p2lig_lig_pocket_mask])
    lig2_coord = rotate_and_translate(p2lig_lig_pocket_coords, R, t)
    lig_graph.ndata['new_x'][p2lig_lig_pocket_mask] = lig2_coord
    ```
    """
    assert isinstance(batch_pocket_masks, (list, tuple))
    assert isinstance(batch_pocket_coords, (list, tuple))
    assert len(batch_pocket_masks) == len(batch_pocket_coords)

    start_idx = 0
    for pocket_mask, pocket_coord in zip(batch_pocket_masks, batch_pocket_coords):
        assert pocket_mask.sum() == len(pocket_coord)
        end_idx = start_idx + len(pocket_mask)
        R, t = model_kabsch(from_points=pocket_coord, dst_points=old_batch_all_coords[start_idx:end_idx][pocket_mask])
        old_batch_all_coords[start_idx:end_idx][pocket_mask] = rotate_and_translate(pocket_coord, R, t)
        start_idx = end_idx

    return old_batch_all_coords


def batch_align_lig_to_pocket(old_batch_all_coords, batch_pocket_masks: list, batch_pocket_coords):
    """
    Used to align the whole ligand to satisfy the pocket.
    """
    assert isinstance(batch_pocket_masks, (list, tuple))
    assert isinstance(batch_pocket_coords, (list, tuple))
    assert len(batch_pocket_masks) == len(batch_pocket_coords)

    start_idx = 0
    for pocket_mask, pocket_coord in zip(batch_pocket_masks, batch_pocket_coords):
        assert pocket_mask.sum() == len(pocket_coord)
        end_idx = start_idx + len(pocket_mask)
        R, t = model_kabsch(from_points=old_batch_all_coords[start_idx:end_idx][pocket_mask], dst_points=pocket_coord)
        old_batch_all_coords[start_idx:end_idx] = rotate_and_translate(old_batch_all_coords[start_idx:end_idx], R, t)
        start_idx = end_idx

    return old_batch_all_coords