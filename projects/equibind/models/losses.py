from typing import Dict, Union, List
import numpy as np
import torch
from torch import Tensor
import ot
from torch.nn.modules.loss import _Loss
from mmengine import MMLogger
from mmpretrain.registry import MODELS

logger = MMLogger.get_instance('mmengine')


# Ligand residue locations: a_i in R^3. Receptor: b_j in R^3
# Ligand: G_l(x) = -sigma * ln( \sum_i  exp(- ||x - a_i||^2 / sigma)  ), same for G_r(x)
# Ligand surface: x such that G_l(x) = surface_ct
# Other properties: G_l(a_i) < 0, G_l(x) = infinity if x is far from all a_i
# Intersection of ligand and receptor: points x such that G_l(x) < surface_ct && G_r(x) < surface_ct
# Intersection loss: IL = \avg_i max(0, surface_ct - G_r(a_i)) + \avg_j max(0, surface_ct - G_l(b_j))
def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1, 1, 3)) ** 2, dim=2) / float(sigma))  # (m, n)
    return - sigma * torch.log(1e-3 + e.sum(dim=1))


def compute_body_intersection_loss(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma, surface_ct):
    loss = torch.mean(
        torch.clamp(surface_ct - G_fn(bound_receptor_repres_nodes_loc_array, model_ligand_coors_deform, sigma),
                    min=0)) + \
           torch.mean(
               torch.clamp(surface_ct - G_fn(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma),
                           min=0))
    return loss


def compute_sq_dist_mat(X_1, X_2):
    '''Computes the l2 squared cost matrix between two point cloud inputs.
    Args:
        X_1: [n, #features] point cloud, tensor
        X_2: [m, #features] point cloud, tensor
    Output:
        [n, m] matrix of the l2 distance between point pairs
    '''
    n_1, _ = X_1.size()
    n_2, _ = X_2.size()
    X_1 = X_1.view(n_1, 1, -1)
    X_2 = X_2.view(1, n_2, -1)
    squared_dist = (X_1 - X_2) ** 2
    cost_mat = torch.sum(squared_dist, dim=2)
    return cost_mat


def compute_ot_emd(cost_mat, device):
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    a = np.ones([cost_mat.shape[0]]) / cost_mat.shape[0]
    b = np.ones([cost_mat.shape[1]]) / cost_mat.shape[1]
    ot_mat = ot.emd(a=a, b=b, M=cost_mat_detach, numItermax=10000)
    ot_mat_attached = torch.tensor(ot_mat, device=device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)
    return ot_dist, ot_mat_attached


def compute_revised_intersection_loss(lig_coords, rec_coords, alpha = 0.2, beta=8, aggression=0):
    distances = compute_sq_dist_mat(lig_coords,rec_coords)
    if aggression > 0:
        aggression_term = torch.clamp(-torch.log(torch.sqrt(distances)/aggression+0.01), min=1)
    else:
        aggression_term = 1
    distance_losses = aggression_term * torch.exp(-alpha*distances * torch.clamp(distances*4-beta, min=1))
    return distance_losses.sum()


@MODELS.register_module()
class BindingLoss(_Loss):
    def __init__(self, ot_loss_weight=1, intersection_loss_weight=0, intersection_sigma=0, geom_reg_loss_weight=1, loss_rescale=True,
                 lig_coord_loss_weight=1.,
                 intersection_surface_ct=0, key_point_alignmen_loss_weight=0,revised_intersection_loss_weight=0, centroid_loss_weight=0,
                  kabsch_rmsd_weight=0,translated_lig_kpt_ot_loss=False, revised_intersection_alpha=0.1, revised_intersection_beta=8, aggression=0) -> None:
        super(BindingLoss, self).__init__()
        self.lig_coord_loss_weight = lig_coord_loss_weight
        self.ot_loss_weight = ot_loss_weight
        self.intersection_loss_weight = intersection_loss_weight
        self.intersection_sigma = intersection_sigma
        self.revised_intersection_loss_weight =revised_intersection_loss_weight
        self.intersection_surface_ct = intersection_surface_ct
        self.key_point_alignmen_loss_weight = key_point_alignmen_loss_weight
        self.centroid_loss_weight = centroid_loss_weight
        self.translated_lig_kpt_ot_loss= translated_lig_kpt_ot_loss
        self.kabsch_rmsd_weight = kabsch_rmsd_weight
        self.revised_intersection_alpha = revised_intersection_alpha
        self.revised_intersection_beta = revised_intersection_beta
        self.aggression =aggression
        self.loss_rescale = loss_rescale
        self.geom_reg_loss_weight = geom_reg_loss_weight
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, ligs_coords, recs_coords, ligs_coords_pred, ligs_pocket_coords, recs_pocket_coords, ligs_keypts,
                recs_keypts, rotations, translations, geom_reg_loss, device, **kwargs):
        # Compute MSE loss for each protein individually, then average over the minibatch.
        # ligs_coords_loss = 0
        # recs_coords_loss = 0
        # ot_loss = 0
        # intersection_loss = 0
        # intersection_loss_revised = 0
        # keypts_loss = 0
        # centroid_loss = 0
        # kabsch_rmsd_loss = 0
        losses: Dict[str, List[Tensor]] = {
            'ligs_coords_loss': [],
            'recs_coords_loss': [],
            'ot_loss': [],
            'intersection_loss': [],
            'intersection_loss_revised': [],
            'keypts_loss': [],
            'centroid_loss': [],
            'kabsch_rmsd_loss': [],
            'geom_reg_loss': [geom_reg_loss],
        }

        for i, _ in enumerate(ligs_coords_pred):
            ## Compute average MSE loss (which is 3 times smaller than average squared RMSD)
            losses['ligs_coords_loss'].append(self.mse_loss(ligs_coords_pred[i], ligs_coords[i]))

            if self.ot_loss_weight > 0:
                # Compute the OT loss for the binding pocket:
                ligand_pocket_coors = ligs_pocket_coords[i]  ##  (N, 3), N = num pocket nodes
                receptor_pocket_coors = recs_pocket_coords[i]  ##  (N, 3), N = num pocket nodes

                ## (N, K) cost matrix
                if self.translated_lig_kpt_ot_loss:
                    cost_mat_ligand = compute_sq_dist_mat(receptor_pocket_coors, (rotations[i] @ ligs_keypts[i].t()).t() + translations[i] )
                else:
                    cost_mat_ligand = compute_sq_dist_mat(ligand_pocket_coors, ligs_keypts[i])
                cost_mat_receptor = compute_sq_dist_mat(receptor_pocket_coors, recs_keypts[i])

                ot_dist, _ = compute_ot_emd(cost_mat_ligand + cost_mat_receptor, device)
                # ot_loss += ot_dist
                losses['ot_loss'].append(ot_dist)
            if self.key_point_alignmen_loss_weight > 0:
                keypts_loss = self.mse_loss((rotations[i] @ ligs_keypts[i].t()).t() + translations[i],
                                             recs_keypts[i])
                losses['keypts_loss'].append(keypts_loss)

            if self.intersection_loss_weight > 0:
                intersection_loss = compute_body_intersection_loss(
                    ligs_coords_pred[i],
                    recs_coords[i],
                    self.intersection_sigma,
                    self.intersection_surface_ct)
                losses['intersection_loss'].append(intersection_loss)

            if self.revised_intersection_loss_weight > 0:
                intersection_loss_revised = compute_revised_intersection_loss(
                    ligs_coords_pred[i], recs_coords[i],
                    alpha=self.revised_intersection_alpha, beta=self.revised_intersection_beta,
                    aggression=self.aggression)
                losses['intersection_loss_revised'].append(intersection_loss_revised)

            if self.kabsch_rmsd_weight > 0:
                lig_coords_pred = ligs_coords_pred[i]
                lig_coords = ligs_coords[i]
                lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
                lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

                A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

                U, S, Vt = torch.linalg.svd(A)

                corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
                rotation = (U @ corr_mat) @ Vt
                translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
                kabsch_rmsd_loss = self.mse_loss((rotation @ lig_coords.t()).t() + translation, lig_coords_pred)
                losses['kabsch_rmsd_loss'].append(kabsch_rmsd_loss)

            centroid_loss = self.mse_loss(ligs_coords_pred[i].mean(dim=0), ligs_coords[i].mean(dim=0))
            losses['centroid_loss'].append(centroid_loss)

        if self.loss_rescale:
            # ligs_coords_loss = ligs_coords_loss / float(len(ligs_coords_pred))
            # ot_loss = ot_loss / float(len(ligs_coords_pred))
            # intersection_loss = intersection_loss / float(len(ligs_coords_pred))
            # keypts_loss = keypts_loss / float(len(ligs_coords_pred))
            # centroid_loss = centroid_loss / float(len(ligs_coords_pred))
            # kabsch_rmsd_loss = kabsch_rmsd_loss / float(len(ligs_coords_pred))
            # intersection_loss_revised = intersection_loss_revised / float(len(ligs_coords_pred))
            # geom_reg_loss = geom_reg_loss / float(len(ligs_coords_pred))
            for k, v in losses.items():
                if len(v) > 0:
                    losses[k] = sum(v) / float(len(v))
                else:   # empty
                    logger.warning(f'Empty values in loss: {k}')
                    losses[k] = 0. * centroid_loss

        losses['ot_loss'] *= self.ot_loss_weight
        losses['intersection_loss'] *= self.intersection_loss_weight
        losses['keypts_loss'] *= self.key_point_alignmen_loss_weight
        losses['centroid_loss'] *= self.centroid_loss_weight
        losses['kabsch_rmsd_loss'] *= self.kabsch_rmsd_weight
        losses['intersection_loss_revised'] *= self.revised_intersection_loss_weight
        losses['geom_reg_loss'] *= self.geom_reg_loss_weight

        return losses

        loss = ligs_coords_loss + self.ot_loss_weight * ot_loss + self.intersection_loss_weight * intersection_loss + keypts_loss * self.key_point_alignmen_loss_weight + centroid_loss * self.centroid_loss_weight + kabsch_rmsd_loss * self.kabsch_rmsd_weight + intersection_loss_revised *self.revised_intersection_loss_weight + geom_reg_loss*self.geom_reg_loss_weight
        return loss, {'ligs_coords_loss': ligs_coords_loss, 'recs_coords_loss': recs_coords_loss, 'ot_loss': ot_loss,
                      'intersection_loss': intersection_loss, 'keypts_loss': keypts_loss, 'centroid_loss:': centroid_loss, 'kabsch_rmsd_loss': kabsch_rmsd_loss, 'intersection_loss_revised': intersection_loss_revised, 'geom_reg_loss': geom_reg_loss}


@MODELS.register_module()
class TripletBindingLoss(BindingLoss):
    def __init__(self, *args, rec2_coord_weight=0, p2_rmsd_pred_weight=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec2_coord_weight = rec2_coord_weight
        self.p2_rmsd_pred_weight = p2_rmsd_pred_weight
        self.l1_loss = torch.nn.L1Loss()

    def forward(
            self, 
            # pred
            ligs_coords_pred, geom_reg_loss,
            ligs_keypts, recs_keypts, rotations, translations,
            ligs_keypts_2, recs_keypts_2, rotations_2, translations_2, valid, p2_rmsd_preds,
            # gt
            ligs_coords, recs_coords, rec2_coords, rec2_coords_input,
            ligs_pocket_coords, recs_pocket_coords,
            ligs_pocket_coords_2, rec2_pocket_coords_2,
            **kwargs):
        # Compute MSE loss for each protein individually, then average over the minibatch.
        if not isinstance(geom_reg_loss, (tuple, list)):
            assert isinstance(geom_reg_loss, torch.Tensor)
            geom_reg_loss = [geom_reg_loss]
        losses: Dict[str, List[float]] = {
            'ligs_coords_loss': [],
            # 'recs_coords_loss': [],
            'ot_loss': [],
            'intersection_loss': [],
            'intersection_loss_revised': [],
            'keypts_loss': [],
            'centroid_loss': [],
            'kabsch_rmsd_loss': [],
            # 'kabsch_rmsd_loss_last2': [],
            'p2_rmsd_pred_loss': [],
            'geom_reg_loss': geom_reg_loss,
            # new added for triplet
            'ot_loss_2': [],
            'intersection_loss_2': [],
            'intersection_loss_revised_2': [],
            'keypts_loss_2': [],
            'rec2_coords_loss': [],
        }
        rec2_coords_rmsds = []

        for i, _ in enumerate(ligs_coords_pred):
            if not valid[i]:
                continue
            ## Compute average MSE loss (which is 3 times smaller than average squared RMSD)
            losses['ligs_coords_loss'].append(self.mse_loss(ligs_coords_pred[i], ligs_coords[i]))

            if self.ot_loss_weight > 0:
                # Compute the OT loss for the binding pocket:
                ligand_pocket_coors = ligs_pocket_coords[i]  ##  (N, 3), N = num pocket nodes
                receptor_pocket_coors = recs_pocket_coords[i]  ##  (N, 3), N = num pocket nodes

                if len(ligand_pocket_coors) < 3 or len(receptor_pocket_coors) < 3:
                    # skip calculating ot_loss when pocket number < 3
                    ot_dist = (ligs_keypts[i].sum() + recs_keypts[i].sum()) * 0.
                else:
                    ## (N, K) cost matrix
                    if self.translated_lig_kpt_ot_loss:  # False
                        cost_mat_ligand = compute_sq_dist_mat(receptor_pocket_coors, (rotations[i] @ ligs_keypts[i].t()).t() + translations[i] )
                    else:
                        cost_mat_ligand = compute_sq_dist_mat(ligand_pocket_coors, ligs_keypts[i])
                    cost_mat_receptor = compute_sq_dist_mat(receptor_pocket_coors, recs_keypts[i])

                    ot_dist, _ = compute_ot_emd(cost_mat_ligand + cost_mat_receptor, device=ligand_pocket_coors.device)
                losses['ot_loss'].append(ot_dist)

                ### new for triplet loss
                ligand_pocket_coors_2 = ligs_pocket_coords_2[i]
                receptor_pocket_coors_2 = rec2_pocket_coords_2[i]
                if len(ligand_pocket_coors_2) < 3 or len(receptor_pocket_coors_2) < 3:
                    ot_dist_2 = (ligs_keypts_2[i].sum() + recs_keypts_2[i].sum()) * 0.
                else:
                    if self.translated_lig_kpt_ot_loss:
                        cost_mat_ligand_2 = compute_sq_dist_mat(receptor_pocket_coors_2, (rotations_2[i] @ ligs_keypts_2[i].t()).t() + translations_2[i])
                    else:
                        cost_mat_ligand_2 = compute_sq_dist_mat(ligand_pocket_coors_2, ligs_keypts_2[i])
                    cost_mat_receptor_2 = compute_sq_dist_mat(receptor_pocket_coors_2, recs_keypts_2[i])
                    ot_dist_2, _ = compute_ot_emd(cost_mat_ligand_2 + cost_mat_receptor_2, device=ligand_pocket_coors_2.device)
                losses['ot_loss_2'].append(ot_dist_2)

            keypts_loss = self.mse_loss((rotations[i] @ ligs_keypts[i].t()).t() + translations[i],
                                            recs_keypts[i])
            losses['keypts_loss'].append(keypts_loss)

            ### new for triplet loss
            keypts_loss_2 = self.mse_loss((rotations_2[i] @ ligs_keypts_2[i].t()).t() + translations_2[i],
                                            recs_keypts_2[i])
            losses['keypts_loss_2'].append(keypts_loss_2)

            if self.intersection_loss_weight > 0:
                intersection_loss = compute_body_intersection_loss(
                    ligs_coords_pred[i],
                    recs_coords[i],
                    self.intersection_sigma,
                    self.intersection_surface_ct)
                losses['intersection_loss'].append(intersection_loss)

                ### new for triplet loss
                intersection_loss_2 = compute_body_intersection_loss(
                    ligs_coords_pred[i],
                    rec2_coords[i],
                    self.intersection_sigma,
                    self.intersection_surface_ct)
                losses['intersection_loss_2'].append(intersection_loss_2)

            intersection_loss_revised = compute_revised_intersection_loss(
                ligs_coords_pred[i], recs_coords[i],
                alpha=self.revised_intersection_alpha, beta=self.revised_intersection_beta,
                aggression=self.aggression)
            losses['intersection_loss_revised'].append(intersection_loss_revised)

            ### new for triplet loss
            intersection_loss_revised_2 = compute_revised_intersection_loss(
                ligs_coords_pred[i], rec2_coords[i],
                alpha=self.revised_intersection_alpha, beta=self.revised_intersection_beta,
                aggression=self.aggression)
            losses['intersection_loss_revised_2'].append(intersection_loss_revised_2)

            if self.kabsch_rmsd_weight > 0:
                lig_coords_pred = ligs_coords_pred[i]
                lig_coords = ligs_coords[i]
                lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
                lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

                A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

                U, S, Vt = torch.linalg.svd(A)

                corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
                rotation = (U @ corr_mat) @ Vt
                translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
                kabsch_rmsd_loss = self.mse_loss((rotation @ lig_coords.t()).t() + translation, lig_coords_pred)
                losses['kabsch_rmsd_loss'].append(kabsch_rmsd_loss)

                # # last 2
                # pred = ligs_evolved_last2[i]
                # gt = ligs_coords[i]
                # pred_mean = pred.mean(dim=0, keepdim=True)  # (1,3)
                # gt_mean = gt.mean(dim=0, keepdim=True)  # (1,3)
                # A = (pred - pred_mean).transpose(0, 1) @ (gt - gt_mean)
                # U, S, Vt = torch.linalg.svd(A)
                # corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
                # rotation = (U @ corr_mat) @ Vt
                # translation = pred_mean - torch.t(rotation @ gt_mean.t())  # (1,3)
                # kabsch_rmsd_loss = self.mse_loss((rotation @ gt.t()).t() + translation, pred)
                # losses['kabsch_rmsd_loss_last2'].append(kabsch_rmsd_loss)

            rec2_coords_pred = (rotations_2[i] @ rec2_coords_input[i].t()).t() + translations_2[i]
            rec2_coords_loss = self.mse_loss(rec2_coords_pred,
                                            rec2_coords[i])
            losses['rec2_coords_loss'].append(rec2_coords_loss)
            rec2_coords_rmsd = self.l1_loss((rotations_2[i] @ rec2_coords_input[i].t()).t() + translations_2[i],
                                            rec2_coords[i]).mean()
            # losses['p2_rmsd_pred_loss'].append(self.mse_loss(rec2_coords_rmsd, p2_rmsd_preds[i]))
            losses['p2_rmsd_pred_loss'].append(self.l1_loss(rec2_coords_rmsd, p2_rmsd_preds[i]))
            rec2_coords_rmsds.append(rec2_coords_rmsd)  # ranges from 0-35

            centroid_loss = self.mse_loss(ligs_coords_pred[i].mean(dim=0), ligs_coords[i].mean(dim=0))
            losses['centroid_loss'].append(centroid_loss)

        # # weight p2_rmsd_pred_loss
        # for i in range(len(losses['p2_rmsd_pred_loss'])):
        #     losses['p2_rmsd_pred_loss'][i] *= rec2_coords_rmsds[i] / 5.
        if self.loss_rescale:
            for k, v in losses.items():
                if len(v) > 0:
                    losses[k] = sum(v) / float(len(v))
                else:   # empty
                    print(f'Empty values in loss: {k}')
                    losses[k] = 0. * p2_rmsd_preds[0].sum()

        losses['p2_rmsd_pred_loss'] *= self.p2_rmsd_pred_weight
        losses['ligs_coords_loss'] *= self.lig_coord_loss_weight
        losses['ot_loss'] *= self.ot_loss_weight
        losses['intersection_loss'] *= self.intersection_loss_weight
        losses['keypts_loss'] *= self.key_point_alignmen_loss_weight
        losses['centroid_loss'] *= self.centroid_loss_weight
        losses['kabsch_rmsd_loss'] *= self.kabsch_rmsd_weight
        losses['intersection_loss_revised'] *= self.revised_intersection_loss_weight
        losses['geom_reg_loss'] *= self.geom_reg_loss_weight

        ### new for triplet loss
        losses['ot_loss_2'] *= self.ot_loss_weight
        losses['intersection_loss_2'] *= self.intersection_loss_weight
        losses['keypts_loss_2'] *= self.key_point_alignmen_loss_weight
        losses['intersection_loss_revised_2'] *= self.revised_intersection_loss_weight
        losses['rec2_coords_loss'] *= self.rec2_coord_weight

        return losses


