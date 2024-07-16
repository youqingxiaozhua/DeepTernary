import random
from typing import List, Tuple

import torch
from torch import Tensor, nn

from mmpretrain.registry import MODELS

from .logger import log
from .rotate_utils import (
    batch_align_lig_to_pocket,
    rigid_align_batch_pockets,
    model_kabsch,
    kabsch,
    rigid_align_batch_pockets,
    rotate_and_translate,
)
from .transformer import MLP, TwoWayTransformer
from .triplet_dock import IEGMN_QueryDecoder, get_two_keys_mask


@MODELS.register_module()
class IEGMN_QueryDecoder_KnownPocket_GtVal(IEGMN_QueryDecoder):
    """
    predict pocket keypoints when training, return gt when testing.
    """

    def __init__(self,
                 *args,
                 lig1_coords_mode='rdkit',
                 lig2_coords_mode='rdkit',
                 lig_kpt_mode_train='mask_pocket',
                 lig_kpt_mode_val='mask_pocket',
                 rec_kpt_mode_train='mask_pocket',
                 rec_kpt_mode_val='mask_pocket',
                 **kwargs):
        """
        align_method: 
            p1-lig-p2: p2 aligns to lig first, and then both p2 and lig align to p1,
            p1-p2: p2 aligns to p1 directly.
        lig1_coords_mode:
            rdkit: use rdkit coords (unchanged)
            pocket: use pocket coords (without align)
            aligned_pocket: use pocket coords (align pocket to rdkit)
        lig_kpt_mode_val:
            mask_pocket: use gt pocket
            mask_predict: use predicted coords with pocket mask
            mask_correct: use predicted coords and correct (only for val/test)
            query_predict: use decoder to predict key points
        rec_kpt_mode_train:
            mask_pocket: use gt pocket
            query_predict: use decoder to predict key points
        """
        super().__init__(*args, **kwargs)
        assert len(self.known_pocket_info) == 6, 'This Decoder only for known pocket'
        assert lig1_coords_mode in ('rdkit', 'pocket', 'aligned_pocket')
        self.lig1_coords_mode = lig1_coords_mode
        assert lig2_coords_mode in ('rdkit', 'aligned_pocket')
        self.lig2_coords_mode = lig2_coords_mode
        assert lig_kpt_mode_train in ('mask_pocket', 'mask_predict', 'query_predict')
        self.lig_kpt_mode_train = lig_kpt_mode_train
        assert lig_kpt_mode_val in ('mask_pocket', 'mask_predict', 'mask_correct', 'query_predict')
        self.lig_kpt_mode_val = lig_kpt_mode_val
        assert rec_kpt_mode_train in ('mask_pocket', 'query_predict')
        self.rec_kpt_mode_train = rec_kpt_mode_train
        assert rec_kpt_mode_val in ('mask_pocket', 'query_predict')
        self.rec_kpt_mode_val = rec_kpt_mode_val

        if lig_kpt_mode_train == 'query_predict':
            assert rec_kpt_mode_train == 'query_predict'
        if lig_kpt_mode_val == 'query_predict':
            assert rec_kpt_mode_val == 'query_predict'

    def forward(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_names, epoch,
                p1lig_p1_pocket_mask, p1lig_p1_pocket_coords, p1lig_lig_pocket_mask, p1lig_lig_pocket_coords,
                p2lig_p2_pocket_mask, p2lig_p2_pocket_coords, p2lig_lig_pocket_mask, p2lig_lig_pocket_coords,
                **kwargs) -> Tuple[List[Tensor], ...]:
        
        p1lig_p1_pocket_mask = [i.to(lig_graph.device) for i in p1lig_p1_pocket_mask]
        p1lig_p1_pocket_coords = [i.to(lig_graph.device) for i in p1lig_p1_pocket_coords]
        p1lig_lig_pocket_mask = [i.to(lig_graph.device) for i in p1lig_lig_pocket_mask]
        p1lig_lig_pocket_coords = [i.to(lig_graph.device) for i in p1lig_lig_pocket_coords]
        p2lig_p2_pocket_mask = [i.to(lig_graph.device) for i in p2lig_p2_pocket_mask]
        p2lig_p2_pocket_coords = [i.to(lig_graph.device) for i in p2lig_p2_pocket_coords]
        p2lig_lig_pocket_mask = [i.to(lig_graph.device) for i in p2lig_lig_pocket_mask]
        p2lig_lig_pocket_coords = [i.to(lig_graph.device) for i in p2lig_lig_pocket_coords]

        SAVE_LIG_RESULT = False
        if SAVE_LIG_RESULT:
            # save lig
            from biopandas.pdb import PandasPdb
            assert len(complex_names) == 1
            name = complex_names[0]
            lig_path = f'output/protac22/{name}/ligand.pdb'
            lig_pdb = PandasPdb().read_pdb(lig_path)
            # save gt
            lig_pdb.df['HETATM'][['x_coord', 'y_coord',
                                  'z_coord']] = lig_graph.ndata['x'].cpu().numpy()
            lig_pdb.to_pdb(f'output/lig_visualize/gt.pdb', records=['HETATM'], gz=False)
            # save rdkit
            lig_pdb.df['HETATM'][['x_coord', 'y_coord',
                                  'z_coord']] = lig_graph.ndata['new_x'].cpu().numpy()
            lig_pdb.to_pdb(f'output/lig_visualize/rdkit.pdb', records=['HETATM'], gz=False)

        # normalize start
        lig_graph.ndata['x'] = self.normalize(lig_graph.ndata['x'])
        lig_graph.ndata['new_x'] = self.normalize(lig_graph.ndata['new_x'])
        rec_graph.ndata['x'] = self.normalize(rec_graph.ndata['x'])
        rec2_graph.ndata['x'] = self.normalize(rec2_graph.ndata['x'])
        # geometry_graph.ndata['x'] = self.normalize(geometry_graph.ndata['x'])
        # normalize end

        h_feats_lig = self.lig_atom_embedder(lig_graph.ndata['feat'])

        if self.use_rec_atoms:
            h_feats_rec = self.rec_embedder(rec_graph.ndata['feat'])
            h_feats_rec2 = self.rec_embedder(rec2_graph.ndata['feat'])
        else:
            h_feats_rec = self.rec_embedder(rec_graph.ndata['feat'])  # (N_res, emb_dim)
            h_feats_rec2 = self.rec_embedder(rec2_graph.ndata['feat'])  # (N_res, emb_dim)

        # add pocket information
        # if 'p1_mask' in self.known_pocket_info:
        p1lig_p1_pocket_mask_batch = torch.cat(p1lig_p1_pocket_mask)
        h_feats_rec[p1lig_p1_pocket_mask_batch] += self.p1lig_pock_gt_embed.weight[1][None, ]
        h_feats_rec[~p1lig_p1_pocket_mask_batch] += self.p1lig_pock_gt_embed.weight[0][None, ]

        p1lig_lig_pocket_mask_batch = torch.cat(p1lig_lig_pocket_mask)
        h_feats_lig[p1lig_lig_pocket_mask_batch] += self.p1lig_pock_gt_embed.weight[1][None, ]
        h_feats_lig[~p1lig_lig_pocket_mask_batch] += self.p1lig_pock_gt_embed.weight[0][None, ]

        if self.lig1_coords_mode == 'rdkit':
            pass
        else:
            # p1lig_lig_pocket_coords = torch.cat(kwargs['p1lig_lig_pocket_coords']).to(
            #     lig_graph.device)
            # p1lig_lig_pocket_coords = self.normalize(p1lig_lig_pocket_coords)
            p1lig_lig_pocket_coords_norm = [self.normalize(i) for i in p1lig_lig_pocket_coords]
            assert len(p1lig_lig_pocket_mask_batch) == len(lig_graph.ndata['new_x'])
            if self.lig1_coords_mode == 'pocket':
                lig_graph.ndata['new_x'][p1lig_lig_pocket_mask_batch] = torch.cat(p1lig_lig_pocket_coords_norm)
            elif self.lig1_coords_mode == 'aligned_pocket':
                lig_graph.ndata['new_x'] = batch_align_lig_to_pocket(
                    old_batch_all_coords=lig_graph.ndata['new_x'],
                    batch_pocket_masks=p1lig_lig_pocket_mask,
                    batch_pocket_coords=p1lig_lig_pocket_coords_norm)
            else:
                raise ValueError(f'Unknown lig1_coords_mode: {self.lig1_coords_mode}')

        # if 'lig2_mask' in self.known_pocket_info:
        p2lig_lig_pocket_mask_batch = torch.cat(p2lig_lig_pocket_mask)
        h_feats_lig[p2lig_lig_pocket_mask_batch] += self.p2lig_pock_gt_embed.weight[1][None, ]
        h_feats_lig[~p2lig_lig_pocket_mask_batch] += self.p2lig_pock_gt_embed.weight[0][None, ]

        # if 'lig2_coord' in self.known_pocket_info:
        if self.lig2_coords_mode == 'rdkit':
            pass
        else:
            # p2lig_lig_pocket_coords = torch.cat(kwargs['p2lig_lig_pocket_coords']).to(
            #     lig_graph.device)
            # p2lig_lig_pocket_coords = self.normalize(p2lig_lig_pocket_coords)
            p2lig_lig_pocket_coords_norm = [self.normalize(i) for i in p2lig_lig_pocket_coords]
            if self.lig2_coords_mode == 'pocket':
                assert ValueError('lig2_coords_mode shoule be aligned')
                lig_graph.ndata['new_x'][p2lig_lig_pocket_mask_batch] = torch.cat(p2lig_lig_pocket_coords_norm)
            elif self.lig2_coords_mode == 'aligned_pocket':
                # align batch gt to input
                lig_graph.ndata['new_x'] = rigid_align_batch_pockets(
                    old_batch_all_coords=lig_graph.ndata['new_x'],
                    batch_pocket_masks=p2lig_lig_pocket_mask,
                    batch_pocket_coords=p2lig_lig_pocket_coords_norm)
            else:
                raise ValueError(f'Unknown lig2_coords_mode: {self.lig2_coords_mode}')

        if 'p2_mask' in self.known_pocket_info:
            p2lig_p2_pocket_mask_batch = torch.cat(p2lig_p2_pocket_mask)
            h_feats_rec2[p2lig_p2_pocket_mask_batch] += self.p2lig_pock_gt_embed.weight[1][None, ]
            h_feats_rec2[~p2lig_p2_pocket_mask_batch] += self.p2lig_pock_gt_embed.weight[0][None, ]

        orig_coords_lig = lig_graph.ndata['new_x']
        orig_coords_rec = rec_graph.ndata['x']
        orig_coords_rec2 = rec2_graph.ndata['x']

        coords_lig = lig_graph.ndata['new_x']
        coords_rec = rec_graph.ndata['x']
        coords_rec2 = rec2_graph.ndata['x']

        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        rand_h_lig = rand_dist.sample([h_feats_lig.size(0),
                                       self.random_vec_dim]).to(h_feats_lig.device)
        rand_h_rec = rand_dist.sample([h_feats_rec.size(0),
                                       self.random_vec_dim]).to(h_feats_lig.device)
        rand_h_rec2 = rand_dist.sample([h_feats_rec2.size(0),
                                        self.random_vec_dim]).to(h_feats_lig.device)
        h_feats_lig = torch.cat([h_feats_lig, rand_h_lig], dim=1)
        h_feats_rec = torch.cat([h_feats_rec, rand_h_rec], dim=1)
        h_feats_rec2 = torch.cat([h_feats_rec2, rand_h_rec2], dim=1)

        # random noise:
        if self.noise_initial > 0 and self.training:
            noise_level = self.noise_initial * \
                self.noise_decay_rate**(epoch + 1)
            h_feats_lig = h_feats_lig + noise_level * \
                torch.randn_like(h_feats_lig)
            h_feats_rec = h_feats_rec + noise_level * \
                torch.randn_like(h_feats_rec)
            h_feats_rec2 = h_feats_rec2 + noise_level * \
                torch.randn_like(h_feats_rec2)
            coords_lig = coords_lig + noise_level * \
                torch.randn_like(coords_lig)
            coords_rec = coords_rec + noise_level * \
                torch.randn_like(coords_rec)
            coords_rec2 = coords_rec2 + noise_level * \
                torch.randn_like(coords_rec2)

        if self.use_mean_node_features:
            h_feats_lig = torch.cat([h_feats_lig, torch.log(lig_graph.ndata['mu_r_norm'])], dim=1)
            h_feats_rec = torch.cat([h_feats_rec, torch.log(rec_graph.ndata['mu_r_norm'])], dim=1)
            h_feats_rec2 = torch.cat(
                [h_feats_rec2, torch.log(rec2_graph.ndata['mu_r_norm'])], dim=1)

        original_ligand_node_features = h_feats_lig
        original_receptor_node_features = h_feats_rec
        original_receptor2_node_features = h_feats_rec2
        lig_graph.edata['feat'] *= self.use_edge_features_in_gmn
        rec_graph.edata['feat'] *= self.use_edge_features_in_gmn
        rec2_graph.edata['feat'] *= self.use_edge_features_in_gmn

        mask_lig_q, mask_rec_q, mask_rec2_q = None, None, None
        if self.cross_msgs:
            # lig <- rec, rec2
            mask_lig_q = get_two_keys_mask(lig_graph.batch_num_nodes(), rec_graph.batch_num_nodes(),
                                           rec2_graph.batch_num_nodes(), self.device)
            # rec <- lig, rec2
            mask_rec_q = get_two_keys_mask(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes(),
                                           rec2_graph.batch_num_nodes(), self.device)
            # rec2 <- lig, rec
            mask_rec2_q = get_two_keys_mask(rec2_graph.batch_num_nodes(),
                                            lig_graph.batch_num_nodes(),
                                            rec_graph.batch_num_nodes(), self.device)

        if self.separate_lig:
            raise NotImplementedError

        geom_losses = 0
        for i, layer in enumerate(self.iegmn_layers):

            if SAVE_LIG_RESULT:
                # save layer
                lig_pdb.df['HETATM'][['x_coord', 'y_coord',
                                      'z_coord']] = self.unnormalize(coords_lig).cpu().numpy()
                lig_pdb.to_pdb(f'output/lig_visualize/layer_{i}.pdb', records=['HETATM'], gz=False)

            coords_lig, h_feats_lig, \
                coords_rec, h_feats_rec, \
                coords_rec2, h_feats_rec2, \
                trajectory, geom_loss = layer(
                    lig_graph=lig_graph,
                    rec_graph=rec_graph,
                    rec2_graph=rec2_graph,
                    coords_lig=coords_lig,
                    h_feats_lig=h_feats_lig,
                    original_ligand_node_features=original_ligand_node_features,
                    orig_coords_lig=orig_coords_lig,
                    coords_rec=coords_rec,
                    h_feats_rec=h_feats_rec,
                    original_receptor_node_features=original_receptor_node_features,
                    orig_coords_rec=orig_coords_rec,
                    coords_rec2=coords_rec2,
                    h_feats_rec2=h_feats_rec2,
                    original_receptor2_node_features=original_receptor2_node_features,
                    orig_coords_rec2=orig_coords_rec2,
                    mask_lig_q=mask_lig_q,
                    mask_rec_q=mask_rec_q,
                    mask_rec2_q=mask_rec2_q,
                    geometry_graph=geometry_graph
                )
            if not self.separate_lig:
                geom_losses = geom_losses + geom_loss
                # full_trajectory.extend(trajectory)

            # if given_gt_pocket:
            # # if 'lig1_coord' in self.known_pocket_info:
            #     p1lig_lig_pocket_mask = torch.cat(kwargs['p1lig_lig_pocket_mask'])
            #     p1lig_lig_pocket_coords = torch.cat(kwargs['p1lig_lig_pocket_coords']).to(lig_graph.device)
            #     p1lig_lig_pocket_coords = self.normalize(p1lig_lig_pocket_coords)
            #     assert len(p1lig_lig_pocket_mask) == len(lig_graph.ndata['new_x'])
            #     lig_graph.ndata['new_x'][p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
            #     coords_lig[p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
            # # if 'lig2_coord' in self.known_pocket_info:
            #     p2lig_lig_pocket_coords = [self.normalize(i).to(lig_graph.device) for i in kwargs['p2lig_lig_pocket_coords']]
            #     # align batch gt to input
            #     lig_graph.ndata['new_x'] = rigid_align_batch_pockets(
            #         old_batch_all_coords=lig_graph.ndata['new_x'],
            #         batch_pocket_masks=kwargs['p2lig_lig_pocket_mask'],
            #         batch_pocket_coords=p2lig_lig_pocket_coords
            #     )
            #     coords_lig = lig_graph.ndata['new_x']

        if self.separate_lig:
            raise NotImplementedError

        rotations = []
        translations = []
        recs_keypts = []
        ligs_keypts = []
        ligs_evolved = []
        ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
        ligs_node_idx.insert(0, 0)
        recs_node_idx = torch.cumsum(rec_graph.batch_num_nodes(), dim=0).tolist()
        recs_node_idx.insert(0, 0)
        # new for triplet
        valid = []  # in case checkpoint return NaN
        rotations_2 = []  # rec2 -> lig
        translations_2 = []
        recs_keypts_2 = []
        ligs_keypts_2 = []
        p2_rmsd_preds = []
        recs_node_idx_2 = torch.cumsum(rec2_graph.batch_num_nodes(), dim=0).tolist()
        recs_node_idx_2.insert(0, 0)

        # TODO: run SVD in batches, if possible
        for idx in range(len(ligs_node_idx) - 1):
            lig_start = ligs_node_idx[idx]
            lig_end = ligs_node_idx[idx + 1]
            rec_start = recs_node_idx[idx]
            rec_end = recs_node_idx[idx + 1]
            rec2_start = recs_node_idx_2[idx]
            rec2_end = recs_node_idx_2[idx + 1]
            # Get H vectors

            rec_feats = h_feats_rec[rec_start:rec_end]  # (m, d)
            # rec_feats_mean = torch.mean(self.h_mean_rec(rec_feats), dim=0, keepdim=True)  # (1, d)
            rec2_feats = h_feats_rec2[rec2_start:rec2_end]  # (m, d)
            # rec2_feats_mean = torch.mean(self.h_mean_rec2(rec2_feats), dim=0, keepdim=True)  # (1, d)
            lig_feats = h_feats_lig[lig_start:lig_end]  # (n, d)
            # lig_feats_mean = torch.mean(self.h_mean_lig(lig_feats), dim=0, keepdim=True)  # (1, d)

            d = lig_feats.shape[1]
            assert d == self.out_feats_dim
            # Z coordinates
            Z_rec_coords = coords_rec[rec_start:rec_end]
            Z_rec2_coords = coords_rec2[rec2_start:rec2_end]
            Z_lig_coords = coords_lig[lig_start:lig_end]

            queries = self.keypts_query_embeddings.weight

            src = torch.cat((lig_feats, rec_feats, rec2_feats), dim=0)
            A, B, C = len(lig_feats), len(rec_feats), len(rec2_feats)
            lig_feats_pe = self.feat_type_embeddings.weight[0].repeat(A, 1)
            rec_feats_pe = self.feat_type_embeddings.weight[1].repeat(B, 1)
            rec2_feats_pe = self.feat_type_embeddings.weight[2].repeat(C, 1)
            src_pe = torch.cat((lig_feats_pe, rec_feats_pe, rec2_feats_pe), dim=0)

            queries, keys = self.kpt_transformer(src[None, ...], src_pe[None, ...], queries[None,
                                                                                            ...])
            queries = queries[0]
            keys = keys[0]

            query_list = []
            for i in range(4):
                start_idx = i * self.num_att_heads
                end_idx = (i + 1) * self.num_att_heads
                query_list.append(queries[start_idx:end_idx])

            p2_rmsd_pred = self.p2_rmsd_pred_head(queries[-1:])[0][0]
            p2_rmsd_preds.append(p2_rmsd_pred)

            keys = keys + src_pe  # only for keys
            lig_feats = keys[:A]
            rec_feats = keys[A:A + B]
            rec2_feats = keys[A + B:]
            assert len(rec2_feats) == C

            if self.unnormalized_kpt_weights:
                lig_scales = self.scale_lig(lig_feats)
                rec_scales = self.scale_rec(rec_feats)
                Z_lig_coords = Z_lig_coords * lig_scales
                Z_rec_coords = Z_rec_coords * rec_scales
                rec2_scales = self.scale_rec2(rec2_feats)
                Z_rec2_coords = Z_rec2_coords * rec2_scales

            # TODO: 只算自己好，还是算上邻近的好
            lig1_keypts = self.keypts_cross_attn[0](query_list[0][None],
                                                    torch.cat((lig_feats, rec_feats))[None],
                                                    torch.cat((Z_lig_coords, Z_rec_coords))[None])
            rec_keypts = self.keypts_cross_attn[1](query_list[1][None],
                                                   torch.cat((rec_feats, lig_feats))[None],
                                                   torch.cat((Z_rec_coords, Z_lig_coords))[None])

            if self.align_method == 'p1-lig-p2':
                lig2_keypts = self.keypts_cross_attn[2](query_list[2][None, ...],
                                                        torch.cat((lig_feats, rec2_feats))[None,
                                                                                           ...],
                                                        torch.cat(
                                                            (Z_lig_coords, Z_rec2_coords))[None,
                                                                                           ...])
                rec2_keypts = self.keypts_cross_attn[3](query_list[3][None, ...],
                                                        torch.cat((rec2_feats, lig_feats))[None,
                                                                                           ...],
                                                        torch.cat(
                                                            (Z_rec2_coords, Z_lig_coords))[None,
                                                                                           ...])
            elif self.align_method == 'p1-p2':
                lig2_keypts = self.keypts_cross_attn[2](query_list[2][None, ...],
                                                        torch.cat((rec_feats, rec2_feats))[None,
                                                                                           ...],
                                                        torch.cat(
                                                            (Z_rec_coords, Z_rec2_coords))[None,
                                                                                           ...])
                rec2_keypts = self.keypts_cross_attn[3](query_list[3][None, ...],
                                                        torch.cat((rec2_feats, rec_feats))[None,
                                                                                           ...],
                                                        torch.cat(
                                                            (Z_rec2_coords, Z_rec_coords))[None,
                                                                                           ...])
            else:
                raise ValueError('Unknown align method: {}'.format(self.align_method))

            # remove batch
            lig_keypts = lig1_keypts[0]
            rec_keypts = rec_keypts[0]
            lig2_keypts = lig2_keypts[0]
            rec2_keypts = rec2_keypts[0]
            # unnormalize start
            lig_keypts, rec_keypts, lig2_keypts, rec2_keypts = self.unnormalize(
                lig_keypts, rec_keypts, lig2_keypts, rec2_keypts)
            Z_lig_coords = self.unnormalize(Z_lig_coords)
            # unnormalize end
            # if not self.training:
            #     # use unbound lig coords
            #     Z_lig_coords[p1lig_lig_pocket_mask[idx]] = p1lig_lig_pocket_coords[idx]
            #     r, t, _ = kabsch(
            #         p2lig_lig_pocket_coords[idx],
            #         Z_lig_coords[p2lig_lig_pocket_mask[idx]])
            #     Z_lig_coords[p2lig_lig_pocket_mask[idx]] = rotate_and_translate(
            #         p2lig_lig_pocket_coords[idx], r, t)
            lig2_mask = p2lig_lig_pocket_mask[idx]
            if self.training:
                zero_grad = (lig_keypts.sum() + lig2_keypts.sum()) * 0.
                if self.lig_kpt_mode_train == 'mask_pocket':
                    lig_keypts = p1lig_lig_pocket_coords[idx].to(
                        lig_graph.device) + zero_grad
                    # lig2 can not directly use gt pocket, because the linker conformation varies
                    lig2_pocket = p2lig_lig_pocket_coords[idx].to(lig_graph.device)
                    R, t = model_kabsch(from_points=lig2_pocket, dst_points=Z_lig_coords[lig2_mask])
                    lig2_keypts = rotate_and_translate(lig2_pocket, R, t)
                elif self.lig_kpt_mode_train == 'mask_predict':
                    lig1_mask = p1lig_lig_pocket_mask[idx]
                    lig_keypts = Z_lig_coords[lig1_mask] + zero_grad
                    lig2_keypts = Z_lig_coords[lig2_mask]
                elif self.lig_kpt_mode_train == 'query_predict':
                    pass
                else:
                    raise ValueError(f'Unknown lig_kpt_mode_train: {self.lig_kpt_mode_train}')
            else:  # test
                if self.lig_kpt_mode_val == 'mask_pocket':
                    lig_keypts = p1lig_lig_pocket_coords[idx].to(lig_graph.device)
                    # lig2 can not directly use gt pocket, because the linker conformation varies
                    lig2_pocket = p2lig_lig_pocket_coords[idx].to(lig_graph.device)
                    R, t = model_kabsch(from_points=lig2_pocket, dst_points=Z_lig_coords[lig2_mask])
                    lig2_keypts = rotate_and_translate(lig2_pocket, R, t)
                elif self.lig_kpt_mode_val == 'mask_predict':
                    lig1_mask = p1lig_lig_pocket_mask[idx]
                    lig_keypts = Z_lig_coords[lig1_mask]
                    lig2_mask = p2lig_lig_pocket_mask[idx]
                    lig2_keypts = Z_lig_coords[lig2_mask]
                elif self.lig_kpt_mode_val == 'mask_correct':
                    # TODO: mask_correct
                    raise NotImplementedError('mask_correct')
                elif self.lig_kpt_mode_val == 'query_predict':
                    pass
                else:
                    raise ValueError(f'Unknown lig_kpt_mode_train: {self.lig_kpt_mode_train}')

            if self.training:
                zero_grad = (rec_keypts.sum() + rec2_keypts.sum()) * 0.
                if self.rec_kpt_mode_train == 'mask_pocket':
                    rec_keypts = p1lig_p1_pocket_coords[idx].to(
                        rec_graph.device) + zero_grad
                    rec2_keypts = p2lig_p2_pocket_coords[idx].to(rec2_graph.device)
                elif self.rec_kpt_mode_train == 'query_predict':
                    pass
                else:
                    raise ValueError(f'Unknown rec_kpt_mode_train: {self.rec_kpt_mode_train}')
            else:
                if self.rec_kpt_mode_val == 'mask_pocket':
                    rec_keypts = p1lig_p1_pocket_coords[idx].to(rec_graph.device)
                    rec2_keypts = p2lig_p2_pocket_coords[idx].to(rec2_graph.device)
                elif self.rec_kpt_mode_val == 'query_predict':
                    pass
                else:
                    raise ValueError(f'Unknown rec_kpt_mode_val: {self.rec_kpt_mode_val}')

            recs_keypts.append(rec_keypts)
            ligs_keypts.append(lig_keypts)
            recs_keypts_2.append(rec2_keypts)
            ligs_keypts_2.append(lig2_keypts)
            if len(lig_keypts) == 0 or len(rec_keypts) == 0 or len(lig2_keypts) == 0 or len(rec2_keypts) == 0:
                valid.append(False)
                rotations.append(None)
                translations.append(None)
                rotations_2.append(None)
                translations_2.append(None)
                ligs_evolved.append(None)
                continue
            valid.append(True)

            # Apply Kabsch algorithm
            rotation, translation = model_kabsch(from_points=lig_keypts,
                                                 dst_points=rec_keypts,
                                                 num_att_heads=self.num_att_heads,
                                                 complex_names=complex_names,
                                                 device=self.device)
            rotations.append(rotation)
            translations.append(translation)
            # (rotation @ evolved_ligs[idx].t()).t() + translation
            moved_lig_coords = (rotation @ Z_lig_coords.T).T + translation
            # # use gt_pocket
            # if not self.training:
            #     Z_lig_coords[p1lig_lig_pocket_mask[idx]] = p1lig_lig_pocket_coords[idx]
            #     r, t = model_kabsch(from_points=p2lig_lig_pocket_coords[idx],
            #                            dst_points=Z_lig_coords[p2lig_lig_pocket_mask[idx]])
            #     Z_lig_coords[p2lig_lig_pocket_mask[idx]] = rotate_and_translate(p2lig_lig_pocket_coords[idx], r, t)

            if self.align_method == 'p1-lig-p2':
                moved_lig2_keypts = (rotation @ lig2_keypts.t()).t() + translation
                rotation2, translation2 = model_kabsch(from_points=rec2_keypts,
                                                       dst_points=moved_lig2_keypts,
                                                       num_att_heads=self.num_att_heads,
                                                       complex_names=complex_names,
                                                       device=self.device)
            elif self.align_method == 'p1-p2':
                rotation2, translation2 = model_kabsch(from_points=rec2_keypts,
                                                       dst_points=lig2_keypts,
                                                       num_att_heads=self.num_att_heads,
                                                       complex_names=complex_names,
                                                       device=self.device)
            else:
                raise ValueError('Unknown align method: {}'.format(self.align_method))
            rotations_2.append(rotation2)
            translations_2.append(translation2)

            ligs_evolved.append(moved_lig_coords)

        # ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss \
        #    ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2
        return ligs_evolved, ligs_keypts, recs_keypts, rotations, translations, geom_losses, \
            ligs_keypts_2, recs_keypts_2, rotations_2, translations_2, valid, p2_rmsd_preds


@MODELS.register_module()
class IEGMN_QueryDecoder_KnownPocket_Predict(IEGMN_QueryDecoder_KnownPocket_GtVal):
    """
    Predict the ligand coords instead of directly use gt
    """

    def forward(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_names, epoch,
                **kwargs) -> Tuple[List[Tensor], ...]:

        if self.training:
            given_gt_pocket = False
        else:
            given_gt_pocket = False

        # # save lig
        # from biopandas.pdb import PandasPdb
        # assert len(complex_names) == 1
        # name = complex_names[0]
        # lig_path = f'output/protac22/{name}/ligand.pdb'
        # lig_pdb = PandasPdb().read_pdb(lig_path)
        # # save gt
        # lig_pdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']] = lig_graph.ndata['x'].cpu().numpy()
        # lig_pdb.to_pdb(f'output/lig_visualize/gt.pdb', records=['HETATM'], gz=False)
        # # save rdkit
        # lig_pdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']] = lig_graph.ndata['new_x'].cpu().numpy()
        # lig_pdb.to_pdb(f'output/lig_visualize/rdkit.pdb', records=['HETATM'], gz=False)

        # normalize start
        lig_graph.ndata['x'] = self.normalize(lig_graph.ndata['x'])
        lig_graph.ndata['new_x'] = self.normalize(lig_graph.ndata['new_x'])
        rec_graph.ndata['x'] = self.normalize(rec_graph.ndata['x'])
        rec2_graph.ndata['x'] = self.normalize(rec2_graph.ndata['x'])
        # geometry_graph.ndata['x'] = self.normalize(geometry_graph.ndata['x'])
        # normalize end

        h_feats_lig = self.lig_atom_embedder(lig_graph.ndata['feat'])

        if self.use_rec_atoms:
            h_feats_rec = self.rec_embedder(rec_graph.ndata['feat'])
            h_feats_rec2 = self.rec_embedder(rec2_graph.ndata['feat'])
        else:
            h_feats_rec = self.rec_embedder(rec_graph.ndata['feat'])  # (N_res, emb_dim)
            h_feats_rec2 = self.rec_embedder(rec2_graph.ndata['feat'])  # (N_res, emb_dim)

        # add pocket information
        if 'p1_mask' in self.known_pocket_info:
            p1lig_p1_pocket_mask = torch.cat(kwargs['p1lig_p1_pocket_mask'])
            h_feats_rec[p1lig_p1_pocket_mask] += self.p1lig_pock_gt_embed.weight[1][None, ]
            h_feats_rec[~p1lig_p1_pocket_mask] += self.p1lig_pock_gt_embed.weight[0][None, ]

        if 'lig1_mask' in self.known_pocket_info:
            p1lig_lig_pocket_mask = torch.cat(kwargs['p1lig_lig_pocket_mask'])
            h_feats_lig[p1lig_lig_pocket_mask] += self.p1lig_pock_gt_embed.weight[1][None, ]
            h_feats_lig[~p1lig_lig_pocket_mask] += self.p1lig_pock_gt_embed.weight[0][None, ]

        # if 'lig1_coord' in self.known_pocket_info:
        #     assert 'lig1_mask' in self.known_pocket_info
        #     p1lig_lig_pocket_mask = torch.cat(kwargs['p1lig_lig_pocket_mask'])
        #     p1lig_lig_pocket_coords = torch.cat(kwargs['p1lig_lig_pocket_coords']).to(lig_graph.device)
        #     p1lig_lig_pocket_coords = self.normalize(p1lig_lig_pocket_coords)
        #     assert len(p1lig_lig_pocket_mask) == len(lig_graph.ndata['new_x'])
        #     lig_graph.ndata['new_x'][p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords

        if 'lig2_mask' in self.known_pocket_info:
            p2lig_lig_pocket_mask = torch.cat(kwargs['p2lig_lig_pocket_mask'])
            h_feats_lig[p2lig_lig_pocket_mask] += self.p2lig_pock_gt_embed.weight[1][None, ]
            h_feats_lig[~p2lig_lig_pocket_mask] += self.p2lig_pock_gt_embed.weight[0][None, ]

        # if 'lig2_coord' in self.known_pocket_info:
        #     p2lig_lig_pocket_coords = [self.normalize(i).to(lig_graph.device) for i in kwargs['p2lig_lig_pocket_coords']]
        #     # align batch gt to input
        #     lig_graph.ndata['new_x'] = rigid_align_batch_pockets(
        #         old_batch_all_coords=lig_graph.ndata['new_x'],
        #         batch_pocket_masks=kwargs['p2lig_lig_pocket_mask'],
        #         batch_pocket_coords=p2lig_lig_pocket_coords
        #     )

        if 'p2_mask' in self.known_pocket_info:
            p2lig_p2_pocket_mask = torch.cat(kwargs['p2lig_p2_pocket_mask'])
            h_feats_rec2[p2lig_p2_pocket_mask] += self.p2lig_pock_gt_embed.weight[1][None, ]
            h_feats_rec2[~p2lig_p2_pocket_mask] += self.p2lig_pock_gt_embed.weight[0][None, ]

        orig_coords_lig = lig_graph.ndata['new_x']
        orig_coords_rec = rec_graph.ndata['x']
        orig_coords_rec2 = rec2_graph.ndata['x']

        coords_lig = lig_graph.ndata['new_x']
        coords_rec = rec_graph.ndata['x']
        coords_rec2 = rec2_graph.ndata['x']

        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        rand_h_lig = rand_dist.sample([h_feats_lig.size(0),
                                       self.random_vec_dim]).to(h_feats_lig.device)
        rand_h_rec = rand_dist.sample([h_feats_rec.size(0),
                                       self.random_vec_dim]).to(h_feats_lig.device)
        rand_h_rec2 = rand_dist.sample([h_feats_rec2.size(0),
                                        self.random_vec_dim]).to(h_feats_lig.device)
        h_feats_lig = torch.cat([h_feats_lig, rand_h_lig], dim=1)
        h_feats_rec = torch.cat([h_feats_rec, rand_h_rec], dim=1)
        h_feats_rec2 = torch.cat([h_feats_rec2, rand_h_rec2], dim=1)

        # random noise:
        if self.noise_initial > 0 and self.training:
            noise_level = self.noise_initial * \
                self.noise_decay_rate**(epoch + 1)
            h_feats_lig = h_feats_lig + noise_level * \
                torch.randn_like(h_feats_lig)
            h_feats_rec = h_feats_rec + noise_level * \
                torch.randn_like(h_feats_rec)
            h_feats_rec2 = h_feats_rec2 + noise_level * \
                torch.randn_like(h_feats_rec2)
            coords_lig = coords_lig + noise_level * \
                torch.randn_like(coords_lig)
            coords_rec = coords_rec + noise_level * \
                torch.randn_like(coords_rec)
            coords_rec2 = coords_rec2 + noise_level * \
                torch.randn_like(coords_rec2)

        if self.use_mean_node_features:
            h_feats_lig = torch.cat([h_feats_lig, torch.log(lig_graph.ndata['mu_r_norm'])], dim=1)
            h_feats_rec = torch.cat([h_feats_rec, torch.log(rec_graph.ndata['mu_r_norm'])], dim=1)
            h_feats_rec2 = torch.cat(
                [h_feats_rec2, torch.log(rec2_graph.ndata['mu_r_norm'])], dim=1)

        original_ligand_node_features = h_feats_lig
        original_receptor_node_features = h_feats_rec
        original_receptor2_node_features = h_feats_rec2
        lig_graph.edata['feat'] *= self.use_edge_features_in_gmn
        rec_graph.edata['feat'] *= self.use_edge_features_in_gmn
        rec2_graph.edata['feat'] *= self.use_edge_features_in_gmn

        mask_lig_q, mask_rec_q, mask_rec2_q = None, None, None
        if self.cross_msgs:
            # lig <- rec, rec2
            mask_lig_q = get_two_keys_mask(lig_graph.batch_num_nodes(), rec_graph.batch_num_nodes(),
                                           rec2_graph.batch_num_nodes(), self.device)
            # rec <- lig, rec2
            mask_rec_q = get_two_keys_mask(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes(),
                                           rec2_graph.batch_num_nodes(), self.device)
            # rec2 <- lig, rec
            mask_rec2_q = get_two_keys_mask(rec2_graph.batch_num_nodes(),
                                            lig_graph.batch_num_nodes(),
                                            rec_graph.batch_num_nodes(), self.device)

        if self.separate_lig:
            raise NotImplementedError
            coords_lig_separate = coords_lig
            h_feats_lig_separate = h_feats_lig
            coords_rec_separate = coords_rec
            h_feats_rec_separate = h_feats_rec
        # full_trajectory = [coords_lig.detach().cpu()]
        geom_losses = 0
        for i, layer in enumerate(self.iegmn_layers):
            if self.debug:
                log('layer ', i)

            # # save layer
            # lig_pdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']] = self.unnormalize(coords_lig).cpu().numpy()
            # lig_pdb.to_pdb(f'output/lig_visualize/layer_{i}.pdb', records=['HETATM'], gz=False)

            coords_lig, h_feats_lig, \
                coords_rec, h_feats_rec, \
                coords_rec2, h_feats_rec2, \
                trajectory, geom_loss = layer(
                    lig_graph=lig_graph,
                    rec_graph=rec_graph,
                    rec2_graph=rec2_graph,
                    coords_lig=coords_lig,
                    h_feats_lig=h_feats_lig,
                    original_ligand_node_features=original_ligand_node_features,
                    orig_coords_lig=orig_coords_lig,
                    coords_rec=coords_rec,
                    h_feats_rec=h_feats_rec,
                    original_receptor_node_features=original_receptor_node_features,
                    orig_coords_rec=orig_coords_rec,
                    coords_rec2=coords_rec2,
                    h_feats_rec2=h_feats_rec2,
                    original_receptor2_node_features=original_receptor2_node_features,
                    orig_coords_rec2=orig_coords_rec2,
                    mask_lig_q=mask_lig_q,
                    mask_rec_q=mask_rec_q,
                    mask_rec2_q=mask_rec2_q,
                    geometry_graph=geometry_graph
                )
            if not self.separate_lig:
                geom_losses = geom_losses + geom_loss
                # full_trajectory.extend(trajectory)

            # update gt_pocket coords

            if given_gt_pocket:
                # if 'lig1_coord' in self.known_pocket_info:
                p1lig_lig_pocket_mask = torch.cat(kwargs['p1lig_lig_pocket_mask'])
                p1lig_lig_pocket_coords = torch.cat(kwargs['p1lig_lig_pocket_coords']).to(
                    lig_graph.device)
                p1lig_lig_pocket_coords = self.normalize(p1lig_lig_pocket_coords)
                assert len(p1lig_lig_pocket_mask) == len(lig_graph.ndata['new_x'])
                lig_graph.ndata['new_x'][p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
                coords_lig[p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
                # if 'lig2_coord' in self.known_pocket_info:
                p2lig_lig_pocket_coords = [
                    self.normalize(i).to(lig_graph.device)
                    for i in kwargs['p2lig_lig_pocket_coords']
                ]
                # align batch gt to input
                lig_graph.ndata['new_x'] = rigid_align_batch_pockets(
                    old_batch_all_coords=lig_graph.ndata['new_x'],
                    batch_pocket_masks=kwargs['p2lig_lig_pocket_mask'],
                    batch_pocket_coords=p2lig_lig_pocket_coords)
                coords_lig = lig_graph.ndata['new_x']

        if self.separate_lig:
            raise NotImplementedError

        rotations = []
        translations = []
        recs_keypts = []
        ligs_keypts = []
        ligs_evolved = []
        ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
        ligs_node_idx.insert(0, 0)
        recs_node_idx = torch.cumsum(rec_graph.batch_num_nodes(), dim=0).tolist()
        recs_node_idx.insert(0, 0)
        # new for triplet
        valid = []  # in case checkpoint return NaN
        rotations_2 = []  # rec2 -> lig
        translations_2 = []
        recs_keypts_2 = []
        ligs_keypts_2 = []
        p2_rmsd_preds = []
        recs_node_idx_2 = torch.cumsum(rec2_graph.batch_num_nodes(), dim=0).tolist()
        recs_node_idx_2.insert(0, 0)

        # TODO: run SVD in batches, if possible
        for idx in range(len(ligs_node_idx) - 1):
            lig_start = ligs_node_idx[idx]
            lig_end = ligs_node_idx[idx + 1]
            rec_start = recs_node_idx[idx]
            rec_end = recs_node_idx[idx + 1]
            rec2_start = recs_node_idx_2[idx]
            rec2_end = recs_node_idx_2[idx + 1]
            # Get H vectors

            rec_feats = h_feats_rec[rec_start:rec_end]  # (m, d)
            # rec_feats_mean = torch.mean(self.h_mean_rec(rec_feats), dim=0, keepdim=True)  # (1, d)
            rec2_feats = h_feats_rec2[rec2_start:rec2_end]  # (m, d)
            # rec2_feats_mean = torch.mean(self.h_mean_rec2(rec2_feats), dim=0, keepdim=True)  # (1, d)
            lig_feats = h_feats_lig[lig_start:lig_end]  # (n, d)
            # lig_feats_mean = torch.mean(self.h_mean_lig(lig_feats), dim=0, keepdim=True)  # (1, d)

            d = lig_feats.shape[1]
            assert d == self.out_feats_dim
            # Z coordinates
            Z_rec_coords = coords_rec[rec_start:rec_end]
            Z_rec2_coords = coords_rec2[rec2_start:rec2_end]
            # TODO: for debug
            Z_lig_coords = orig_coords_lig[lig_start:lig_end]

            queries = self.keypts_query_embeddings.weight

            src = torch.cat((lig_feats, rec_feats, rec2_feats), dim=0)
            A, B, C = len(lig_feats), len(rec_feats), len(rec2_feats)
            lig_feats_pe = self.feat_type_embeddings.weight[0].repeat(A, 1)
            rec_feats_pe = self.feat_type_embeddings.weight[1].repeat(B, 1)
            rec2_feats_pe = self.feat_type_embeddings.weight[2].repeat(C, 1)
            src_pe = torch.cat((lig_feats_pe, rec_feats_pe, rec2_feats_pe), dim=0)

            queries, keys = self.kpt_transformer(src[None, ...], src_pe[None, ...], queries[None,
                                                                                            ...])
            queries = queries[0]
            keys = keys[0]

            query_list = []
            for i in range(4):
                start_idx = i * self.num_att_heads
                end_idx = (i + 1) * self.num_att_heads
                query_list.append(queries[start_idx:end_idx])

            p2_rmsd_pred = self.p2_rmsd_pred_head(queries[-1:])[0][0]
            p2_rmsd_preds.append(p2_rmsd_pred)

            keys = keys + src_pe  # only for keys
            lig_feats = keys[:A]
            rec_feats = keys[A:A + B]
            rec2_feats = keys[A + B:]
            assert len(rec2_feats) == C

            if self.unnormalized_kpt_weights:
                lig_scales = self.scale_lig(lig_feats)
                rec_scales = self.scale_rec(rec_feats)
                Z_lig_coords = Z_lig_coords * lig_scales
                Z_rec_coords = Z_rec_coords * rec_scales
                rec2_scales = self.scale_rec2(rec2_feats)
                Z_rec2_coords = Z_rec2_coords * rec2_scales

            # TODO: 只算自己好，还是算上邻近的好
            lig1_keypts = self.keypts_cross_attn[0](query_list[0][None, ...],
                                                    torch.cat((lig_feats, rec_feats))[None, ...],
                                                    torch.cat((Z_lig_coords, Z_rec_coords))[None,
                                                                                            ...])
            rec_keypts = self.keypts_cross_attn[1](query_list[1][None, ...],
                                                   torch.cat((rec_feats, lig_feats))[None, ...],
                                                   torch.cat((Z_rec_coords, Z_lig_coords))[None,
                                                                                           ...])
            if self.align_method == 'p1-lig-p2':
                lig2_keypts = self.keypts_cross_attn[2](query_list[2][None, ...],
                                                        torch.cat((lig_feats, rec2_feats))[None,
                                                                                           ...],
                                                        torch.cat(
                                                            (Z_lig_coords, Z_rec2_coords))[None,
                                                                                           ...])
                rec2_keypts = self.keypts_cross_attn[3](query_list[3][None, ...],
                                                        torch.cat((rec2_feats, lig_feats))[None,
                                                                                           ...],
                                                        torch.cat(
                                                            (Z_rec2_coords, Z_lig_coords))[None,
                                                                                           ...])
            elif self.align_method == 'p1-p2':
                lig2_keypts = self.keypts_cross_attn[2](query_list[2][None, ...],
                                                        torch.cat((rec_feats, rec2_feats))[None,
                                                                                           ...],
                                                        torch.cat(
                                                            (Z_rec_coords, Z_rec2_coords))[None,
                                                                                           ...])
                rec2_keypts = self.keypts_cross_attn[3](query_list[3][None, ...],
                                                        torch.cat((rec2_feats, rec_feats))[None,
                                                                                           ...],
                                                        torch.cat(
                                                            (Z_rec2_coords, Z_rec_coords))[None,
                                                                                           ...])
            else:
                raise ValueError('Unknown align method: {}'.format(self.align_method))

            # remove batch
            lig_keypts = lig1_keypts[0]
            rec_keypts = rec_keypts[0]
            lig2_keypts = lig2_keypts[0]
            rec2_keypts = rec2_keypts[0]
            # unnormalize start
            lig_keypts, rec_keypts, lig2_keypts, rec2_keypts = self.unnormalize(
                lig_keypts, rec_keypts, lig2_keypts, rec2_keypts)
            Z_lig_coords = self.unnormalize(Z_lig_coords)
            # unnormalize end

            # zero_grad = 0. * (lig_keypts.sum() + rec_keypts.sum() + lig2_keypts.sum() + rec2_keypts.sum())
            if not self.training:
                rec_keypts = kwargs['p1lig_p1_pocket_coords'][idx].to(lig_graph.device)
                # p1lig_lig_pocket_mask = kwargs['p1lig_lig_pocket_mask'][idx]
                # p1lig_lig_pocket_coords = kwargs['p1lig_lig_pocket_coords'][idx].to(lig_graph.device)
                # R, t = model_kabsch(from_points=Z_lig_coords[p1lig_lig_pocket_mask], dst_points=rec_keypts)
                # Z_lig_coords = rotate_and_translate(Z_lig_coords, R, t)
                lig_keypts = Z_lig_coords[p1lig_lig_pocket_mask]
                # lig_keypts = Z_lig_coords[p1lig_lig_pocket_mask]
                # Z_lig_coords[p1lig_lig_pocket_mask] = lig_keypts
                # p2lig_lig_pocket_mask = kwargs['p2lig_lig_pocket_mask'][idx]
                # p2lig_lig_pocket_coords = kwargs['p2lig_lig_pocket_coords'][idx].to(lig_graph.device)
                # R, t = model_kabsch(from_points=p2lig_lig_pocket_coords, dst_points=Z_lig_coords[p2lig_lig_pocket_mask])
                # lig2_coord = rotate_and_translate(p2lig_lig_pocket_coords, R, t)
                # Z_lig_coords[p2lig_lig_pocket_mask] = lig2_coord
                # lig2_keypts = lig2_coord
                lig2_keypts = Z_lig_coords[p2lig_lig_pocket_mask]
                rec2_keypts = kwargs['p2lig_p2_pocket_coords'][idx].to(lig_graph.device)

            recs_keypts.append(rec_keypts)
            ligs_keypts.append(lig_keypts)
            recs_keypts_2.append(rec2_keypts)
            ligs_keypts_2.append(lig2_keypts)
            valid.append(True)

            # Apply Kabsch algorithm
            rotation, translation = model_kabsch(from_points=lig_keypts,
                                                 dst_points=rec_keypts,
                                                 num_att_heads=self.num_att_heads,
                                                 complex_names=complex_names,
                                                 device=self.device)
            rotations.append(rotation)
            translations.append(translation)
            # (rotation @ evolved_ligs[idx].t()).t() + translation
            moved_lig_coords = (rotation @ Z_lig_coords.T).T + translation
            if self.align_method == 'p1-lig-p2':
                moved_lig2_keypts = (rotation @ lig2_keypts.t()).t() + translation
                rotation2, translation2 = model_kabsch(from_points=rec2_keypts,
                                                       dst_points=moved_lig2_keypts,
                                                       num_att_heads=self.num_att_heads,
                                                       complex_names=complex_names,
                                                       device=self.device)
            elif self.align_method == 'p1-p2':
                rotation2, translation2 = model_kabsch(from_points=rec2_keypts,
                                                       dst_points=lig2_keypts,
                                                       num_att_heads=self.num_att_heads,
                                                       complex_names=complex_names,
                                                       device=self.device)
            else:
                raise ValueError('Unknown align method: {}'.format(self.align_method))
            rotations_2.append(rotation2)
            translations_2.append(translation2)

            ligs_evolved.append(moved_lig_coords)

        # ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss \
        #    ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2
        return ligs_evolved, ligs_keypts, recs_keypts, rotations, translations, geom_losses, \
            ligs_keypts_2, recs_keypts_2, rotations_2, translations_2, valid, p2_rmsd_preds
