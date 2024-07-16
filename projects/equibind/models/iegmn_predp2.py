from typing import Tuple, List

import torch
from torch import nn, Tensor

from mmpretrain.registry import MODELS

from .rotate_utils import model_kabsch, rigid_align_batch_pockets, rotate_and_translate
from .logger import log
from .triplet_dock import IEGMN, get_two_keys_mask
from .transformer import TwoWayTransformer, MLP

@MODELS.register_module()
class IEGMN_QueryDecoder_KnownPocket(IEGMN):

    def __init__(
            self,
            *args,
            align_method='p1-lig-p2',   # p1-p2
            known_pocket_info=(),
            **kwargs):
        """
        align_method: 
            p1-lig-p2: p2 aligns to lig first, and then both p2 and lig align to p1,
            p1-p2: p2 aligns to p1 directly.
        """
        super().__init__(*args, **kwargs)
        assert align_method in ('p1-lig-p2', 'p1-p2')
        self.align_method = align_method
        availabel_pocket_info = ('p1_mask', 'lig1_mask', 'lig1_coord', 'lig2_mask', 'lig2_coord', 'p2_mask')
        assert set(known_pocket_info) & set(availabel_pocket_info) == set(known_pocket_info), \
            f'known_pocket_info {known_pocket_info} not supported, only support {availabel_pocket_info}'
        assert len(set(known_pocket_info)) == 6, 'This Decoder only for known pocket'
        if len(known_pocket_info) > 0:
            assert align_method == 'p1-lig-p2'
        self.known_pocket_info = set(known_pocket_info)
        del self.keypts_attention_lig, self.keypts_attention_lig2, self.keypts_queries_lig, self.keypts_queries_lig2, \
            self.keypts_attention_rec, self.keypts_attention_rec2, self.keypts_queries_rec, self.keypts_queries_rec2
        del self.h_mean_rec, self.h_mean_rec2, self.h_mean_lig

        if len(set(('p1_mask', 'lig1_mask')) & set(known_pocket_info)) > 0:
            self.p1lig_pock_gt_embed = nn.Embedding(2, self.out_feats_dim) # 0: no_gt, 1: gt
            self.p1lig_pock_gt_embed.weight.data = self.p1lig_pock_gt_embed.weight.data * 0.
        if len(set(('p2_mask', 'lig2_mask')) & set(known_pocket_info)) > 0:
            self.p2lig_pock_gt_embed = nn.Embedding(2, self.out_feats_dim) # 0: no_gt, 1: gt
            self.p2lig_pock_gt_embed.weight.data = self.p2lig_pock_gt_embed.weight.data * 0.

        # self.coord_embeder = nn.Linear(3, self.out_feats_dim, bias=False)
        # self.feat_type_embeddings = nn.Embedding(3, self.out_feats_dim)
        # self.feat_type_embeddings.weight = nn.parameter.Parameter(
        #     torch.zeros_like(self.feat_type_embeddings.weight))
        self.p2_rmsd_query = nn.Embedding(1, self.out_feats_dim)
        self.kpt_transformer = TwoWayTransformer(depth=2, embedding_dim=self.out_feats_dim, num_heads=1, mlp_dim=self.out_feats_dim * 4)
        self.p2_rmsd_pred_head = MLP(input_dim=self.out_feats_dim, hidden_dim=self.out_feats_dim, output_dim=1, num_layers=3, sigmoid_output=False)
        self.coord2feat = MLP(input_dim=3, hidden_dim=self.out_feats_dim, output_dim=self.out_feats_dim, num_layers=3, sigmoid_output=False)
        self.feat_downsample = MLP(input_dim=self.out_feats_dim * 2, hidden_dim=self.out_feats_dim, output_dim=self.out_feats_dim, num_layers=3, sigmoid_output=False)

    def forward(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_names, epoch, **kwargs) -> Tuple[List[Tensor], ...]:
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

        if 'lig1_coord' in self.known_pocket_info:
            assert 'lig1_mask' in self.known_pocket_info
            p1lig_lig_pocket_mask = torch.cat(kwargs['p1lig_lig_pocket_mask'])
            p1lig_lig_pocket_coords = torch.cat(kwargs['p1lig_lig_pocket_coords']).to(lig_graph.device)
            p1lig_lig_pocket_coords = self.normalize(p1lig_lig_pocket_coords)
            assert len(p1lig_lig_pocket_mask) == len(lig_graph.ndata['new_x'])
            lig_graph.ndata['new_x'][p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords

        if 'lig2_mask' in self.known_pocket_info:
            p2lig_lig_pocket_mask = torch.cat(kwargs['p2lig_lig_pocket_mask'])
            h_feats_lig[p2lig_lig_pocket_mask] += self.p2lig_pock_gt_embed.weight[1][None, ]
            h_feats_lig[~p2lig_lig_pocket_mask] += self.p2lig_pock_gt_embed.weight[0][None, ]
        
        if 'lig2_coord' in self.known_pocket_info:
            p2lig_lig_pocket_coords = [self.normalize(i).to(lig_graph.device) for i in kwargs['p2lig_lig_pocket_coords']]
            # align batch gt to input
            lig_graph.ndata['new_x'] = rigid_align_batch_pockets(
                old_batch_all_coords=lig_graph.ndata['new_x'],
                batch_pocket_masks=kwargs['p2lig_lig_pocket_mask'],
                batch_pocket_coords=p2lig_lig_pocket_coords
            )
        
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
        rand_h_lig = rand_dist.sample([h_feats_lig.size(0), self.random_vec_dim]).to(h_feats_lig.device)
        rand_h_rec = rand_dist.sample([h_feats_rec.size(0), self.random_vec_dim]).to(h_feats_lig.device)
        rand_h_rec2 = rand_dist.sample([h_feats_rec2.size(0), self.random_vec_dim]).to(h_feats_lig.device)
        h_feats_lig = torch.cat([h_feats_lig, rand_h_lig], dim=1)
        h_feats_rec = torch.cat([h_feats_rec, rand_h_rec], dim=1)
        h_feats_rec2 = torch.cat([h_feats_rec2, rand_h_rec2], dim=1)

        # random noise:
        if self.noise_initial > 0 and self.training:
            noise_level = self.noise_initial * self.noise_decay_rate ** (epoch + 1)
            h_feats_lig = h_feats_lig + noise_level * torch.randn_like(h_feats_lig)
            h_feats_rec = h_feats_rec + noise_level * torch.randn_like(h_feats_rec)
            h_feats_rec2 = h_feats_rec2 + noise_level * torch.randn_like(h_feats_rec2)
            coords_lig = coords_lig + noise_level * torch.randn_like(coords_lig)
            coords_rec = coords_rec + noise_level * torch.randn_like(coords_rec)
            coords_rec2 = coords_rec2 + noise_level * torch.randn_like(coords_rec2)

        if self.use_mean_node_features:
            h_feats_lig = torch.cat([h_feats_lig, torch.log(lig_graph.ndata['mu_r_norm'])],
                                    dim=1)
            h_feats_rec = torch.cat(
                [h_feats_rec, torch.log(rec_graph.ndata['mu_r_norm'])], dim=1)
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
            mask_lig_q = get_two_keys_mask(lig_graph.batch_num_nodes(), rec_graph.batch_num_nodes(), rec2_graph.batch_num_nodes(), self.device)
            # rec <- lig, rec2
            mask_rec_q = get_two_keys_mask(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes(), rec2_graph.batch_num_nodes(), self.device)
            # rec2 <- lig, rec
            mask_rec2_q = get_two_keys_mask(rec2_graph.batch_num_nodes(), lig_graph.batch_num_nodes(), rec_graph.batch_num_nodes(), self.device)

        if self.separate_lig:
            raise NotImplementedError
            coords_lig_separate =coords_lig
            h_feats_lig_separate =h_feats_lig
            coords_rec_separate =coords_rec
            h_feats_rec_separate =h_feats_rec
        # full_trajectory = [coords_lig.detach().cpu()]
        geom_losses = 0
        for i, layer in enumerate(self.iegmn_layers):
            if self.debug: log('layer ', i)
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
            
            ### update gt_pocket coords

            if i <= 8:
            # if 'lig1_coord' in self.known_pocket_info:
                p1lig_lig_pocket_mask = torch.cat(kwargs['p1lig_lig_pocket_mask'])
                p1lig_lig_pocket_coords = torch.cat(kwargs['p1lig_lig_pocket_coords']).to(lig_graph.device)
                p1lig_lig_pocket_coords = self.normalize(p1lig_lig_pocket_coords)
                assert len(p1lig_lig_pocket_mask) == len(lig_graph.ndata['new_x'])
                lig_graph.ndata['new_x'][p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
                coords_lig[p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
            # if 'lig2_coord' in self.known_pocket_info:
                p2lig_lig_pocket_coords = [self.normalize(i).to(lig_graph.device) for i in kwargs['p2lig_lig_pocket_coords']]
                # align batch gt to input
                lig_graph.ndata['new_x'] = rigid_align_batch_pockets(
                    old_batch_all_coords=lig_graph.ndata['new_x'],
                    batch_pocket_masks=kwargs['p2lig_lig_pocket_mask'],
                    batch_pocket_coords=p2lig_lig_pocket_coords
                )
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
        ### new for triplet
        valid = [] # in case checkpoint return NaN 
        rotations_2 = []    # rec2 -> lig
        translations_2 = []
        recs_keypts_2 = []
        ligs_keypts_2 = []
        p2_rmsd_preds = []
        recs_node_idx_2 = torch.cumsum(rec2_graph.batch_num_nodes(), dim=0).tolist()
        recs_node_idx_2.insert(0, 0)

        ### TODO: run SVD in batches, if possible
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



            queries = self.p2_rmsd_query.weight

            src = torch.cat((lig_feats, rec_feats, rec2_feats), dim=0)
            # # combine feats and coords
            coords = torch.cat((Z_lig_coords, Z_rec_coords, Z_rec2_coords), dim=0)
            coords_feat = self.coord2feat(coords)
            src = torch.cat((src, coords_feat), dim=1)
            src = self.feat_downsample(src)

            # A, B, C = len(lig_feats), len(rec_feats), len(rec2_feats)
            # lig_feats_pe = self.feat_type_embeddings.weight[0].repeat(A, 1)
            # rec_feats_pe = self.feat_type_embeddings.weight[1].repeat(B, 1)
            # rec2_feats_pe = self.feat_type_embeddings.weight[2].repeat(C, 1)
            # src_pe = torch.cat((lig_feats_pe, rec_feats_pe, rec2_feats_pe), dim=0)
            src_pe = torch.zeros_like(src)  # fake position embeddings

            queries, keys = self.kpt_transformer(src[None, ...], src_pe[None, ...], queries[None, ...])
            queries = queries[0]    # [1, 256]
            keys = keys[0]

            # zero_grad = 0. * (Z_rec_coords.sum() + Z_rec2_coords.sum())
            p2_rmsd_pred = self.p2_rmsd_pred_head(queries)[0][0]
            p2_rmsd_preds.append(p2_rmsd_pred)

            if self.unnormalized_kpt_weights:
                lig_scales = self.scale_lig(lig_feats)
                rec_scales = self.scale_rec(rec_feats)
                Z_lig_coords = Z_lig_coords * lig_scales
                Z_rec_coords = Z_rec_coords * rec_scales
                rec2_scales = self.scale_rec2(rec2_feats)
                Z_rec2_coords = Z_rec2_coords * rec2_scales

            # unnormalize start
            Z_lig_coords = self.unnormalize(Z_lig_coords)
            # unnormalize end

            rec_keypts = kwargs['p1lig_p1_pocket_coords'][idx].to(lig_graph.device)
            p1lig_lig_pocket_mask = kwargs['p1lig_lig_pocket_mask'][idx]
            lig_keypts = kwargs['p1lig_lig_pocket_coords'][idx].to(lig_graph.device)
            # Z_lig_coords[p1lig_lig_pocket_mask] = lig_keypts
            p2lig_lig_pocket_mask = kwargs['p2lig_lig_pocket_mask'][idx]
            p2lig_lig_pocket_coords = kwargs['p2lig_lig_pocket_coords'][idx].to(lig_graph.device)
            R, t = model_kabsch(from_points=p2lig_lig_pocket_coords, dst_points=Z_lig_coords[p2lig_lig_pocket_mask])
            lig2_coord = rotate_and_translate(p2lig_lig_pocket_coords, R, t)
            # Z_lig_coords[p2lig_lig_pocket_mask] = lig2_coord
            lig2_keypts = lig2_coord
            rec2_keypts = kwargs['p2lig_p2_pocket_coords'][idx].to(lig_graph.device)

            recs_keypts.append(rec_keypts)
            ligs_keypts.append(lig_keypts)
            recs_keypts_2.append(rec2_keypts)
            ligs_keypts_2.append(lig2_keypts)
            valid.append(True)

            ## Apply Kabsch algorithm
            rotation, translation = model_kabsch(
                from_points=lig_keypts, dst_points=rec_keypts,
                num_att_heads=self.num_att_heads, complex_names=complex_names, device=self.device)
            rotations.append(rotation)
            translations.append(translation)
            # (rotation @ evolved_ligs[idx].t()).t() + translation
            moved_lig_coords = (rotation @ Z_lig_coords.T).T + translation
            if self.align_method == 'p1-lig-p2':
                moved_lig2_keypts = (rotation @ lig2_keypts.t()).t() + translation
                rotation2, translation2 = model_kabsch(
                    from_points=rec2_keypts, dst_points=moved_lig2_keypts,
                    num_att_heads=self.num_att_heads, complex_names=complex_names, device=self.device)
            elif self.align_method == 'p1-p2':
                rotation2, translation2 = model_kabsch(
                    from_points=rec2_keypts, dst_points=lig2_keypts,
                    num_att_heads=self.num_att_heads, complex_names=complex_names, device=self.device
                )
            else:
                raise ValueError('Unknown align method: {}'.format(self.align_method))
            rotations_2.append(rotation2)
            translations_2.append(translation2)

            ligs_evolved.append(moved_lig_coords)

        # ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss \
        #    ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2
        return ligs_evolved, ligs_keypts, recs_keypts, rotations, translations, geom_losses, \
                ligs_keypts_2, recs_keypts_2, rotations_2, translations_2, valid, p2_rmsd_preds

