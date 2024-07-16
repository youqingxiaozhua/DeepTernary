import os
from typing import Tuple, List, Union
import logging
import math
import random
from datetime import datetime

import torch
from torch import nn, Tensor
import dgl
from dgl import function as fn
from mmengine.model import BaseModel
from mmengine.logging import MessageHub
from mmengine.runner import load_state_dict
from mmpretrain.registry import MODELS

from .process_mols import AtomEncoder, rec_atom_feature_dims, rec_residue_feature_dims, lig_feature_dims
from .logger import log
from .losses import TripletBindingLoss
from .equibind import get_non_lin, get_layer_norm, get_norm, apply_norm, CoordsNorm, cross_attention
from .rotate_utils import model_kabsch, rotate_and_translate, rigid_align_batch_pockets, batch_align_lig_to_pocket
from .transformer import TwoWayTransformer, CoordAttention, MLP


def get_mask(ligand_batch_num_nodes, receptor_batch_num_nodes, device):
    rows = ligand_batch_num_nodes.sum()
    cols = receptor_batch_num_nodes.sum()
    mask = torch.zeros(rows, cols, device=device)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask


def get_two_keys_mask(lig_batch_num_nodes, rec_batch_num_nodes, rec2_batch_num_nodes, device):
    """
    calculate attention mask when two rec feats are concated.
    """
    rows = lig_batch_num_nodes.sum()
    cols = rec_batch_num_nodes.sum() + rec2_batch_num_nodes.sum()
    mask = torch.zeros(rows, cols).to(device)
    partial_l = 0
    partial_r = 0
    partial_r2 = rec_batch_num_nodes.sum()
    for l_n, r_n, r2_n in zip(lig_batch_num_nodes, rec_batch_num_nodes, rec2_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        mask[partial_l: partial_l + l_n, partial_r2: partial_r2 + r2_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
        partial_r2 = partial_r2 + r2_n
    return mask


class IEGMN_Layer(nn.Module):
    def __init__(
            self,
            orig_h_feats_dim,
            h_feats_dim,  # in dim of h
            out_feats_dim,  # out dim of h
            lig_input_edge_feats_dim,
            rec_input_edge_feats_dim,
            nonlin,
            cross_msgs,
            layer_norm,
            layer_norm_coords,
            final_h_layer_norm,
            use_dist_in_layers,
            skip_weight_h,
            x_connection_init,
            leakyrelu_neg_slope,
            debug,
            device,
            dropout,
            save_trajectories=False,
            rec_square_distance_scale=1,
            standard_norm_order=False,
            normalize_coordinate_update=False,
            lig_evolve=True,
            rec_evolve=True,
            fine_tune=False,
            geometry_regularization=False,
            pre_crossmsg_norm_type=0,
            post_crossmsg_norm_type=0,
            norm_cross_coords_update= False,
            loss_geometry_regularization = False,
            geom_reg_steps= 1,
            geometry_reg_step_size=0.1,
            **kwargs,
    ):

        super(IEGMN_Layer, self).__init__()

        self.fine_tune = fine_tune
        self.cross_msgs = cross_msgs
        self.normalize_coordinate_update = normalize_coordinate_update
        self.final_h_layer_norm = final_h_layer_norm
        self.use_dist_in_layers = use_dist_in_layers
        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init
        self.rec_square_distance_scale = rec_square_distance_scale
        self.geometry_reg_step_size = geometry_reg_step_size
        self.norm_cross_coords_update =norm_cross_coords_update
        self.loss_geometry_regularization = loss_geometry_regularization

        self.debug = debug
        self.device = device
        self.lig_evolve = lig_evolve
        self.rec_evolve = rec_evolve
        self.h_feats_dim = h_feats_dim
        self.out_feats_dim = out_feats_dim
        self.standard_norm_order = standard_norm_order
        self.pre_crossmsg_norm_type = pre_crossmsg_norm_type
        self.post_crossmsg_norm_type = post_crossmsg_norm_type
        self.all_sigmas_dist = [1.5 ** x for x in range(15)]
        self.geometry_regularization = geometry_regularization
        self.geom_reg_steps = geom_reg_steps
        self.save_trajectories = save_trajectories

        # EDGES
        lig_edge_mlp_input_dim = (h_feats_dim * 2) + lig_input_edge_feats_dim
        if self.use_dist_in_layers and self.lig_evolve:
            lig_edge_mlp_input_dim += len(self.all_sigmas_dist)
        if self.standard_norm_order:
            self.lig_edge_mlp = nn.Sequential(
                nn.Linear(lig_edge_mlp_input_dim, self.out_feats_dim),
                get_layer_norm(layer_norm, self.out_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                get_layer_norm(layer_norm, self.out_feats_dim),
            )
        else:
            self.lig_edge_mlp = nn.Sequential(
                nn.Linear(lig_edge_mlp_input_dim, self.out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, self.out_feats_dim),
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
            )
        rec_edge_mlp_input_dim = (h_feats_dim * 2) + rec_input_edge_feats_dim
        if self.use_dist_in_layers and self.rec_evolve:
            rec_edge_mlp_input_dim += len(self.all_sigmas_dist)
        if self.standard_norm_order:
            self.rec_edge_mlp = nn.Sequential(
                nn.Linear(rec_edge_mlp_input_dim, self.out_feats_dim),
                get_layer_norm(layer_norm, self.out_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                get_layer_norm(layer_norm, self.out_feats_dim),
            )
        else:
            self.rec_edge_mlp = nn.Sequential(
                nn.Linear(rec_edge_mlp_input_dim, self.out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, self.out_feats_dim),
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
            )

        # NODES
        self.node_norm = nn.Identity()  # nn.LayerNorm(h_feats_dim)

        # normalization of x_i - x_j is not currently used
        if self.normalize_coordinate_update:
            self.lig_coords_norm = CoordsNorm(scale_init=1e-2)
            self.rec_coords_norm = CoordsNorm(scale_init=1e-2)
        if self.fine_tune:
            if self.norm_cross_coords_update:
                self.lig_cross_coords_norm = CoordsNorm(scale_init=1e-2)
                self.rec_cross_coords_norm = CoordsNorm(scale_init=1e-2)
            else:
                self.lig_cross_coords_norm =nn.Identity()
                self.rec_cross_coords_norm = nn.Identity()

        self.att_mlp_Q_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        self.att_mlp_Q = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        self.att_mlp_Q_rec2 = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K_rec2 = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V_rec2 = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        if self.standard_norm_order:
            self.node_mlp_lig = nn.Sequential(
                nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim),
                get_layer_norm(layer_norm, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(h_feats_dim, out_feats_dim),
                get_layer_norm(layer_norm, out_feats_dim),
            )
        else:
            self.node_mlp_lig = nn.Sequential(
                nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, h_feats_dim),
                nn.Linear(h_feats_dim, out_feats_dim),
            )
        if self.standard_norm_order:
            self.node_mlp = nn.Sequential(
                nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim),
                get_layer_norm(layer_norm, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(h_feats_dim, out_feats_dim),
                get_layer_norm(layer_norm, out_feats_dim),
            )
        else:
            self.node_mlp = nn.Sequential(
                nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, h_feats_dim),
                nn.Linear(h_feats_dim, out_feats_dim),
            )

        self.final_h_layernorm_layer_lig = get_norm(self.final_h_layer_norm, out_feats_dim)
        self.final_h_layernorm_layer = get_norm(self.final_h_layer_norm, out_feats_dim)

        self.pre_crossmsg_norm_lig = get_norm(self.pre_crossmsg_norm_type, h_feats_dim)
        self.pre_crossmsg_norm_rec = get_norm(self.pre_crossmsg_norm_type, h_feats_dim)
        self.pre_crossmsg_norm_rec2 = get_norm(self.pre_crossmsg_norm_type, h_feats_dim)

        self.post_crossmsg_norm_lig = get_norm(self.post_crossmsg_norm_type, h_feats_dim)
        self.post_crossmsg_norm_rec = get_norm(self.post_crossmsg_norm_type, h_feats_dim)
        self.post_crossmsg_norm_rec2 = get_norm(self.post_crossmsg_norm_type, h_feats_dim)

        if self.standard_norm_order:
            self.coords_mlp_lig = nn.Sequential(
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                get_layer_norm(layer_norm_coords, self.out_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim, 1)
            )
        else:
            self.coords_mlp_lig = nn.Sequential(
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm_coords, self.out_feats_dim),
                nn.Linear(self.out_feats_dim, 1)
            )
        if self.standard_norm_order:
            self.coords_mlp_rec = nn.Sequential(
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                get_layer_norm(layer_norm_coords, self.out_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim, 1)
            )
        else:
            self.coords_mlp_rec = nn.Sequential(
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm_coords, self.out_feats_dim),
                nn.Linear(self.out_feats_dim, 1)
            )
        if self.fine_tune:
            self.att_mlp_cross_coors_Q = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_K = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_V = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Linear(h_feats_dim, 1),
            )
            self.att_mlp_cross_coors_Q_lig = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_K_lig = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_V_lig = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Linear(h_feats_dim, 1),
            )

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def apply_edges_lig(self, edges):
        if self.use_dist_in_layers and self.lig_evolve:
            x_rel_mag = edges.data['x_rel'] ** 2
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
            return {'msg': self.lig_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
        else:
            return {
                'msg': self.lig_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def apply_edges_rec(self, edges):
        if self.use_dist_in_layers and self.rec_evolve:
            squared_distance = torch.sum(edges.data['x_rel'] ** 2, dim=1, keepdim=True)
            # divide square distance by 10 to have a nicer separation instead of many 0.00000
            x_rel_mag = torch.cat([torch.exp(-(squared_distance / self.rec_square_distance_scale) / sigma) for sigma in
                                   self.all_sigmas_dist], dim=-1)
            return {'msg': self.rec_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
        else:
            return {
                'msg': self.rec_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def update_x_moment_lig(self, edges):
        edge_coef_ligand = self.coords_mlp_lig(edges.data['msg'])  # \phi^x(m_{i->j})
        x_rel = self.lig_coords_norm(edges.data['x_rel']) if self.normalize_coordinate_update else edges.data['x_rel']
        return {'m': x_rel * edge_coef_ligand}  # (x_i - x_j) * \phi^x(m_{i->j})

    def update_x_moment_rec(self, edges):
        edge_coef_rec = self.coords_mlp_rec(edges.data['msg'])  # \phi^x(m_{i->j})
        x_rel = self.rec_coords_norm(edges.data['x_rel']) if self.normalize_coordinate_update else edges.data['x_rel']
        return {'m': x_rel * edge_coef_rec}  # (x_i - x_j) * \phi^x(m_{i->j})

    def attention_coefficients(self, edges):  # for when using cross edges (but this is super slow so dont do it)
        return {'attention_coefficient': torch.sum(edges.dst['q'] * edges.src['k'], dim=1), 'values': edges.src['v']}

    def attention_aggregation(self, nodes):  # for when using cross edges (but this is super slow so dont do it)
        attention = torch.softmax(nodes.mailbox['attention_coefficient'], dim=1)
        return {'cross_attention_feat': torch.sum(attention[:, :, None] * nodes.mailbox['values'], dim=1)}

    def forward(self, lig_graph, rec_graph, rec2_graph, coords_lig, h_feats_lig, original_ligand_node_features, orig_coords_lig,
                coords_rec, h_feats_rec, original_receptor_node_features, orig_coords_rec,
                coords_rec2, h_feats_rec2, original_receptor2_node_features, orig_coords_rec2,
                mask_lig_q, mask_rec_q, mask_rec2_q, geometry_graph):
        with lig_graph.local_scope() and rec_graph.local_scope() and rec2_graph.local_scope():
            lig_graph.ndata['x_now'] = coords_lig
            rec_graph.ndata['x_now'] = coords_rec
            rec2_graph.ndata['x_now'] = coords_rec2
            lig_graph.ndata['feat'] = h_feats_lig  # first time set here
            rec_graph.ndata['feat'] = h_feats_rec
            rec2_graph.ndata['feat'] = h_feats_rec2

            if self.debug:
                log(torch.max(lig_graph.ndata['x_now'].abs()), 'x_now : x_i at layer entrance')
                log(torch.max(lig_graph.ndata['feat'].abs()), 'data[feat] = h_i at layer entrance')

            if self.lig_evolve:
                lig_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))  # x_i - x_j
                if self.debug:
                    log(torch.max(lig_graph.edata['x_rel'].abs()), 'x_rel : x_i - x_j')
            if self.rec_evolve:
                rec_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))
                rec2_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))

            lig_graph.apply_edges(self.apply_edges_lig)  ## i->j edge:  [h_i h_j]
            rec_graph.apply_edges(self.apply_edges_rec)
            rec2_graph.apply_edges(self.apply_edges_rec)

            if self.debug:
                log(torch.max(lig_graph.edata['msg'].abs()),
                    'data[msg] = m_{i->j} = phi^e(h_i, h_j, f_{i,j}, x_rel_mag_ligand)')

            # h_feats_lig_norm = apply_norm(lig_graph, h_feats_lig, self.final_h_layer_norm, self.final_h_layernorm_layer)
            # h_feats_rec_norm = apply_norm(rec_graph, h_feats_rec, self.final_h_layer_norm, self.final_h_layernorm_layer)
            # h_feats_rec2_norm = apply_norm(rec2_graph, h_feats_rec2, self.final_h_layer_norm, self.final_h_layernorm_layer)
            h_feats_lig_norm = apply_norm(lig_graph, h_feats_lig, self.pre_crossmsg_norm_type, self.pre_crossmsg_norm_lig)   # [M, C]
            h_feats_rec_norm = apply_norm(rec_graph, h_feats_rec, self.pre_crossmsg_norm_type, self.pre_crossmsg_norm_rec)   # [N1, C]
            h_feats_rec2_norm = apply_norm(rec2_graph, h_feats_rec2, self.pre_crossmsg_norm_type, self.pre_crossmsg_norm_rec2) # [N2, C]

            lig_q_kv = torch.cat((h_feats_rec_norm, h_feats_rec2_norm), dim=0)
            cross_attention_lig_feat = cross_attention(
                self.att_mlp_Q_lig(h_feats_lig_norm),
                self.att_mlp_K(lig_q_kv),
                self.att_mlp_V(lig_q_kv), mask_lig_q, self.cross_msgs)

            rec_q_kv = torch.cat((h_feats_lig_norm, h_feats_rec2_norm), dim=0)
            cross_attention_rec_feat = cross_attention(
                self.att_mlp_Q(h_feats_rec_norm),
                self.att_mlp_K_lig(rec_q_kv),
                self.att_mlp_V_lig(rec_q_kv), mask_rec_q,
                self.cross_msgs)

            rec2_q_kv = torch.cat((h_feats_lig_norm, h_feats_rec_norm), dim=0)
            cross_attention_rec2_feat = cross_attention(
                self.att_mlp_Q_rec2(h_feats_rec2_norm),
                self.att_mlp_K_rec2(rec2_q_kv),
                self.att_mlp_V_rec2(rec2_q_kv), mask_rec2_q,
                self.cross_msgs)

            # cross_attention_lig_feat = apply_norm(lig_graph, cross_attention_lig_feat, self.final_h_layer_norm,
            #                                       self.final_h_layernorm_layer)
            # cross_attention_rec_feat = apply_norm(rec_graph, cross_attention_rec_feat, self.final_h_layer_norm,
            #                                       self.final_h_layernorm_layer)
            # cross_attention_rec2_feat = apply_norm(rec2_graph, cross_attention_rec2_feat, self.final_h_layer_norm,
            #                                         self.final_h_layernorm_layer)
            cross_attention_lig_feat = apply_norm(lig_graph, cross_attention_lig_feat, self.post_crossmsg_norm_type,
                                                  self.post_crossmsg_norm_lig)
            cross_attention_rec_feat = apply_norm(rec_graph, cross_attention_rec_feat, self.post_crossmsg_norm_type,
                                                  self.post_crossmsg_norm_rec)
            cross_attention_rec2_feat = apply_norm(rec2_graph, cross_attention_rec2_feat, self.post_crossmsg_norm_type,
                                                  self.post_crossmsg_norm_rec2)

            if self.debug:
                log(torch.max(cross_attention_lig_feat.abs()), 'aggr_cross_msg(i) = sum_j a_{i,j} * h_j')

            if self.lig_evolve:
                lig_graph.update_all(self.update_x_moment_lig, fn.mean('m', 'x_update'))
                # Inspired by https://arxiv.org/pdf/2108.10521.pdf we use original X and not only graph.ndata['x_now']
                x_evolved_lig = self.x_connection_init * orig_coords_lig + (1. - self.x_connection_init) * \
                                lig_graph.ndata['x_now'] + lig_graph.ndata['x_update']
            else:
                x_evolved_lig = coords_lig

            if self.rec_evolve:
                rec_graph.update_all(self.update_x_moment_rec, fn.mean('m', 'x_update'))
                x_evolved_rec = self.x_connection_init * orig_coords_rec + (1. - self.x_connection_init) * \
                                rec_graph.ndata['x_now'] + rec_graph.ndata['x_update']
                rec2_graph.update_all(self.update_x_moment_rec, fn.mean('m', 'x_update'))
                x_evolved_rec2 = self.x_connection_init * orig_coords_rec2 + (1. - self.x_connection_init) * \
                                rec2_graph.ndata['x_now'] + rec2_graph.ndata['x_update']
            else:
                x_evolved_rec = coords_rec
                x_evolved_rec2 = coords_rec2

            lig_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))
            rec_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))
            rec2_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))

            if self.fine_tune:
                x_evolved_lig = x_evolved_lig + self.att_mlp_cross_coors_V_lig(h_feats_lig) * (
                        self.lig_cross_coords_norm(lig_graph.ndata['x_now'] - cross_attention(self.att_mlp_cross_coors_Q_lig(h_feats_lig),
                                                                   self.att_mlp_cross_coors_K(h_feats_rec),
                                                                   rec_graph.ndata['x_now'], mask, self.cross_msgs)))
            if self.fine_tune:
                raise NotImplementedError
                x_evolved_rec = x_evolved_rec + self.att_mlp_cross_coors_V(h_feats_rec) * (
                        self.rec_cross_coords_norm(rec_graph.ndata['x_now'] - cross_attention(self.att_mlp_cross_coors_Q(h_feats_rec),
                                                                   self.att_mlp_cross_coors_K_lig(h_feats_lig),
                                                                   lig_graph.ndata['x_now'], mask.transpose(0, 1),
                                                                   self.cross_msgs)))
            trajectory = []
            if self.save_trajectories: trajectory.append(x_evolved_lig.detach().cpu())
            if self.loss_geometry_regularization:
                src, dst = geometry_graph.edges()
                src = src.long()
                dst = dst.long()
                d_squared = torch.sum((x_evolved_lig[src] - x_evolved_lig[dst]) ** 2, dim=1)
                geom_loss = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2)
            else:
                geom_loss = 0. * x_evolved_lig.sum()
            if self.geometry_regularization:
                src, dst = geometry_graph.edges()
                src = src.long()
                dst = dst.long()
                for step in range(self.geom_reg_steps):
                    d_squared = torch.sum((x_evolved_lig[src] - x_evolved_lig[dst]) ** 2, dim=1)
                    Loss = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2)**2) # this is the loss whose gradient we are calculating here
                    grad_d_squared = 2 * (x_evolved_lig[src] - x_evolved_lig[dst])
                    geometry_graph.edata['partial_grads'] = 2 * (d_squared - geometry_graph.edata['feat'] ** 2)[:,None] * grad_d_squared
                    geometry_graph.update_all(fn.copy_e('partial_grads', 'partial_grads_msg'),
                                              fn.sum('partial_grads_msg', 'grad_x_evolved'))
                    grad_x_evolved = geometry_graph.ndata['grad_x_evolved']
                    x_evolved_lig = x_evolved_lig + self.geometry_reg_step_size * grad_x_evolved
                    if self.save_trajectories:
                        trajectory.append(x_evolved_lig.detach().cpu())

            if self.debug:
                log(torch.max(lig_graph.ndata['aggr_msg'].abs()), 'data[aggr_msg]: \sum_j m_{i->j} ')
                if self.lig_evolve:
                    log(torch.max(lig_graph.ndata['x_update'].abs()),
                        'data[x_update] : \sum_j (x_i - x_j) * \phi^x(m_{i->j})')
                    log(torch.max(x_evolved_lig.abs()), 'x_i new = x_evolved_lig : x_i + data[x_update]')

            input_node_upd_ligand = torch.cat((self.node_norm(lig_graph.ndata['feat']),
                                               lig_graph.ndata['aggr_msg'],
                                               cross_attention_lig_feat,
                                               original_ligand_node_features), dim=-1)

            input_node_upd_receptor = torch.cat((self.node_norm(rec_graph.ndata['feat']),
                                                 rec_graph.ndata['aggr_msg'],
                                                 cross_attention_rec_feat,
                                                 original_receptor_node_features), dim=-1)
            input_node_upd_receptor2 = torch.cat((self.node_norm(rec2_graph.ndata['feat']),
                                                    rec2_graph.ndata['aggr_msg'],
                                                    cross_attention_rec2_feat,
                                                    original_receptor2_node_features), dim=-1)

            # Skip connections
            if self.h_feats_dim == self.out_feats_dim:
                node_upd_ligand = self.skip_weight_h * self.node_mlp_lig(input_node_upd_ligand) + (
                        1. - self.skip_weight_h) * h_feats_lig
                node_upd_receptor = self.skip_weight_h * self.node_mlp(input_node_upd_receptor) + (
                        1. - self.skip_weight_h) * h_feats_rec
                node_upd_receptor2 = self.skip_weight_h * self.node_mlp(input_node_upd_receptor2) + (
                        1. - self.skip_weight_h) * h_feats_rec2
            else:
                node_upd_ligand = self.node_mlp_lig(input_node_upd_ligand)
                node_upd_receptor = self.node_mlp(input_node_upd_receptor)
                node_upd_receptor2 = self.node_mlp(input_node_upd_receptor2)

            if self.debug:
                log('node_mlp params')
                for p in self.node_mlp.parameters():
                    log(torch.max(p.abs()), 'max node_mlp_params')
                    log(torch.min(p.abs()), 'min of abs node_mlp_params')
                log(torch.max(input_node_upd_ligand.abs()), 'concat(h_i, aggr_msg, aggr_cross_msg)')
                log(torch.max(node_upd_ligand), 'h_i new = h_i + MLP(h_i, aggr_msg, aggr_cross_msg)')

            node_upd_ligand = apply_norm(lig_graph, node_upd_ligand, self.final_h_layer_norm,
                                         self.final_h_layernorm_layer_lig)
            node_upd_receptor = apply_norm(rec_graph, node_upd_receptor,
                                           self.final_h_layer_norm, self.final_h_layernorm_layer)
            node_upd_receptor2 = apply_norm(rec2_graph, node_upd_receptor2,
                                            self.final_h_layer_norm, self.final_h_layernorm_layer)

            """ return order:
            coords_lig, h_feats_lig, \
            coords_rec, h_feats_rec, \
            coords_rec2, h_feats_rec2, \
            trajectory, geom_loss"""
            return x_evolved_lig, node_upd_ligand, \
                    x_evolved_rec, node_upd_receptor, \
                    x_evolved_rec2, node_upd_receptor2, trajectory, geom_loss

    def __repr__(self):
        return "IEGMN Layer " + str(self.__dict__)


class IEGMN_LigCoordLayer(nn.Module):
    """
    Used for the last iegmn layer for given pocket setting, only output lig_coords
    """
    def __init__(
            self,
            orig_h_feats_dim,
            h_feats_dim,  # in dim of h
            out_feats_dim,  # out dim of h
            lig_input_edge_feats_dim,
            rec_input_edge_feats_dim,
            nonlin,
            cross_msgs,
            layer_norm,
            layer_norm_coords,
            final_h_layer_norm,
            use_dist_in_layers,
            skip_weight_h,
            x_connection_init,
            leakyrelu_neg_slope,
            debug,
            device,
            dropout,
            save_trajectories=False,
            rec_square_distance_scale=1,
            standard_norm_order=False,
            normalize_coordinate_update=False,
            lig_evolve=True,
            rec_evolve=True,
            fine_tune=False,
            geometry_regularization=False,
            pre_crossmsg_norm_type=0,
            post_crossmsg_norm_type=0,
            norm_cross_coords_update= False,
            loss_geometry_regularization = False,
            geom_reg_steps= 1,
            geometry_reg_step_size=0.1,
            **kwargs,
    ):

        super().__init__()

        assert fine_tune is True
        assert rec_evolve is False

        self.fine_tune = fine_tune
        self.cross_msgs = cross_msgs
        self.normalize_coordinate_update = normalize_coordinate_update
        self.final_h_layer_norm = final_h_layer_norm
        self.use_dist_in_layers = use_dist_in_layers
        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init
        self.rec_square_distance_scale = rec_square_distance_scale
        self.geometry_reg_step_size = geometry_reg_step_size
        self.norm_cross_coords_update =norm_cross_coords_update
        self.loss_geometry_regularization = loss_geometry_regularization

        self.debug = debug
        self.device = device
        self.lig_evolve = lig_evolve
        self.rec_evolve = rec_evolve
        self.h_feats_dim = h_feats_dim
        self.out_feats_dim = out_feats_dim
        self.standard_norm_order = standard_norm_order
        self.pre_crossmsg_norm_type = pre_crossmsg_norm_type
        self.post_crossmsg_norm_type = post_crossmsg_norm_type
        self.all_sigmas_dist = [1.5 ** x for x in range(15)]
        self.geometry_regularization = geometry_regularization
        self.geom_reg_steps = geom_reg_steps
        self.save_trajectories = save_trajectories

        # EDGES
        lig_edge_mlp_input_dim = (h_feats_dim * 2) + lig_input_edge_feats_dim
        if self.use_dist_in_layers and self.lig_evolve:
            lig_edge_mlp_input_dim += len(self.all_sigmas_dist)
        if self.standard_norm_order:
            self.lig_edge_mlp = nn.Sequential(
                nn.Linear(lig_edge_mlp_input_dim, self.out_feats_dim),
                get_layer_norm(layer_norm, self.out_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                get_layer_norm(layer_norm, self.out_feats_dim),
            )
        else:
            self.lig_edge_mlp = nn.Sequential(
                nn.Linear(lig_edge_mlp_input_dim, self.out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, self.out_feats_dim),
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
            )
        rec_edge_mlp_input_dim = (h_feats_dim * 2) + rec_input_edge_feats_dim
        if self.use_dist_in_layers and self.rec_evolve:
            rec_edge_mlp_input_dim += len(self.all_sigmas_dist)

        # if self.standard_norm_order:
        #     self.rec_edge_mlp = nn.Sequential(
        #         nn.Linear(rec_edge_mlp_input_dim, self.out_feats_dim),
        #         get_layer_norm(layer_norm, self.out_feats_dim),
        #         get_non_lin(nonlin, leakyrelu_neg_slope),
        #         nn.Dropout(dropout),
        #         nn.Linear(self.out_feats_dim, self.out_feats_dim),
        #         get_layer_norm(layer_norm, self.out_feats_dim),
        #     )
        # else:
        #     self.rec_edge_mlp = nn.Sequential(
        #         nn.Linear(rec_edge_mlp_input_dim, self.out_feats_dim),
        #         nn.Dropout(dropout),
        #         get_non_lin(nonlin, leakyrelu_neg_slope),
        #         get_layer_norm(layer_norm, self.out_feats_dim),
        #         nn.Linear(self.out_feats_dim, self.out_feats_dim),
        #     )

        # NODES
        self.node_norm = nn.Identity()  # nn.LayerNorm(h_feats_dim)

        # normalization of x_i - x_j is not currently used
        if self.normalize_coordinate_update:
            self.lig_coords_norm = CoordsNorm(scale_init=1e-2)
            self.rec_coords_norm = CoordsNorm(scale_init=1e-2)
        if self.fine_tune:
            if self.norm_cross_coords_update:
                self.lig_cross_coords_norm = CoordsNorm(scale_init=1e-2)
                self.rec_cross_coords_norm = CoordsNorm(scale_init=1e-2)
            else:
                self.lig_cross_coords_norm =nn.Identity()
                self.rec_cross_coords_norm = nn.Identity()

        self.att_mlp_Q_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V_lig = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        self.att_mlp_Q = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        self.att_mlp_Q_rec2 = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K_rec2 = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V_rec2 = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )

        self.final_h_layernorm_layer_lig = get_norm(self.final_h_layer_norm, out_feats_dim)
        self.final_h_layernorm_layer = get_norm(self.final_h_layer_norm, out_feats_dim)

        self.pre_crossmsg_norm_lig = get_norm(self.pre_crossmsg_norm_type, h_feats_dim)
        self.pre_crossmsg_norm_rec = get_norm(self.pre_crossmsg_norm_type, h_feats_dim)
        self.pre_crossmsg_norm_rec2 = get_norm(self.pre_crossmsg_norm_type, h_feats_dim)

        self.post_crossmsg_norm_lig = get_norm(self.post_crossmsg_norm_type, h_feats_dim)
        self.post_crossmsg_norm_rec = get_norm(self.post_crossmsg_norm_type, h_feats_dim)
        self.post_crossmsg_norm_rec2 = get_norm(self.post_crossmsg_norm_type, h_feats_dim)

        if self.standard_norm_order:
            self.coords_mlp_lig = nn.Sequential(
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                get_layer_norm(layer_norm_coords, self.out_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Dropout(dropout),
                nn.Linear(self.out_feats_dim, 1)
            )
        else:
            self.coords_mlp_lig = nn.Sequential(
                nn.Linear(self.out_feats_dim, self.out_feats_dim),
                nn.Dropout(dropout),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm_coords, self.out_feats_dim),
                nn.Linear(self.out_feats_dim, 1)
            )

        if self.fine_tune:
            self.att_mlp_cross_coors_K = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_Q_lig = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_V_lig = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Linear(h_feats_dim, 1),
            )

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def apply_edges_lig(self, edges):
        if self.use_dist_in_layers and self.lig_evolve:
            x_rel_mag = edges.data['x_rel'] ** 2
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
            return {'msg': self.lig_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
        else:
            return {
                'msg': self.lig_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def apply_edges_rec(self, edges):
        if self.use_dist_in_layers and self.rec_evolve:
            squared_distance = torch.sum(edges.data['x_rel'] ** 2, dim=1, keepdim=True)
            # divide square distance by 10 to have a nicer separation instead of many 0.00000
            x_rel_mag = torch.cat([torch.exp(-(squared_distance / self.rec_square_distance_scale) / sigma) for sigma in
                                   self.all_sigmas_dist], dim=-1)
            return {'msg': self.rec_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
        else:
            return {
                'msg': self.rec_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}

    def update_x_moment_lig(self, edges):
        edge_coef_ligand = self.coords_mlp_lig(edges.data['msg'])  # \phi^x(m_{i->j})
        x_rel = self.lig_coords_norm(edges.data['x_rel']) if self.normalize_coordinate_update else edges.data['x_rel']
        return {'m': x_rel * edge_coef_ligand}  # (x_i - x_j) * \phi^x(m_{i->j})

    def update_x_moment_rec(self, edges):
        edge_coef_rec = self.coords_mlp_rec(edges.data['msg'])  # \phi^x(m_{i->j})
        x_rel = self.rec_coords_norm(edges.data['x_rel']) if self.normalize_coordinate_update else edges.data['x_rel']
        return {'m': x_rel * edge_coef_rec}  # (x_i - x_j) * \phi^x(m_{i->j})

    def attention_coefficients(self, edges):  # for when using cross edges (but this is super slow so dont do it)
        return {'attention_coefficient': torch.sum(edges.dst['q'] * edges.src['k'], dim=1), 'values': edges.src['v']}

    def attention_aggregation(self, nodes):  # for when using cross edges (but this is super slow so dont do it)
        attention = torch.softmax(nodes.mailbox['attention_coefficient'], dim=1)
        return {'cross_attention_feat': torch.sum(attention[:, :, None] * nodes.mailbox['values'], dim=1)}

    def forward(self, lig_graph, rec_graph, rec2_graph, coords_lig, h_feats_lig, original_ligand_node_features, orig_coords_lig,
                coords_rec, h_feats_rec, original_receptor_node_features, orig_coords_rec,
                coords_rec2, h_feats_rec2, original_receptor2_node_features, orig_coords_rec2,
                mask_lig_q, mask_rec_q, mask_rec2_q, geometry_graph):
        with lig_graph.local_scope() and rec_graph.local_scope() and rec2_graph.local_scope():
            lig_graph.ndata['x_now'] = coords_lig
            rec_graph.ndata['x_now'] = coords_rec
            rec2_graph.ndata['x_now'] = coords_rec2
            lig_graph.ndata['feat'] = h_feats_lig  # first time set here
            rec_graph.ndata['feat'] = h_feats_rec
            rec2_graph.ndata['feat'] = h_feats_rec2

            if self.lig_evolve:
                lig_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))  # x_i - x_j
                if self.debug:
                    log(torch.max(lig_graph.edata['x_rel'].abs()), 'x_rel : x_i - x_j')
            if self.rec_evolve:
                rec_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))
                rec2_graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel'))

            lig_graph.apply_edges(self.apply_edges_lig)  ## i->j edge:  [h_i h_j]
            # rec_graph.apply_edges(self.apply_edges_rec)
            # rec2_graph.apply_edges(self.apply_edges_rec)

            h_feats_lig_norm = apply_norm(lig_graph, h_feats_lig, self.pre_crossmsg_norm_type, self.pre_crossmsg_norm_lig)   # [M, C]
            h_feats_rec_norm = apply_norm(rec_graph, h_feats_rec, self.pre_crossmsg_norm_type, self.pre_crossmsg_norm_rec)   # [N1, C]
            h_feats_rec2_norm = apply_norm(rec2_graph, h_feats_rec2, self.pre_crossmsg_norm_type, self.pre_crossmsg_norm_rec2) # [N2, C]

            lig_q_kv = torch.cat((h_feats_rec_norm, h_feats_rec2_norm), dim=0)
            cross_attention_lig_feat = cross_attention(
                self.att_mlp_Q_lig(h_feats_lig_norm),
                self.att_mlp_K(lig_q_kv),
                self.att_mlp_V(lig_q_kv), mask_lig_q, self.cross_msgs)

            rec_q_kv = torch.cat((h_feats_lig_norm, h_feats_rec2_norm), dim=0)
            cross_attention_rec_feat = cross_attention(
                self.att_mlp_Q(h_feats_rec_norm),
                self.att_mlp_K_lig(rec_q_kv),
                self.att_mlp_V_lig(rec_q_kv), mask_rec_q,
                self.cross_msgs)

            rec2_q_kv = torch.cat((h_feats_lig_norm, h_feats_rec_norm), dim=0)
            cross_attention_rec2_feat = cross_attention(
                self.att_mlp_Q_rec2(h_feats_rec2_norm),
                self.att_mlp_K_rec2(rec2_q_kv),
                self.att_mlp_V_rec2(rec2_q_kv), mask_rec2_q,
                self.cross_msgs)

            h_feats_lig_norm = apply_norm(lig_graph, cross_attention_lig_feat, self.post_crossmsg_norm_type,
                                                  self.post_crossmsg_norm_lig)
            h_feats_rec_norm = apply_norm(rec_graph, cross_attention_rec_feat, self.post_crossmsg_norm_type,
                                                  self.post_crossmsg_norm_rec)
            h_feats_rec2_norm = apply_norm(rec2_graph, cross_attention_rec2_feat, self.post_crossmsg_norm_type,
                                                  self.post_crossmsg_norm_rec2)

            if self.lig_evolve:
                lig_graph.update_all(self.update_x_moment_lig, fn.mean('m', 'x_update'))
                # Inspired by https://arxiv.org/pdf/2108.10521.pdf we use original X and not only graph.ndata['x_now']
                x_evolved_lig = self.x_connection_init * orig_coords_lig + (1. - self.x_connection_init) * \
                                lig_graph.ndata['x_now'] + lig_graph.ndata['x_update']
            else:
                x_evolved_lig = coords_lig

            if self.rec_evolve:
                rec_graph.update_all(self.update_x_moment_rec, fn.mean('m', 'x_update'))
                x_evolved_rec = self.x_connection_init * orig_coords_rec + (1. - self.x_connection_init) * \
                                rec_graph.ndata['x_now'] + rec_graph.ndata['x_update']
                rec2_graph.update_all(self.update_x_moment_rec, fn.mean('m', 'x_update'))
                x_evolved_rec2 = self.x_connection_init * orig_coords_rec2 + (1. - self.x_connection_init) * \
                                rec2_graph.ndata['x_now'] + rec2_graph.ndata['x_update']
            else:
                x_evolved_rec = coords_rec
                x_evolved_rec2 = coords_rec2

            lig_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))
            # rec_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))
            # rec2_graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))

            if self.fine_tune:
                lig_q_kv = torch.cat((h_feats_rec_norm, h_feats_rec2_norm), dim=0)
                x_evolved_lig = x_evolved_lig + self.att_mlp_cross_coors_V_lig(h_feats_lig_norm) * (
                        self.lig_cross_coords_norm(lig_graph.ndata['x_now'] - cross_attention(
                            self.att_mlp_cross_coors_Q_lig(h_feats_lig_norm),
                            self.att_mlp_cross_coors_K(lig_q_kv),
                            torch.cat((rec_graph.ndata['x_now'], rec2_graph.ndata['x_now']), dim=0),
                            mask_lig_q, self.cross_msgs)))

            if self.loss_geometry_regularization:
                src, dst = geometry_graph.edges()
                src = src.long()
                dst = dst.long()
                d_squared = torch.sum((x_evolved_lig[src] - x_evolved_lig[dst]) ** 2, dim=1)
                geom_loss = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2)
            else:
                geom_loss = 0. * x_evolved_lig.sum()
            if self.geometry_regularization:
                src, dst = geometry_graph.edges()
                src = src.long()
                dst = dst.long()
                for step in range(self.geom_reg_steps):
                    d_squared = torch.sum((x_evolved_lig[src] - x_evolved_lig[dst]) ** 2, dim=1)
                    Loss = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2)**2) # this is the loss whose gradient we are calculating here
                    grad_d_squared = 2 * (x_evolved_lig[src] - x_evolved_lig[dst])
                    geometry_graph.edata['partial_grads'] = 2 * (d_squared - geometry_graph.edata['feat'] ** 2)[:,None] * grad_d_squared
                    geometry_graph.update_all(fn.copy_e('partial_grads', 'partial_grads_msg'),
                                              fn.sum('partial_grads_msg', 'grad_x_evolved'))
                    grad_x_evolved = geometry_graph.ndata['grad_x_evolved']
                    x_evolved_lig = x_evolved_lig + self.geometry_reg_step_size * grad_x_evolved

            """ return order:
            coords_lig, h_feats_lig, \
            coords_rec, h_feats_rec, \
            coords_rec2, h_feats_rec2, \
            trajectory, geom_loss"""
            return x_evolved_lig, h_feats_lig_norm, \
                    x_evolved_rec, h_feats_rec_norm, \
                    x_evolved_rec2, h_feats_rec2_norm, None, geom_loss


# =================================================================================================================
class IEGMN(nn.Module):

    def __init__(self, n_lays, debug, device, use_rec_atoms, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, residue_emb_dim, iegmn_lay_hid_dim, num_att_heads,
                 dropout, nonlin, leakyrelu_neg_slope, random_vec_dim=0, random_vec_std=1, use_scalar_features=True,
                 num_lig_feats=None, move_keypts_back=False, normalize_Z_lig_directions=False,
                 unnormalized_kpt_weights=False, centroid_keypts_construction_rec=False,
                 centroid_keypts_construction_lig=False, rec_no_softmax=False, lig_no_softmax=False,
                 normalize_Z_rec_directions=False,
                 centroid_keypts_construction=False, evolve_only=False, separate_lig=False, save_trajectories=False, **kwargs):
        super(IEGMN, self).__init__()
        self.mean = torch.tensor([23.43060995, 30.87135408, 37.01917729], requires_grad=False)
        self.std = torch.tensor([13.93822256, 14.37470306, 15.5935168], requires_grad=False)
        self.debug = debug
        self.cross_msgs = cross_msgs
        self.device = device
        self.save_trajectories = save_trajectories
        self.unnormalized_kpt_weights = unnormalized_kpt_weights
        self.separate_lig =separate_lig
        self.use_rec_atoms = use_rec_atoms
        self.noise_decay_rate = noise_decay_rate
        self.noise_initial = noise_initial
        self.use_edge_features_in_gmn = use_edge_features_in_gmn
        self.use_mean_node_features = use_mean_node_features
        self.random_vec_dim = random_vec_dim
        self.random_vec_std = random_vec_std
        self.move_keypts_back = move_keypts_back
        self.normalize_Z_lig_directions = normalize_Z_lig_directions
        self.centroid_keypts_construction = centroid_keypts_construction
        self.centroid_keypts_construction_rec = centroid_keypts_construction_rec
        self.centroid_keypts_construction_lig = centroid_keypts_construction_lig
        self.normalize_Z_rec_directions = normalize_Z_rec_directions
        self.rec_no_softmax = rec_no_softmax
        self.lig_no_softmax = lig_no_softmax
        self.evolve_only = evolve_only

        self.lig_atom_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                             feature_dims=lig_feature_dims, use_scalar_feat=use_scalar_features,
                                             n_feats_to_use=num_lig_feats)
        if self.separate_lig:
            self.lig_separate_atom_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                                 feature_dims=lig_feature_dims, use_scalar_feat=use_scalar_features,
                                                 n_feats_to_use=num_lig_feats)
        if self.use_rec_atoms:
            self.rec_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                            feature_dims=rec_atom_feature_dims, use_scalar_feat=use_scalar_features)
        else:
            self.rec_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                            feature_dims=rec_residue_feature_dims, use_scalar_feat=use_scalar_features)

        input_node_feats_dim = residue_emb_dim
        if self.use_mean_node_features:
            input_node_feats_dim += 5  ### Additional features from mu_r_norm
        self.iegmn_layers = nn.ModuleList()
        self.iegmn_layers.append(
            IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                        h_feats_dim=input_node_feats_dim,
                        out_feats_dim=iegmn_lay_hid_dim,
                        nonlin=nonlin,
                        cross_msgs=self.cross_msgs,
                        leakyrelu_neg_slope=leakyrelu_neg_slope,
                        debug=debug,
                        device=device,
                        dropout=dropout,
                        save_trajectories=save_trajectories,**kwargs))

        if shared_layers:
            interm_lay = IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                                     h_feats_dim=iegmn_lay_hid_dim,
                                     out_feats_dim=iegmn_lay_hid_dim,
                                     cross_msgs=self.cross_msgs,
                                     nonlin=nonlin,
                                     leakyrelu_neg_slope=leakyrelu_neg_slope,
                                     debug=debug,
                                     device=device,
                                     dropout=dropout,
                                     save_trajectories=save_trajectories,**kwargs)
            for layer_idx in range(1, n_lays):
                self.iegmn_layers.append(interm_lay)
        else:
            for layer_idx in range(1, n_lays):
                debug_this_layer = debug if n_lays - 1 == layer_idx else False
                self.iegmn_layers.append(
                    IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                                h_feats_dim=iegmn_lay_hid_dim,
                                out_feats_dim=iegmn_lay_hid_dim,
                                cross_msgs=self.cross_msgs,
                                nonlin=nonlin,
                                leakyrelu_neg_slope=leakyrelu_neg_slope,
                                debug=debug_this_layer,
                                device=device,
                                dropout=dropout,
                                save_trajectories=save_trajectories,**kwargs))
        if self.separate_lig:
            self.iegmn_layers_separate = nn.ModuleList()
            self.iegmn_layers_separate.append(
                IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                            h_feats_dim=input_node_feats_dim,
                            out_feats_dim=iegmn_lay_hid_dim,
                            nonlin=nonlin,
                            cross_msgs=self.cross_msgs,
                            leakyrelu_neg_slope=leakyrelu_neg_slope,
                            debug=debug,
                            device=device,
                            dropout=dropout,
                            save_trajectories=save_trajectories,**kwargs))

            if shared_layers:
                interm_lay = IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                                         h_feats_dim=iegmn_lay_hid_dim,
                                         out_feats_dim=iegmn_lay_hid_dim,
                                         cross_msgs=self.cross_msgs,
                                         nonlin=nonlin,
                                         leakyrelu_neg_slope=leakyrelu_neg_slope,
                                         debug=debug,
                                         device=device,
                                         dropout=dropout,
                                         save_trajectories=save_trajectories,**kwargs)
                for layer_idx in range(1, n_lays):
                    self.iegmn_layers_separate.append(interm_lay)
            else:
                for layer_idx in range(1, n_lays):
                    debug_this_layer = debug if n_lays - 1 == layer_idx else False
                    self.iegmn_layers_separate.append(
                        IEGMN_Layer(orig_h_feats_dim=input_node_feats_dim,
                                    h_feats_dim=iegmn_lay_hid_dim,
                                    out_feats_dim=iegmn_lay_hid_dim,
                                    cross_msgs=self.cross_msgs,
                                    nonlin=nonlin,
                                    leakyrelu_neg_slope=leakyrelu_neg_slope,
                                    debug=debug_this_layer,
                                    device=device,
                                    dropout=dropout,
                                    save_trajectories=save_trajectories,**kwargs))
        # Attention layers
        self.num_att_heads = num_att_heads
        self.out_feats_dim = iegmn_lay_hid_dim
        self.keypts_attention_lig = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))
        self.keypts_queries_lig = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))
        self.keypts_attention_lig2 = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))
        self.keypts_queries_lig2 = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))
        self.keypts_attention_rec = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))
        self.keypts_queries_rec = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))
        self.keypts_attention_rec2 = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))
        self.keypts_queries_rec2 = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False))

        self.h_mean_lig = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.h_mean_rec = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.h_mean_rec2 = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )

        if self.unnormalized_kpt_weights:
            self.scale_lig = nn.Linear(self.out_feats_dim, 1)
            self.scale_rec = nn.Linear(self.out_feats_dim, 1)
            self.scale_rec2 = nn.Linear(self.out_feats_dim, 1)
        # self.reset_parameters()

        if self.normalize_Z_lig_directions:
            self.Z_lig_dir_norm = CoordsNorm()
        if self.normalize_Z_rec_directions:
            self.Z_rec_dir_norm = CoordsNorm()
            self.Z_rec2_dir_norm = CoordsNorm()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def normalize(self, *args) -> Union[Tuple[Tensor], Tensor]:
        self.mean = self.mean.to(args[0].device)
        self.std = self.std.to(args[0].device)
        out = tuple([(i - self.mean) / self.std for i in args])
        return out[0] if len(out) == 1 else out
    
    def unnormalize(self, *args):
        self.mean = self.mean.to(args[0].device)
        self.std = self.std.to(args[0].device)
        out =  tuple([i * self.std + self.mean for i in args])
        return out[0] if len(out) == 1 else out

    def forward(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_names, epoch) -> Tuple[List[Tensor], ...]:
        # normalize start
        lig_graph.ndata['x'] = self.normalize(lig_graph.ndata['x'])
        lig_graph.ndata['new_x'] = self.normalize(lig_graph.ndata['new_x'])
        rec_graph.ndata['x'] = self.normalize(rec_graph.ndata['x'])
        rec2_graph.ndata['x'] = self.normalize(rec2_graph.ndata['x'])
        # geometry_graph.ndata['x'] = self.normalize(geometry_graph.ndata['x'])
        # normalize end

        orig_coords_lig = lig_graph.ndata['new_x']
        orig_coords_rec = rec_graph.ndata['x']
        orig_coords_rec2 = rec2_graph.ndata['x']

        coords_lig = lig_graph.ndata['new_x']
        coords_rec = rec_graph.ndata['x']
        coords_rec2 = rec2_graph.ndata['x']

        h_feats_lig = self.lig_atom_embedder(lig_graph.ndata['feat'])

        if self.use_rec_atoms:
            h_feats_rec = self.rec_embedder(rec_graph.ndata['feat'])
            h_feats_rec2 = self.rec_embedder(rec2_graph.ndata['feat'])
        else:
            h_feats_rec = self.rec_embedder(rec_graph.ndata['feat'])  # (N_res, emb_dim)
            h_feats_rec2 = self.rec_embedder(rec2_graph.ndata['feat'])  # (N_res, emb_dim)

        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        rand_h_lig = rand_dist.sample([h_feats_lig.size(0), self.random_vec_dim]).to(h_feats_lig.device)
        rand_h_rec = rand_dist.sample([h_feats_rec.size(0), self.random_vec_dim]).to(h_feats_lig.device)
        rand_h_rec2 = rand_dist.sample([h_feats_rec2.size(0), self.random_vec_dim]).to(h_feats_lig.device)
        h_feats_lig = torch.cat([h_feats_lig, rand_h_lig], dim=1)
        h_feats_rec = torch.cat([h_feats_rec, rand_h_rec], dim=1)
        h_feats_rec2 = torch.cat([h_feats_rec2, rand_h_rec2], dim=1)

        if self.debug:
            log(torch.max(h_feats_lig.abs()), 'max h_feats_lig before layers and noise ')
            log(torch.max(h_feats_rec.abs()), 'max h_feats_rec before layers and noise ')

        # random noise:
        if self.noise_initial > 0:
            noise_level = self.noise_initial * self.noise_decay_rate ** (epoch + 1)
            h_feats_lig = h_feats_lig + noise_level * torch.randn_like(h_feats_lig)
            h_feats_rec = h_feats_rec + noise_level * torch.randn_like(h_feats_rec)
            h_feats_rec2 = h_feats_rec2 + noise_level * torch.randn_like(h_feats_rec2)
            coords_lig = coords_lig + noise_level * torch.randn_like(coords_lig)
            coords_rec = coords_rec + noise_level * torch.randn_like(coords_rec)
            coords_rec2 = coords_rec2 + noise_level * torch.randn_like(coords_rec2)

        if self.debug:
            log(torch.max(h_feats_lig.abs()), 'h_feats_lig before layers but after noise ')
            log(torch.max(h_feats_rec.abs()), 'h_feats_rec before layers but after noise ')

        if self.use_mean_node_features:
            h_feats_lig = torch.cat([h_feats_lig, torch.log(lig_graph.ndata['mu_r_norm'])],
                                    dim=1)
            h_feats_rec = torch.cat(
                [h_feats_rec, torch.log(rec_graph.ndata['mu_r_norm'])], dim=1)
            h_feats_rec2 = torch.cat(
                [h_feats_rec2, torch.log(rec2_graph.ndata['mu_r_norm'])], dim=1)

        if self.debug:
            log(torch.max(h_feats_lig.abs()), torch.norm(h_feats_lig),
                'max and norm of h_feats_lig before layers but after noise and mu_r_norm')
            log(torch.max(h_feats_rec.abs()), torch.norm(h_feats_lig),
                'max and norm of h_feats_rec before layers but after noise and mu_r_norm')

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
        if self.separate_lig:
            for i, layer in enumerate(self.iegmn_layers_separate):
                if self.debug: log('layer ', i)
                coords_lig_separate, \
                h_feats_lig_separate, \
                coords_rec_separate, \
                h_feats_rec_separate, trajectory, geom_loss = layer(lig_graph=lig_graph,
                                    rec_graph=rec_graph,
                                    coords_lig=coords_lig_separate,
                                    h_feats_lig=h_feats_lig_separate,
                                    original_ligand_node_features=original_ligand_node_features,
                                    orig_coords_lig=orig_coords_lig,
                                    coords_rec=coords_rec_separate,
                                    h_feats_rec=h_feats_rec_separate,
                                    original_receptor_node_features=original_receptor_node_features,
                                    orig_coords_rec=orig_coords_rec,
                                    mask=mask,
                                    geometry_graph=geometry_graph
                                    )
                geom_losses = geom_losses + geom_loss
                # full_trajectory.extend(trajectory)
        if self.save_trajectories:
            save_name = '_'.join(complex_names)
            # torch.save({'trajectories': full_trajectory, 'names': complex_names}, f'data/results/trajectories/{save_name}.pt')
        if self.debug:
            log(torch.max(h_feats_lig.abs()), 'max h_feats_lig after MPNN')
            log(torch.max(coords_lig.abs()), 'max coords_lig before after MPNN')

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
        valid_list = [] # in case checkpoint return NaN 
        rotations_2 = []    # rec2 -> lig
        translations_2 = []
        recs_keypts_2 = []
        ligs_keypts_2 = []
        recs_node_idx_2 = torch.cumsum(rec2_graph.batch_num_nodes(), dim=0).tolist()
        recs_node_idx_2.insert(0, 0)

        if self.evolve_only:
            raise NotImplementedError
            for idx in range(len(ligs_node_idx) - 1):
                lig_start = ligs_node_idx[idx]
                lig_end = ligs_node_idx[idx + 1]
                Z_lig_coords = coords_lig[lig_start:lig_end]
                ligs_evolved.append(Z_lig_coords)
            return [rotations, translations, ligs_keypts, recs_keypts, ligs_evolved, geom_losses]

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
            rec_feats_mean = torch.mean(self.h_mean_rec(rec_feats), dim=0, keepdim=True)  # (1, d)
            rec2_feats = h_feats_rec2[rec2_start:rec2_end]  # (m, d)
            rec2_feats_mean = torch.mean(self.h_mean_rec2(rec2_feats), dim=0, keepdim=True)  # (1, d)
            lig_feats = h_feats_lig[lig_start:lig_end]  # (n, d)
            lig_feats_mean = torch.mean(self.h_mean_lig(lig_feats), dim=0, keepdim=True)  # (1, d)

            d = lig_feats.shape[1]
            assert d == self.out_feats_dim
            # Z coordinates
            Z_rec_coords = coords_rec[rec_start:rec_end]
            Z_rec2_coords = coords_rec2[rec2_start:rec2_end]
            Z_lig_coords = coords_lig[lig_start:lig_end]

            # Att weights to compute the receptor centroid. They query is the average_h_ligand. Keys are each h_receptor_j
            att_weights_rec = (self.keypts_attention_rec(rec_feats).view(-1, self.num_att_heads, d).transpose(0, 1) @
                               self.keypts_queries_rec(lig_feats_mean).view(1, self.num_att_heads, d).transpose(0,
                                                                                                                1).transpose(
                                   1, 2) /
                               math.sqrt(d)).view(self.num_att_heads, -1)
            if not self.rec_no_softmax:
                att_weights_rec = torch.softmax(att_weights_rec, dim=1)
            att_weights_rec = att_weights_rec.view(self.num_att_heads, -1)


            # Att weights to compute the ligand centroid. They query is the average_h_receptor. Keys are each h_ligand_i
            att_weights_lig = (self.keypts_attention_lig(lig_feats).view(-1, self.num_att_heads, d).transpose(0, 1) @
                               self.keypts_queries_lig(rec_feats_mean).view(1, self.num_att_heads, d).transpose(0,
                                                                                                                1).transpose(
                                   1, 2) /
                               math.sqrt(d))
            if not self.lig_no_softmax:
                att_weights_lig = torch.softmax(att_weights_lig, dim=1)
            att_weights_lig = att_weights_lig.view(self.num_att_heads, -1)

            # for rec2
            att_weights_rec2 = (self.keypts_attention_rec2(rec2_feats).view(-1, self.num_att_heads, d).transpose(0, 1) @
                                 self.keypts_queries_rec2(lig_feats_mean).view(1, self.num_att_heads, d).transpose(0,
                                                                                                                 1).transpose(
                                      1, 2) /
                                 math.sqrt(d)).view(self.num_att_heads, -1)
            if not self.rec_no_softmax:
                att_weights_rec2 = torch.softmax(att_weights_rec2, dim=1)
            att_weights_rec2 = att_weights_rec2.view(self.num_att_heads, -1)
            # for lig2
            att_weights_lig2 = (self.keypts_attention_lig2(lig_feats).view(-1, self.num_att_heads, d).transpose(0, 1) @
                                    self.keypts_queries_lig2(rec2_feats_mean).view(1, self.num_att_heads, d).transpose(0,
                                                                                                                    1).transpose(
                                         1, 2) /
                                    math.sqrt(d))
            if not self.lig_no_softmax:
                att_weights_lig2 = torch.softmax(att_weights_lig2, dim=1)
            att_weights_lig2 = att_weights_lig2.view(self.num_att_heads, -1)
            
            if self.unnormalized_kpt_weights:
                lig_scales = self.scale_lig(lig_feats)
                rec_scales = self.scale_rec(rec_feats)
                Z_lig_coords = Z_lig_coords * lig_scales
                Z_rec_coords = Z_rec_coords * rec_scales
                rec2_scales = self.scale_rec2(rec2_feats)
                Z_rec2_coords = Z_rec2_coords * rec2_scales

            valid = True
            if torch.isnan(Z_lig_coords).any():
                valid = False
                Z_lig_coords = torch.nan_to_num(Z_lig_coords)
            if torch.isnan(Z_rec_coords).any():
                valid = False
                Z_rec_coords = torch.nan_to_num(Z_rec_coords)
            if torch.isnan(Z_rec2_coords).any():
                valid = False
                Z_rec2_coords = torch.nan_to_num(Z_rec2_coords)

            if self.centroid_keypts_construction_rec:
                raise NotImplementedError
                Z_rec_mean = Z_rec_coords.mean(dim=0)
                Z_rec_directions = Z_rec_coords - Z_rec_mean
                if self.normalize_Z_rec_directions:
                    Z_rec_directions = self.Z_rec_dir_norm(Z_rec_directions)
                rec_keypts = att_weights_rec @ Z_rec_directions  # K_heads, 3
                if self.move_keypts_back:
                    rec_keypts += Z_rec_mean
            else:
                rec_keypts = att_weights_rec @ Z_rec_coords  # K_heads, 3
                rec2_keypts = att_weights_rec2 @ Z_rec2_coords  # K_heads, 3

            if self.centroid_keypts_construction or self.centroid_keypts_construction_lig:
                raise NotImplementedError
                Z_lig_mean = Z_lig_coords.mean(dim=0)
                Z_lig_directions = Z_lig_coords - Z_lig_mean
                if self.normalize_Z_lig_directions:
                    Z_lig_directions = self.Z_lig_dir_norm(Z_lig_directions)
                lig_keypts = att_weights_lig @ Z_lig_directions  # K_heads, 3
                if self.move_keypts_back:
                    lig_keypts += Z_lig_mean
            else:
                lig_keypts = att_weights_lig @ Z_lig_coords  # K_heads, 3
                lig2_keypts = att_weights_lig2 @ Z_lig_coords  # K_heads, 3

            if torch.isnan(lig_keypts).any() or torch.isinf(lig_keypts).any() \
                or torch.isnan(lig2_keypts).any() or torch.isinf(lig2_keypts).any():
                 log(complex_names[idx], 'Nan encountered before unnormalize')
                 log(Z_lig_coords.max(), Z_rec_coords.max(), Z_rec2_coords.max())

            # unnormalize start
            lig_keypts, rec_keypts, lig2_keypts, rec2_keypts = self.unnormalize(
                lig_keypts, rec_keypts, lig2_keypts, rec2_keypts)
            # unnormalize end

            recs_keypts.append(rec_keypts)
            ligs_keypts.append(lig_keypts)
            recs_keypts_2.append(rec2_keypts)
            ligs_keypts_2.append(lig2_keypts)

            if not valid:
                # fake result
                rotations.append(None)
                translations.append(None)
                rotations_2.append(None)
                translations_2.append(None)
                ligs_evolved.append(Z_lig_coords)
                valid_list.append(False)
                continue
            else:
                valid_list.append(True)

            ## Apply Kabsch algorithm
            rotation, translation = model_kabsch(
                from_points=lig_keypts, dst_points=rec_keypts,
                num_att_heads=self.num_att_heads, complex_names=complex_names, device=self.device)
            rotations.append(rotation)
            translations.append(translation)
            # (rotation @ evolved_ligs[idx].t()).t() + translation
            moved_lig_coords = (rotation @ Z_lig_coords.T).T + translation
            moved_lig2_keypts = (rotation @ lig2_keypts.t()).t() + translation
            rotation2, translation2 = model_kabsch(
                from_points=rec2_keypts, dst_points=moved_lig2_keypts,
                num_att_heads=self.num_att_heads, complex_names=complex_names, device=self.device)
            rotations_2.append(rotation2)
            translations_2.append(translation2)
    
            # if self.separate_lig:
            #     raise NotImplementedError
            #     Z_lig_coords = coords_lig_separate[lig_start:lig_end]
            
            ligs_evolved.append(moved_lig_coords)

        # ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss \
        #    ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2
        return ligs_evolved, ligs_keypts, recs_keypts, rotations, translations, geom_losses, \
                ligs_keypts_2, recs_keypts_2, rotations_2, translations_2, valid_list

    def __repr__(self):
        return "IEGMN " + str(self.__dict__)


# =================================================================================================================


@MODELS.register_module()
class TripletDock(BaseModel):

    def __init__(self, iegmn_class, device='cuda', debug=False, use_evolved_lig=False, evolve_only=False, loss_cfg=dict(), **kwargs):
        super().__init__()
        self.debug = debug
        self.evolve_only = evolve_only
        self.use_evolved_lig = use_evolved_lig
        self.device = device
        # self.iegmn = IEGMN(device=self.device, debug=self.debug, evolve_only=self.evolve_only, **kwargs)
        self.build_layers(iegmn_class, device, debug, evolve_only, **kwargs)
        self.loss_func = TripletBindingLoss(**loss_cfg)
        # self.load_pretrained()
    
    def build_layers(self, iegmn_class, device, debug, evolve_only, **kwargs):
        iegmn_module = MODELS.build(dict(type=iegmn_class, device=device, debug=debug, evolve_only=evolve_only, **kwargs))
        self.iegmn = iegmn_module

    def load_pretrained(self):
        """Load EquiBind pretrained."""
        log('Load state dict from Equibind pretrain...')
        equibind_ckpt = torch.load('weights/EquiBind_flexible_self_docking_best_checkpoint.pt')
        equibind_states = equibind_ckpt['model_state_dict']
        lig_atom_embed_weight = torch.zeros(7, 64)
        lig_atom_embed_weight[:4] = equibind_states['iegmn.lig_atom_embedder.atom_embedding_list.1.weight']
        equibind_states['iegmn.lig_atom_embedder.atom_embedding_list.1.weight'] = lig_atom_embed_weight
        copy_keys = {
            'iegmn.keypts_attention_lig2.0.weight': 'iegmn.keypts_attention_lig.0.weight',
            'iegmn.keypts_queries_lig2.0.weight': 'iegmn.keypts_queries_lig.0.weight',
            'iegmn.keypts_attention_rec2.0.weight': 'iegmn.keypts_attention_rec.0.weight',
            'iegmn.keypts_queries_rec2.0.weight': 'iegmn.keypts_queries_rec.0.weight',
            'iegmn.h_mean_rec2.0.weight': 'iegmn.h_mean_rec.0.weight',
            'iegmn.h_mean_rec2.0.bias': 'iegmn.h_mean_rec.0.bias',
        }
        for i in range(len(self.iegmn.iegmn_layers)):
            copy_keys.update({
                f'iegmn.iegmn_layers.{i}.att_mlp_Q_rec2.0.weight': f'iegmn.iegmn_layers.{i}.att_mlp_Q.0.weight',
                f'iegmn.iegmn_layers.{i}.att_mlp_K_rec2.0.weight': f'iegmn.iegmn_layers.{i}.att_mlp_K.0.weight',
                f'iegmn.iegmn_layers.{i}.att_mlp_V_rec2.0.weight': f'iegmn.iegmn_layers.{i}.att_mlp_V.0.weight',
            })
        for k, v in copy_keys.items():
            equibind_states[k] = equibind_states[v]
        
        # delete unused linear
        del equibind_states['iegmn.lig_atom_embedder.linear.weight']
        del equibind_states['iegmn.lig_atom_embedder.linear.bias']
        del equibind_states['iegmn.rec_embedder.linear.weight']
        del equibind_states['iegmn.rec_embedder.linear.bias']
        load_state_dict(self, equibind_states, strict=False)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)
    
    def extract_feats(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_name, **kwargs):
        message_hub = MessageHub.get_current_instance()
        epoch = message_hub.get_info('epoch')
        lig_graphs = lig_graph.to(self.device)
        rec_graphs = rec_graph.to(self.device)
        rec2_graphs = rec2_graph.to(self.device)
        geometry_graphs = geometry_graph.to(self.device)

        # ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss, \
        #     ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2 
        output = self.forward_bind(
            lig_graphs, rec_graphs, rec2_graphs, geometry_graphs,
            complex_names=complex_name,
            epoch=epoch,
            **kwargs
        )
        return output

    # def forward(self, lig_graphs, rec_graphs, ligs_coords, recs_coords, ligs_pocket_coords, recs_pocket_coords, geometry_graphs, complex_names, idx, mode='loss'):
    def forward(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_name, mode='loss', **kwargs):
    
        ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss, \
            ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2, valid, p2_rmsd_preds = self.extract_feats(
                lig_graph, rec_graph, rec2_graph, geometry_graph, complex_name, **kwargs)

        if mode == 'loss':
            """
            # pred
            ligs_coords_pred, geom_reg_loss,
            ligs_keypts, recs_keypts, rotations, translations,
            ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2,
            # gt
            ligs_coords, recs_coords, rec2_coords,
            ligs_pocket_coords, recs_pocket_coords, 
            ligs_pocket_coords_2, rec2_pocket_coords_2,
            """
            lig_coords = kwargs['lig_coords']
            rec_coords = kwargs['rec_coords']
            rec2_coords = kwargs['rec2_coords']
            rec2_coords_input = kwargs['rec2_coords_input']

            # p1lig pocket
            p1lig_lig_pocket_coords = kwargs['p1lig_lig_pocket_coords']
            p1lig_p1_pocket_coords = kwargs['p1lig_p1_pocket_coords']

            # p2lig pocket
            p2lig_lig_pocket_coords = kwargs['p2lig_lig_pocket_coords']
            p2lig_p2_pocket_coords = kwargs['p2lig_p2_pocket_coords']

            # p12 pocket
            p12_p1_pocket_coords = kwargs['p12_p1_pocket_coords']
            p12_p2_pocket_coords = kwargs['p12_p2_pocket_coords']

            # lig_pocket_coords = kwargs['lig_pocket_coords']
            # rec_pocket_coords = kwargs['rec_pocket_coords']
            # lig2_pocket_coords = kwargs['lig2_pocket_coords']
            # rec2_pocket_coords = kwargs['rec2_pocket_coords']
            align_method = getattr(self.iegmn, 'align_method')
            if align_method == 'p1-lig-p2':
                lig2_pocket_gt = p2lig_lig_pocket_coords
                rec2_pocket_gt = p2lig_p2_pocket_coords
            elif align_method == 'p1-p2':
                lig2_pocket_gt = p12_p1_pocket_coords
                rec2_pocket_gt = p12_p2_pocket_coords
            else:
                raise ValueError(f'align_method {align_method} not supported')

            losses = self.loss_func(
                ligs_coords_pred, geom_reg_loss,
                ligs_keypts, recs_keypts, rotations, translations,
                ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2, valid, p2_rmsd_preds,
                # gt
                lig_coords, rec_coords, rec2_coords, rec2_coords_input,
                # lig_pocket_coords, rec_pocket_coords,
                # lig2_pocket_coords, rec2_pocket_coords,
                p1lig_lig_pocket_coords, p1lig_p1_pocket_coords,
                lig2_pocket_gt, rec2_pocket_gt,
             )
            return losses

        # for test
        result =  []
        for i in range(len(ligs_coords_pred)):
            result.append({
                'ligs_coords_pred': ligs_coords_pred[i].detach().cpu(),
                'rotation_2': rotations_2[i].detach().cpu(),
                'translation_2': translations_2[i].detach().cpu(),
                'p2_rmsd_pred': p2_rmsd_preds[i].detach().cpu(),
            })
        return result

    def forward_encoder(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_name, epoch=0, **kwargs):
        return self.iegmn.forward_encoder(
            lig_graph, rec_graph, rec2_graph, geometry_graph, complex_names=complex_name, epoch=epoch, **kwargs)

    def forward_bind(self, lig_graph, rec_graph, rec2_graph, geometry_graph=None, complex_names=None, epoch=0, **kwargs):
        # predicted_ligs_coords_list = []
        outputs = self.iegmn(lig_graph, rec_graph, rec2_graph, geometry_graph, complex_names, epoch, **kwargs)
        return outputs
        evolved_ligs = outputs[4]
        if self.evolve_only:
            return evolved_ligs, outputs[2], outputs[3], outputs[0], outputs[1], outputs[5]
        ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
        ligs_node_idx.insert(0, 0)
        for idx in range(len(ligs_node_idx) - 1):
            start = ligs_node_idx[idx]
            end = ligs_node_idx[idx + 1]
            orig_coords_lig = lig_graph.ndata['new_x'][start:end]
            rotation = outputs[0][idx]
            translation = outputs[1][idx]
            assert translation.shape[0] == 1 and translation.shape[1] == 3

            if self.use_evolved_lig:
                predicted_coords = (rotation @ evolved_ligs[idx].t()).t() + translation  # (n,3)
            else:
                predicted_coords = (rotation @ orig_coords_lig.t()).t() + translation  # (n,3)
            if self.debug:
                log('rotation', rotation)
                log('rotation @ rotation.t() - eye(3)', rotation @ rotation.t() - torch.eye(3).to(self.device))
                log('translation', translation)
                log('\n ---> predicted_coords mean - true ligand mean ',
                    predicted_coords.mean(dim=0) - lig_graph.ndata['x'][
                                                   start:end].mean(dim=0), '\n')
            predicted_ligs_coords_list.append(predicted_coords)
        #torch.save({'predictions': predicted_ligs_coords_list, 'names': complex_names})
        return predicted_ligs_coords_list, outputs[2], outputs[3], outputs[0], outputs[1], outputs[5]

    def __repr__(self):
        return "EquiBind " + str(self.__dict__)


@MODELS.register_module()
class IEGMN_QueryDecoder(IEGMN):
    """Use keypoint queries to generate keypoints"""

    def __init__(
            self,
            *args,
            align_method='p1-lig-p2',   # p1-p2
            known_pocket_info=(),
            kpt_transformer_depth=1,
            kpt_transformer_heads=1,
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
        self.keypts_query_embeddings = nn.Embedding(self.num_att_heads * 4 + 1, self.out_feats_dim)  # + 1 p2 rmsd predictor
        self.feat_type_embeddings = nn.Embedding(3, self.out_feats_dim)
        # self.gt_type_embeddings = nn.Embedding(4, self.out_feats_dim)
        self.kpt_transformer = TwoWayTransformer(depth=kpt_transformer_depth, embedding_dim=self.out_feats_dim,
            num_heads=kpt_transformer_heads, mlp_dim=self.out_feats_dim * 4)
        self.keypts_cross_attn = nn.ModuleList([CoordAttention(self.out_feats_dim, num_heads=1) for _ in range(4)])

        # p2 rmsd predictor
        # self.p2_rmsd_query = nn.Embedding(1, self.out_feats_dim)
        self.p2_rmsd_pred_head = MLP(input_dim=self.out_feats_dim, hidden_dim=self.out_feats_dim, output_dim=1, num_layers=3, sigmoid_output=False)
        # self.coord2feat = MLP(input_dim=3, hidden_dim=self.out_feats_dim, output_dim=self.out_feats_dim, num_layers=3, sigmoid_output=False)
        # self.feat_downsample = MLP(input_dim=self.out_feats_dim * 2, hidden_dim=self.out_feats_dim, output_dim=self.out_feats_dim, num_layers=3, sigmoid_output=False)

    def forward_encoder(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_names, epoch, **kwargs):
        """extract encoder feature for downstream analysis, such as t-SNE visualization"""
        
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
            # align gt to input
            # R, t = model_kabsch(from_points=p1lig_lig_pocket_coords, dst_points=lig_graph.ndata['new_x'][p1lig_lig_pocket_mask])
            # lig1_coord = rotate_and_translate(p1lig_lig_pocket_coords, R, t)
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
        lig_graph.ndata['feat'] = h_feats_lig
        rec_graph.ndata['feat'] = h_feats_rec
        rec2_graph.ndata['feat'] = h_feats_rec2
        return lig_graph, rec_graph, rec2_graph

    def forward(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_names, epoch, **kwargs) -> Tuple[List[Tensor], ...]:
        SAVE_LIG_RESULT = False
        if SAVE_LIG_RESULT:
            # save lig
            from biopandas.pdb import PandasPdb
            from projects.equibind.models.path import PDB_PATH
            assert len(complex_names) == 1
            name = complex_names[0]
            lig_path = os.path.join(PDB_PATH, name, 'ligand.pdb')
            lig_pdb = PandasPdb().read_pdb(lig_path)
            # save gt
            lig_pdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']] = lig_graph.ndata['x'].cpu().numpy()
            lig_pdb.to_pdb(f'output/lig_visualize/gt.pdb', records=['HETATM'], gz=False)
            # save rdkit
            lig_pdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']] = lig_graph.ndata['new_x'].cpu().numpy()
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
            # align gt to input
            # R, t = model_kabsch(from_points=p1lig_lig_pocket_coords, dst_points=lig_graph.ndata['new_x'][p1lig_lig_pocket_mask])
            # lig1_coord = rotate_and_translate(p1lig_lig_pocket_coords, R, t)
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
            if not self.training:
                if 'lig1_coord' in self.known_pocket_info:
                    p1lig_lig_pocket_mask = torch.cat(kwargs['p1lig_lig_pocket_mask'])
                    p1lig_lig_pocket_coords = torch.cat(kwargs['p1lig_lig_pocket_coords']).to(lig_graph.device)
                    p1lig_lig_pocket_coords = self.normalize(p1lig_lig_pocket_coords)
                    assert len(p1lig_lig_pocket_mask) == len(lig_graph.ndata['new_x'])
                    lig_graph.ndata['new_x'][p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
                    coords_lig[p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
                if 'lig2_coord' in self.known_pocket_info:
                    p2lig_lig_pocket_coords = [self.normalize(i).to(lig_graph.device) for i in kwargs['p2lig_lig_pocket_coords']]
                    # align batch gt to input
                    lig_graph.ndata['new_x'] = rigid_align_batch_pockets(
                        old_batch_all_coords=lig_graph.ndata['new_x'],
                        batch_pocket_masks=kwargs['p2lig_lig_pocket_mask'],
                        batch_pocket_coords=p2lig_lig_pocket_coords
                    )
                    coords_lig = lig_graph.ndata['new_x']

            if SAVE_LIG_RESULT:
                # save layer
                lig_pdb.df['HETATM'][['x_coord', 'y_coord', 'z_coord']] = self.unnormalize(coords_lig).cpu().numpy()
                lig_pdb.to_pdb(f'output/lig_visualize/layer_{i}.pdb', records=['HETATM'], gz=False)

        if self.separate_lig:
            raise NotImplementedError
            for i, layer in enumerate(self.iegmn_layers_separate):
                if self.debug: log('layer ', i)
                coords_lig_separate, \
                h_feats_lig_separate, \
                coords_rec_separate, \
                h_feats_rec_separate, trajectory, geom_loss = layer(lig_graph=lig_graph,
                                    rec_graph=rec_graph,
                                    coords_lig=coords_lig_separate,
                                    h_feats_lig=h_feats_lig_separate,
                                    original_ligand_node_features=original_ligand_node_features,
                                    orig_coords_lig=orig_coords_lig,
                                    coords_rec=coords_rec_separate,
                                    h_feats_rec=h_feats_rec_separate,
                                    original_receptor_node_features=original_receptor_node_features,
                                    orig_coords_rec=orig_coords_rec,
                                    mask=mask,
                                    geometry_graph=geometry_graph
                                    )
                geom_losses = geom_losses + geom_loss
                full_trajectory.extend(trajectory)
        if self.save_trajectories:
            save_name = '_'.join(complex_names)
            # torch.save({'trajectories': full_trajectory, 'names': complex_names}, f'data/results/trajectories/{save_name}.pt')
        if self.debug:
            log(torch.max(h_feats_lig.abs()), 'max h_feats_lig after MPNN')
            log(torch.max(coords_lig.abs()), 'max coords_lig before after MPNN')

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

            queries = self.keypts_query_embeddings.weight

            src = torch.cat((lig_feats, rec_feats, rec2_feats), dim=0)
            A, B, C = len(lig_feats), len(rec_feats), len(rec2_feats)
            lig_feats_pe = self.feat_type_embeddings.weight[0].repeat(A, 1)
            rec_feats_pe = self.feat_type_embeddings.weight[1].repeat(B, 1)
            rec2_feats_pe = self.feat_type_embeddings.weight[2].repeat(C, 1)
            src_pe = torch.cat((lig_feats_pe, rec_feats_pe, rec2_feats_pe), dim=0)

            queries, keys = self.kpt_transformer(src[None, ...], src_pe[None, ...], queries[None, ...])
            queries = queries[0]
            keys = keys[0]

            assert len(queries) == 4 * self.num_att_heads + 1
            query_list = []
            for i in range(4):
                start_idx = i * self.num_att_heads
                end_idx = (i + 1) * self.num_att_heads
                query_list.append(queries[start_idx:end_idx])

            p2_rmsd_pred = self.p2_rmsd_pred_head(queries[-1:])[0][0]
            p2_rmsd_preds.append(p2_rmsd_pred)

            keys  = keys + src_pe   # only for keys
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

            # 只算自己好，还是算上邻近的好(结果差不多)
            # lig1_keypts = self.keypts_cross_attn[0](query_list[0][None, ...], torch.cat((lig_feats, rec_feats))[None, ...], torch.cat((Z_lig_coords, Z_rec_coords))[None, ...])
            # rec_keypts = self.keypts_cross_attn[1](query_list[1][None, ...], torch.cat((rec_feats, lig_feats))[None, ...], torch.cat((Z_rec_coords, Z_lig_coords))[None, ...])
            # if self.align_method == 'p1-lig-p2':
            #     lig2_keypts = self.keypts_cross_attn[2](query_list[2][None, ...], torch.cat((lig_feats, rec2_feats))[None, ...], torch.cat((Z_lig_coords, Z_rec2_coords))[None, ...])
            #     rec2_keypts = self.keypts_cross_attn[3](query_list[3][None, ...], torch.cat((rec2_feats, lig_feats))[None, ...], torch.cat((Z_rec2_coords, Z_lig_coords))[None, ...])
            # elif self.align_method == 'p1-p2':
            #     lig2_keypts = self.keypts_cross_attn[2](query_list[2][None, ...], torch.cat((rec_feats, rec2_feats))[None, ...], torch.cat((Z_rec_coords, Z_rec2_coords))[None, ...])
            #     rec2_keypts = self.keypts_cross_attn[3](query_list[3][None, ...], torch.cat((rec2_feats, rec_feats))[None, ...], torch.cat((Z_rec2_coords, Z_rec_coords))[None, ...])
            # else:
            #     raise ValueError('Unknown align method: {}'.format(self.align_method))

            lig1_keypts = self.keypts_cross_attn[0](query_list[0][None, ...], lig_feats[None], Z_lig_coords[None])
            rec_keypts = self.keypts_cross_attn[1](query_list[1][None, ...], rec_feats[None], Z_rec_coords[None])
            if self.align_method == 'p1-lig-p2':
                raise NotImplementedError
            elif self.align_method == 'p1-p2':
                lig2_keypts = self.keypts_cross_attn[2](query_list[2][None, ...], rec_feats[None], Z_rec_coords[None])
                rec2_keypts = self.keypts_cross_attn[3](query_list[3][None, ...], rec2_feats[None], Z_rec2_coords[None])

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

            recs_keypts.append(rec_keypts)
            ligs_keypts.append(lig_keypts)
            recs_keypts_2.append(rec2_keypts)
            ligs_keypts_2.append(lig2_keypts)

            if torch.isnan(lig_keypts).any() or torch.isinf(lig_keypts).any() \
                or torch.isnan(lig2_keypts).any() or torch.isinf(lig2_keypts).any() \
                or torch.isnan(rec_keypts).any() or torch.isinf(rec_keypts).any() \
                or torch.isnan(rec2_keypts).any() or torch.isinf(rec2_keypts).any():
                    log(complex_names, 'complex_names where Nan or inf encountered')
                    # fake result
                    rotations.append(None)
                    translations.append(None)
                    rotations_2.append(None)
                    translations_2.append(None)
                    ligs_evolved.append(None)
                    valid.append(False)
                    continue
            else:
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
    
            # if self.separate_lig:
            #     raise NotImplementedError
            #     Z_lig_coords = coords_lig_separate[lig_start:lig_end]
            
            ligs_evolved.append(moved_lig_coords)

        # ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss \
        #    ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2
        return ligs_evolved, ligs_keypts, recs_keypts, rotations, translations, geom_losses, \
                ligs_keypts_2, recs_keypts_2, rotations_2, translations_2, valid, p2_rmsd_preds


@MODELS.register_module()
class IEGMN_QueryDecoder_Seperate(IEGMN_QueryDecoder):
    """
    do not use batched graphs
    """
    def forward(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_names, epoch, **kwargs) -> Tuple[List[Tensor], ...]:
        lig_graphs = dgl.unbatch(lig_graph)
        rec_graphs = dgl.unbatch(rec_graph)
        rec2_graph = dgl.unbatch(rec2_graph)
        geometry_graph = dgl.unbatch(geometry_graph)
        output_list = []    # [(tuple, ), (tuple, )]
        for i in range(len(lig_graphs)):
            single_kwargs = dict()
            for k, v in kwargs.items():
                assert len(v) == len(lig_graphs)
                single_kwargs[k] = v[i]
            output_list.append(
                self.forward_single(lig_graphs[i], rec_graphs[i], rec2_graph[i], geometry_graph[i],
                                    complex_names[i], epoch, **single_kwargs))
        
        return_list = tuple(list(x) for x in zip(*output_list))
        return return_list

    def forward_single(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_name, epoch, **kwargs):
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
            p1lig_p1_pocket_mask = kwargs['p1lig_p1_pocket_mask']
            h_feats_rec[p1lig_p1_pocket_mask] += self.p1lig_pock_gt_embed.weight[1][None, ]
            h_feats_rec[~p1lig_p1_pocket_mask] += self.p1lig_pock_gt_embed.weight[0][None, ]

        if 'lig1_mask' in self.known_pocket_info:
            p1lig_lig_pocket_mask = kwargs['p1lig_lig_pocket_mask']
            h_feats_lig[p1lig_lig_pocket_mask] += self.p1lig_pock_gt_embed.weight[1][None, ]
            h_feats_lig[~p1lig_lig_pocket_mask] += self.p1lig_pock_gt_embed.weight[0][None, ]

        if 'lig1_coord' in self.known_pocket_info:
            assert 'lig1_mask' in self.known_pocket_info
            p1lig_lig_pocket_mask = kwargs['p1lig_lig_pocket_mask']
            p1lig_lig_pocket_coords = kwargs['p1lig_lig_pocket_coords'].to(lig_graph.device)
            p1lig_lig_pocket_coords = self.normalize(p1lig_lig_pocket_coords)
            assert len(p1lig_lig_pocket_mask) == len(lig_graph.ndata['new_x'])
            # align gt to input
            # R, t = model_kabsch(from_points=p1lig_lig_pocket_coords, dst_points=lig_graph.ndata['new_x'][p1lig_lig_pocket_mask])
            # lig1_coord = rotate_and_translate(p1lig_lig_pocket_coords, R, t)
            lig_graph.ndata['new_x'][p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords

        if 'lig2_mask' in self.known_pocket_info:
            p2lig_lig_pocket_mask = kwargs['p2lig_lig_pocket_mask']
            h_feats_lig[p2lig_lig_pocket_mask] += self.p2lig_pock_gt_embed.weight[1][None, ]
            h_feats_lig[~p2lig_lig_pocket_mask] += self.p2lig_pock_gt_embed.weight[0][None, ]
        
        if 'lig2_coord' in self.known_pocket_info:
            p2lig_lig_pocket_mask = kwargs['p2lig_lig_pocket_mask']
            p2lig_lig_pocket_coords = kwargs['p2lig_lig_pocket_coords'].to(lig_graph.device)
            p2lig_lig_pocket_coords = self.normalize(p2lig_lig_pocket_coords)
            assert len(p2lig_lig_pocket_mask) == len(lig_graph.ndata['new_x'])
            # align gt to input
            R, t = model_kabsch(from_points=p2lig_lig_pocket_coords, dst_points=lig_graph.ndata['new_x'][p2lig_lig_pocket_mask])
            lig2_coord = rotate_and_translate(p2lig_lig_pocket_coords, R, t)
            lig_graph.ndata['new_x'][p2lig_lig_pocket_mask] = lig2_coord
        
        if 'p2_mask' in self.known_pocket_info:
            p2lig_p2_pocket_mask = kwargs['p2lig_p2_pocket_mask']
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
            if 'lig1_coord' in self.known_pocket_info:
                p1lig_lig_pocket_mask = kwargs['p1lig_lig_pocket_mask']
                p1lig_lig_pocket_coords = kwargs['p1lig_lig_pocket_coords'].to(lig_graph.device)
                p1lig_lig_pocket_coords = self.normalize(p1lig_lig_pocket_coords)
                assert len(p1lig_lig_pocket_mask) == len(lig_graph.ndata['new_x'])
                # align gt to input
                # R, t = model_kabsch(from_points=p1lig_lig_pocket_coords, dst_points=lig_graph.ndata['new_x'][p1lig_lig_pocket_mask])
                # lig1_coord = rotate_and_translate(p1lig_lig_pocket_coords, R, t)
                lig_graph.ndata['new_x'][p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
                coords_lig[p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
            if 'lig2_coord' in self.known_pocket_info:
                p2lig_lig_pocket_mask = kwargs['p2lig_lig_pocket_mask']
                p2lig_lig_pocket_coords = kwargs['p2lig_lig_pocket_coords'].to(lig_graph.device)
                p2lig_lig_pocket_coords = self.normalize(p2lig_lig_pocket_coords)
                assert len(p2lig_lig_pocket_mask) == len(lig_graph.ndata['new_x'])
                # align gt to input
                R, t = model_kabsch(from_points=p2lig_lig_pocket_coords, dst_points=lig_graph.ndata['new_x'][p2lig_lig_pocket_mask])
                lig2_coord = rotate_and_translate(p2lig_lig_pocket_coords, R, t)
                lig_graph.ndata['new_x'][p2lig_lig_pocket_mask] = lig2_coord
                coords_lig[p2lig_lig_pocket_mask] = lig2_coord

        if self.separate_lig:
            raise NotImplementedError
        if self.save_trajectories:
            raise NotImplementedError


        rec_feats = h_feats_rec  # (m, d)
        rec2_feats = h_feats_rec2  # (m, d)
        lig_feats = h_feats_lig  # (n, d)

        d = lig_feats.shape[1]
        assert d == self.out_feats_dim
        # Z coordinates
        Z_rec_coords = coords_rec
        Z_rec2_coords = coords_rec2
        Z_lig_coords = coords_lig

        queries = self.keypts_query_embeddings.weight

        src = torch.cat((lig_feats, rec_feats, rec2_feats), dim=0)
        A, B, C = len(lig_feats), len(rec_feats), len(rec2_feats)
        lig_feats_pe = self.feat_type_embeddings.weight[0].repeat(A, 1)
        rec_feats_pe = self.feat_type_embeddings.weight[1].repeat(B, 1)
        rec2_feats_pe = self.feat_type_embeddings.weight[2].repeat(C, 1)
        src_pe = torch.cat((lig_feats_pe, rec_feats_pe, rec2_feats_pe), dim=0)

        queries, keys = self.kpt_transformer(src[None, ...], src_pe[None, ...], queries[None, ...])
        queries = queries[0]
        keys = keys[0]

        query_list = []
        for i in range(4):
            start_idx = i * self.num_att_heads
            end_idx = (i + 1) * self.num_att_heads
            query_list.append(queries[start_idx:end_idx])
            
        keys  = keys + src_pe   # only for keys
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
        lig1_keypts = self.keypts_cross_attn[0](query_list[0][None, ...], torch.cat((lig_feats, rec_feats))[None, ...], torch.cat((Z_lig_coords, Z_rec_coords))[None, ...])
        rec_keypts = self.keypts_cross_attn[1](query_list[1][None, ...], torch.cat((rec_feats, lig_feats))[None, ...], torch.cat((Z_rec_coords, Z_lig_coords))[None, ...])
        if self.align_method == 'p1-lig-p2':
            lig2_keypts = self.keypts_cross_attn[2](query_list[2][None, ...], torch.cat((lig_feats, rec2_feats))[None, ...], torch.cat((Z_lig_coords, Z_rec2_coords))[None, ...])
            rec2_keypts = self.keypts_cross_attn[3](query_list[3][None, ...], torch.cat((rec2_feats, lig_feats))[None, ...], torch.cat((Z_rec2_coords, Z_lig_coords))[None, ...])
        elif self.align_method == 'p1-p2':
            lig2_keypts = self.keypts_cross_attn[2](query_list[2][None, ...], torch.cat((rec_feats, rec2_feats))[None, ...], torch.cat((Z_rec_coords, Z_rec2_coords))[None, ...])
            rec2_keypts = self.keypts_cross_attn[3](query_list[3][None, ...], torch.cat((rec2_feats, rec_feats))[None, ...], torch.cat((Z_rec2_coords, Z_rec_coords))[None, ...])
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

        # update Z_lig_coords
        # if not self.training:
        if 'lig1_coord' in self.known_pocket_info:
            p1lig_lig_pocket_mask = kwargs['p1lig_lig_pocket_mask']
            p1lig_lig_pocket_coords = kwargs['p1lig_lig_pocket_coords'].to(lig_graph.device)
            Z_lig_coords[p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
        if 'lig2_coord' in self.known_pocket_info:
            p2lig_lig_pocket_mask = kwargs['p2lig_lig_pocket_mask']
            p2lig_lig_pocket_coords = kwargs['p2lig_lig_pocket_coords'].to(lig_graph.device)
            R, t = model_kabsch(from_points=p2lig_lig_pocket_coords, dst_points=Z_lig_coords[p2lig_lig_pocket_mask])
            lig2_coord = rotate_and_translate(p2lig_lig_pocket_coords, R, t)
            Z_lig_coords[p2lig_lig_pocket_mask] = lig2_coord

        # if both protein and ligand masks are known, use coords as keypoints
        # if not self.training:
        if 'p1_mask' in self.known_pocket_info and \
            'lig1_mask' in self.known_pocket_info:
            coord = kwargs['p1lig_p1_pocket_coords'].to(lig_graph.device)
            rec_keypts = coord + 0. * rec_keypts.sum()
            lig_keypts = Z_lig_coords[kwargs['p1lig_lig_pocket_mask']] + 0. * lig_keypts.sum()
        if 'p1_mask' in self.known_pocket_info and \
            'lig1_coord' in self.known_pocket_info:    # overwrite the above if
            coord = kwargs['p1lig_p1_pocket_coords'].to(lig_graph.device)
            rec_keypts = coord + 0. * rec_keypts.sum()
            p1lig_lig_pocket_coords = kwargs['p1lig_lig_pocket_coords'].to(lig_graph.device)
            lig_keypts = p1lig_lig_pocket_coords + 0. * lig_keypts.sum()

        if 'p2_mask' in self.known_pocket_info and \
            'lig2_mask' in self.known_pocket_info:
            coord = kwargs['p2lig_p2_pocket_coords'].to(lig_graph.device)
            rec2_keypts = coord + 0. * rec2_keypts.sum()
            lig2_keypts = Z_lig_coords[kwargs['p2lig_lig_pocket_mask']] + 0. * lig2_keypts.sum()
        if 'p2_mask' in self.known_pocket_info and \
            'lig2_coord' in self.known_pocket_info: # overwrite
            coord = kwargs['p2lig_p2_pocket_coords'].to(lig_graph.device)
            rec2_keypts = coord + 0. * rec2_keypts.sum()
            # TODO: which one is better?
            # lig2_keypts = Z_lig_coords[kwargs['p2lig_lig_pocket_mask']] + 0. * lig2_keypts.sum()
            p2lig_lig_pocket_mask = kwargs['p2lig_lig_pocket_mask']
            p2lig_lig_pocket_coords = kwargs['p2lig_lig_pocket_coords'].to(lig_graph.device)
            R, t = model_kabsch(from_points=p2lig_lig_pocket_coords, dst_points=Z_lig_coords[p2lig_lig_pocket_mask])
            lig2_keypts = rotate_and_translate(p2lig_lig_pocket_coords, R, t) + 0. * lig2_keypts.sum()

        if torch.isnan(lig_keypts).any() or torch.isinf(lig_keypts).any() \
            or torch.isnan(lig2_keypts).any() or torch.isinf(lig2_keypts).any() \
            or torch.isnan(rec_keypts).any() or torch.isinf(rec_keypts).any() \
            or torch.isnan(rec2_keypts).any() or torch.isinf(rec2_keypts).any():
                log(complex_name, 'complex_name where Nan or inf encountered')
                # fake result
                rotation, translation = None, None
                rotation2, translation2 = None, None
                lig_evolved = None
                valid = False
        else:
            valid = True

        ## Apply Kabsch algorithm
        rotation, translation = model_kabsch(
            from_points=lig_keypts, dst_points=rec_keypts,
            num_att_heads=self.num_att_heads, complex_names=complex_name, device=self.device)

        lig_evolved = (rotation @ Z_lig_coords.T).T + translation
        if self.align_method == 'p1-lig-p2':
            moved_lig2_keypts = (rotation @ lig2_keypts.t()).t() + translation
            rotation2, translation2 = model_kabsch(
                from_points=rec2_keypts, dst_points=moved_lig2_keypts,
                num_att_heads=self.num_att_heads, complex_names=complex_name, device=self.device)
        elif self.align_method == 'p1-p2':
            rotation2, translation2 = model_kabsch(
                from_points=rec2_keypts, dst_points=lig2_keypts,
                num_att_heads=self.num_att_heads, complex_names=complex_name, device=self.device
            )
        else:
            raise ValueError('Unknown align method: {}'.format(self.align_method))

        # ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss \
        #    ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2
        return lig_evolved, lig_keypts, rec_keypts, rotation, translation, geom_losses, \
                lig2_keypts, rec2_keypts, rotation2, translation2, valid


@MODELS.register_module()
class IEGMN_QueryDecoder_Pocket(IEGMN_QueryDecoder):
    def __init__(self, *args, **kwargs):
        known_pocket_info = kwargs.get('known_pocket_info', [])
        assert len(set(known_pocket_info)) == 6
        super().__init__(*args, **kwargs)
        del self.feat_type_embeddings
        del self.keypts_query_embeddings
        del self.kpt_transformer
        del self.keypts_cross_attn

        self.last_iegmn_layer = IEGMN_LigCoordLayer(
            orig_h_feats_dim=kwargs['residue_emb_dim'],
            h_feats_dim=kwargs['iegmn_lay_hid_dim'],
            out_feats_dim=kwargs['iegmn_lay_hid_dim'],
            lig_input_edge_feats_dim=kwargs['lig_input_edge_feats_dim'],
            rec_input_edge_feats_dim=kwargs['rec_input_edge_feats_dim'],
            cross_msgs=True,
            layer_norm=kwargs['layer_norm'],
            layer_norm_coords=kwargs['layer_norm_coords'],
            final_h_layer_norm=kwargs['final_h_layer_norm'],
            use_dist_in_layers=kwargs['use_dist_in_layers'],
            skip_weight_h=kwargs['skip_weight_h'],
            x_connection_init=kwargs['x_connection_init'],
            leakyrelu_neg_slope=kwargs['leakyrelu_neg_slope'],
            dropout=kwargs['dropout'],
            rec_evolve=False,
            fine_tune=True,
            nonlin=kwargs['nonlin'],
            debug=False,
            device=self.device,
            geometry_regularization=kwargs['geometry_regularization'],
            geometry_reg_step_size=kwargs['geometry_reg_step_size'],
        )

    def forward(self, lig_graph, rec_graph, rec2_graph, geometry_graph, complex_name, epoch, **kwargs):
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
            # p1lig_lig_pocket_coords = [self.normalize(i.to(lig_graph.device)) for i in kwargs['p1lig_lig_pocket_coords']]
            # lig_graph.ndata['new_x'] = batch_align_lig_to_pocket(
            #     old_batch_all_coords=lig_graph.ndata['new_x'],
            #     batch_pocket_masks=kwargs['p1lig_lig_pocket_mask'],
            #     batch_pocket_coords=p1lig_lig_pocket_coords
            # )

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
            p2lig_lig_pocket_coords = [self.normalize(i.to(lig_graph.device)) for i in kwargs['p2lig_lig_pocket_coords']]
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
            # if not self.training:
            if 'lig1_coord' in self.known_pocket_info:
                # p1lig_lig_pocket_coords = [self.normalize(i.to(lig_graph.device)) for i in kwargs['p1lig_lig_pocket_coords']]
                # coords_lig = batch_align_lig_to_pocket(
                #     old_batch_all_coords=coords_lig,
                #     batch_pocket_masks=kwargs['p1lig_lig_pocket_mask'],
                #     batch_pocket_coords=p1lig_lig_pocket_coords
                # )
                p1lig_lig_pocket_mask = torch.cat(kwargs['p1lig_lig_pocket_mask'])
                p1lig_lig_pocket_coords = torch.cat(kwargs['p1lig_lig_pocket_coords']).to(lig_graph.device)
                p1lig_lig_pocket_coords = self.normalize(p1lig_lig_pocket_coords)
                assert len(p1lig_lig_pocket_mask) == len(coords_lig)
                coords_lig[p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
            if 'lig2_coord' in self.known_pocket_info:
                p2lig_lig_pocket_coords = [self.normalize(i.to(lig_graph.device)) for i in kwargs['p2lig_lig_pocket_coords']]
                # align batch gt to input
                coords_lig = rigid_align_batch_pockets(
                    old_batch_all_coords=coords_lig,
                    batch_pocket_masks=kwargs['p2lig_lig_pocket_mask'],
                    batch_pocket_coords=p2lig_lig_pocket_coords
                )

        # last layer
        Z_lig_coords_batch, h_feats_lig, \
            coords_rec, h_feats_rec, \
            coords_rec2, h_feats_rec2, \
            trajectory, geom_loss = self.last_iegmn_layer(
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

        rotations = []
        translations = []
        recs_keypts = []
        ligs_keypts = []
        ligs_evolved = []
        ligs_evolved_last2 = []
        ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
        ligs_node_idx.insert(0, 0)
        recs_node_idx = torch.cumsum(rec_graph.batch_num_nodes(), dim=0).tolist()
        recs_node_idx.insert(0, 0)
        ### new for triplet
        valid_list = [] # in case checkpoint return NaN 
        rotations_2 = []    # rec2 -> lig
        translations_2 = []
        recs_keypts_2 = []
        ligs_keypts_2 = []
        recs_node_idx_2 = torch.cumsum(rec2_graph.batch_num_nodes(), dim=0).tolist()
        recs_node_idx_2.insert(0, 0)


        for idx in range(len(ligs_node_idx) - 1):
            lig_start = ligs_node_idx[idx]
            lig_end = ligs_node_idx[idx + 1]
            rec_start = recs_node_idx[idx]
            rec_end = recs_node_idx[idx + 1]
            rec2_start = recs_node_idx_2[idx]
            rec2_end = recs_node_idx_2[idx + 1]

            # unnormalize start
            Z_lig_coords = self.unnormalize(Z_lig_coords_batch[lig_start:lig_end])
            coord_lig = self.unnormalize(coords_lig[lig_start:lig_end])
            # unnormalize end
            ligs_evolved_last2.append(coord_lig)    # without rotate, only for kabsch loss

            lig_keypts, rec_keypts = None, None
            lig2_keypts, rec2_keypts = None, None

            # update Z_lig_coords
            # if not self.training:
            if 'lig1_coord' in self.known_pocket_info:
                p1lig_lig_pocket_mask = kwargs['p1lig_lig_pocket_mask'][idx]
                p1lig_lig_pocket_coords = kwargs['p1lig_lig_pocket_coords'][idx].to(lig_graph.device)
                # R, t = model_kabsch(from_points=Z_lig_coords[p1lig_lig_pocket_mask], dst_points=p1lig_lig_pocket_coords)
                # Z_lig_coords = rotate_and_translate(Z_lig_coords, R, t)
                Z_lig_coords[p1lig_lig_pocket_mask] = p1lig_lig_pocket_coords
            if 'lig2_coord' in self.known_pocket_info:
                p2lig_lig_pocket_mask = kwargs['p2lig_lig_pocket_mask'][idx]
                p2lig_lig_pocket_coords = kwargs['p2lig_lig_pocket_coords'][idx].to(lig_graph.device)
                R, t = model_kabsch(from_points=p2lig_lig_pocket_coords, dst_points=Z_lig_coords[p2lig_lig_pocket_mask])
                lig2_coord = rotate_and_translate(p2lig_lig_pocket_coords, R, t)
                Z_lig_coords[p2lig_lig_pocket_mask] = lig2_coord

            # if both protein and ligand masks are known, use coords as keypoints
            if 'p1_mask' in self.known_pocket_info and \
                'lig1_mask' in self.known_pocket_info:
                coord = kwargs['p1lig_p1_pocket_coords'][idx].to(lig_graph.device)
                rec_keypts = coord
                lig_keypts = Z_lig_coords[kwargs['p1lig_lig_pocket_mask'][idx]]
            if 'p1_mask' in self.known_pocket_info and \
                'lig1_coord' in self.known_pocket_info:    # overwrite the above if
                rec_keypts = kwargs['p1lig_p1_pocket_coords'][idx].to(lig_graph.device)
                lig_keypts = Z_lig_coords[kwargs['p1lig_lig_pocket_mask'][idx]]

            if 'p2_mask' in self.known_pocket_info and \
                'lig2_mask' in self.known_pocket_info:
                rec2_keypts = kwargs['p2lig_p2_pocket_coords'][idx].to(lig_graph.device)
                lig2_keypts = Z_lig_coords[kwargs['p2lig_lig_pocket_mask'][idx]]
            if 'p2_mask' in self.known_pocket_info and \
                'lig2_coord' in self.known_pocket_info: # overwrite
                rec2_keypts = kwargs['p2lig_p2_pocket_coords'][idx].to(lig_graph.device)
                # TODO: which one is better?
                # lig2_keypts = Z_lig_coords[kwargs['p2lig_lig_pocket_mask'][idx]]
                p2lig_lig_pocket_mask = kwargs['p2lig_lig_pocket_mask'][idx]
                p2lig_lig_pocket_coords = kwargs['p2lig_lig_pocket_coords'][idx].to(lig_graph.device)
                R, t = model_kabsch(from_points=p2lig_lig_pocket_coords, dst_points=Z_lig_coords[p2lig_lig_pocket_mask])
                lig2_keypts = rotate_and_translate(p2lig_lig_pocket_coords, R, t)

            valid = True

            ## Apply Kabsch algorithm
            rotation, translation = model_kabsch(
                from_points=lig_keypts, dst_points=rec_keypts,
                num_att_heads=self.num_att_heads, complex_names=complex_name, device=self.device)

            lig_evolved = (rotation @ Z_lig_coords.T).T + translation
            if self.align_method == 'p1-lig-p2':
                moved_lig2_keypts = (rotation @ lig2_keypts.t()).t() + translation
                rotation2, translation2 = model_kabsch(
                    from_points=rec2_keypts, dst_points=moved_lig2_keypts,
                    num_att_heads=self.num_att_heads, complex_names=complex_name, device=self.device)
            elif self.align_method == 'p1-p2':
                rotation2, translation2 = model_kabsch(
                    from_points=rec2_keypts, dst_points=lig2_keypts,
                    num_att_heads=self.num_att_heads, complex_names=complex_name, device=self.device
                )
            else:
                raise ValueError('Unknown align method: {}'.format(self.align_method))

            ligs_evolved.append(lig_evolved)
            ligs_keypts.append(lig_keypts)
            recs_keypts.append(rec_keypts)
            rotations.append(rotation)
            translations.append(translation)
            ligs_keypts_2.append(lig2_keypts)
            recs_keypts_2.append(rec2_keypts)
            rotations_2.append(rotation2)
            translations_2.append(translation2)
            valid_list.append(valid)

        # ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss \
        #    ligs_keypts_2, rec2_keypts_2, rotations_2, translations_2
        return ligs_evolved, ligs_keypts, recs_keypts, rotations, translations, geom_losses, \
                ligs_keypts_2, recs_keypts_2, rotations_2, translations_2, valid_list, ligs_evolved_last2


@MODELS.register_module()
class TripletDockQueryDecoder(TripletDock):
    """Use keypoint queries to generate keypoints"""

    def build_layers(self, device, debug, evolve_only, **kwargs):
        self.iegmn = IEGMN_QueryDecoder(device=device, debug=debug, evolve_only=evolve_only, **kwargs)


@MODELS.register_module()
class TripletDockQueryDecoderKnownPocket(TripletDock):
    """Use keypoint queries to generate keypoints"""

    def build_layers(self, device, debug, evolve_only, **kwargs):
        from .iegmn_predp2 import IEGMN_QueryDecoder_KnownPocket
        self.iegmn = IEGMN_QueryDecoder_KnownPocket(device=device, debug=debug, evolve_only=evolve_only, **kwargs)
