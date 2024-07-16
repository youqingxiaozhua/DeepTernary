
from projects.equibind.models import EquiBind, PDBBind, EquiBindMetric


# custom_imports = dict(
#     imports=['projects.equibind.models'], allow_failed_imports=False)

model = {
  "type": EquiBind,
  "lig_input_edge_feats_dim": 15,
  "rec_input_edge_feats_dim": 27,
  "geometry_reg_step_size": 0.001,
  "geometry_regularization": True,
  "use_evolved_lig": True, # Whether or not to use the evolved lig as final prediction
  "standard_norm_order": True,
  "unnormalized_kpt_weights": False, # no softmax for the weights that create the keypoints
  "lig_evolve": True, # whether or not the coordinates are changed in the EGNN layers
  "rec_evolve": True,
  "rec_no_softmax": False,
  "lig_no_softmax": False,
  "centroid_keypts_construction_rec": False,
  "centroid_keypts_construction_lig": False,
  "centroid_keypts_construction": False, # this is old. use the two above
  "move_keypts_back": True, # move the constructed keypoints back to the location of the ligand
  "normalize_Z_rec_directions": False,
  "normalize_Z_lig_directions": False,
  "n_lays": 8, # 5 in good run
  "debug": False,
  "use_rec_atoms": False,
  "shared_layers": False, # False in good run
  "noise_decay_rate": 0.5,
  "noise_initial": 1,
  "use_edge_features_in_gmn": True,
  "use_mean_node_features": True,
  "residue_emb_dim": 64,
  "iegmn_lay_hid_dim": 64,
  "num_att_heads": 30, # 20 in good run
  "dropout": 0.1,
  "nonlin": 'lkyrelu', # ['swish', 'lkyrelu']
  "leakyrelu_neg_slope": 1.0e-2, # 1.0e-2 in good run
  "cross_msgs": True,
  "layer_norm": 'BN', # ['0', 'BN', 'LN'] # BN in good run
  "layer_norm_coords": '0', # ['0', 'LN'] # 0 in good run
  "final_h_layer_norm": '0', # ['0', 'GN', 'BN', 'LN'] # 0 in good run
  "pre_crossmsg_norm_type": '0', # ['0', 'GN', 'BN', 'LN']
  "post_crossmsg_norm_type": '0', # ['0', 'GN', 'BN', 'LN']
  "use_dist_in_layers": True,
  "skip_weight_h": 0.5, # 0.5 in good run
  "x_connection_init": 0.25, # 0.25 in good run
  "random_vec_dim": 0, # set to zero to have no stochasticity
  "random_vec_std":1,
   "use_scalar_features": False, # Have a look at lig_feature_dims in process_mols.py to see what features we are talking about.
   "num_lig_feats": None, # leave as None to use all ligand features. Have a look at lig_feature_dims in process_mols.py to see what features we are talking about. If this is one, only the first of those will be used.
   "normalize_coordinate_update": True,
   "rec_square_distance_scale": 10, # divide square distance by ten to have a nicer separation instead of many zero point zero zero zero zero zero
   "loss_cfg": {
        'ot_loss_weight': 1,
        'key_point_alignmen_loss_weight': 0,  # this does only work if ot_loss_weight is not 0
        'centroid_loss_weight': 0,
        'intersection_loss_weight': 1,
        'intersection_sigma': 8,  # 8 was determined by gridsearch over data, 25 for EquiDock
        'intersection_surface_ct': 1,  # grid search says 2.5,  10 for EquiDock
        'translated_lig_kpt_ot_loss': False,
        'kabsch_rmsd_weight': 1
    }
}


dataset_args = {
    ### triplet preprocess dataset args start
    # 'min_lig_atoms': 10,
    # 'min_pocket_atoms': 9,
    # 'preprocess_sub_path': 'noHs_MG',
    # 'freeze': 'protein1',
    ### triplet preprocess dataset args end
    "geometry_regularization_ring": True,
    "use_rdkit_coords": True,
    "bsp_proteins": False,
    "dataset_size": None,
    "translation_distance": 5.0,
    "n_jobs": 20,
    "chain_radius": 10,
    "rec_graph_radius": 30,
    "c_alpha_max_neighbors": 10,
    "lig_graph_radius": 5,
    "lig_max_neighbors": None,
    "pocket_cutoff": 4,
    "pocket_mode": "match_atoms_to_lig",
    "remove_h": False,
    "only_polar_hydrogens": False,
    "use_rec_atoms": False,
    "surface_max_neighbors": 5,
    "surface_graph_cutoff": 5,
    "surface_mesh_cutoff": 2,
    "subgraph_augmentation": False,
    "min_shell_thickness": 3,
    "rec_subgraph": False,
    "subgraph_radius": 10,
    "subgraph_max_neigbor": 8,
    "subgraph_cutoff": 4}

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type=PDBBind,
        complex_names_path='data/timesplit_no_lig_overlap_train',
        # type='TernaryPreprocessedDataset',
        # complex_names_path='data/DeepTernary/pdb1223_all_exclude22_noHs_eSimilar.txt',
        lig_predictions_name=None,
        is_train_data=True,
        **dataset_args,
        pipeline=[]),
    collate_fn=dict(type='dgl_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=PDBBind,
        complex_names_path='data/timesplit_no_lig_overlap_val',
        # type='TernaryPreprocessedDataset',
        # complex_names_path='data/DeepTernary/protac22.txt',
        lig_predictions_name=None,
        is_train_data=False,
        **dataset_args,
        pipeline=[]),
    collate_fn=dict(type='dgl_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False))


test_dataloader = val_dataloader

val_evaluator = dict(
    type=EquiBindMetric,)
test_evaluator = val_evaluator

lr = 1e-4
optim_wrapper = dict(
    # optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-5),
    optimizer=dict(type='AdamW', lr=lr, weight_decay=1e-4),
    clip_grad=dict(max_norm=100),
    paramwise_cfg=dict(
        custom_keys={
            'graph_feat_pock_gt_embed': dict(decay_mult=0.),
            'semantic_embedding': dict(decay_mult=0.),
            'feat_type_embeddings': dict(decay_mult=0.),
            'gt_type_embeddings': dict(decay_mult=0.),

            })
)
max_epoch = 1000
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch-5,
        eta_min=lr * 0.1,
        by_epoch=True,
        begin=5,
        end=max_epoch),
    # dict(
    #     type='MultiStepLR',
    #     begin=10,
    #     end=max_epoch,
    #     by_epoch=True,
    #     milestones=[80],
    #     gamma=0.1)
]

train_cfg = dict(by_epoch=True, max_epochs=max_epoch, val_interval=10)
val_cfg = dict()
test_cfg = dict()
# auto_scale_lr = dict(base_batch_size=32*8)

default_scope = 'mmpretrain'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=3, save_best='bind/loss'),
    checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=0, deterministic=False)
custom_hooks = [dict(type='InsepectHook')]
