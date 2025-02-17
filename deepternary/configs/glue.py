from mmengine.config import read_base
from deepternary.models.metrics import TripletBindMetric
from deepternary.models.ternary_pdb import TernaryPreprocessedDataset
from deepternary.models.triplet_dock import TripletDock
from deepternary.models.custom_collate import dgl_collate
with read_base():
    from .equibind import *
    from .equibind import model, default_hooks

model.update(dict(
    # type='TripletDock',
    type=TripletDock,
    iegmn_class='IEGMN_QueryDecoder',
    align_method='p1-p2',
    residue_emb_dim=256,
    iegmn_lay_hid_dim=256,
    n_lays=8,
    noise_initial=2,
    num_att_heads=40,
    kpt_transformer_depth=4,
    kpt_transformer_heads=1,
))
model['loss_cfg'].update(dict(
    rec2_coord_weight=0.,
    p2_rmsd_pred_weight=1,
    ))


dataset_args = {
    ### triplet preprocess dataset args start
    'min_lig_atoms': 3,
    'min_pocket_atoms': 3,
    'preprocess_sub_path': 'pdb2311_merge',
    'freeze': 'protein1',
    'random_flip_proteins': True,
    ### triplet preprocess dataset args end
    "geometry_regularization_ring": True,
    "use_rdkit_coords": True,
    "bsp_proteins": False,
    "dataset_size": None,
    "translation_distance": 5.0,
    "n_jobs": 20,
    "chain_radius": 11,
    "rec_graph_radius": 30,
    "c_alpha_max_neighbors": 10,
    "lig_graph_radius": 5,
    "lig_max_neighbors": None,
    "pocket_cutoff": 6,     # pocket cutoff between alpha C of protein and the ligand
    'pocket_cutoff_p12': 10,    # pocket cutoff between alpha C of proteins
    "pocket_mode": "match_atoms_to_lig",
    "remove_h": True,
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
    batch_size=16,
    num_workers=2,
    dataset=dict(
        # type='PDBBind',
        # complex_names_path='data/timesplit_no_lig_overlap_train',
        type=TernaryPreprocessedDataset,
        # complex_names_path='data/DeepTernary/pdb1223_all_exclude22_noHs_eSimilar.txt',
        # complex_names_path='data/MolecularGlue/test_clusters.json',
        # complex_names_path='data/MolecularGlue/train_clusters_poc3.json',
        complex_names_path='data/MolecularGlue/train_clusters_poc3_exUNL.json',
        unrepresent_aug_rate=0.8,
        lig_predictions_name=None,
        is_train_data=True,
        **dataset_args,
        pipeline=[]),
    collate_fn=dict(type=dgl_collate),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        # type='PDBBind',
        # complex_names_path='data/timesplit_no_lig_overlap_val',
        type=TernaryPreprocessedDataset,
        complex_names_path='data/MolecularGlue/test_all.txt',
        # complex_names_path='data/DeepTernary/protac22.txt',
        # complex_names_path='data/MolecularGlue/group1_representative.txt',
        lig_predictions_name=None,
        is_train_data=False,
        **dataset_args,
        pipeline=[]),
    collate_fn=dict(type=dgl_collate),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
    drop_last=False)


test_dataloader = val_dataloader

val_evaluator = dict(
    type=TripletBindMetric,)
test_evaluator = val_evaluator

lr = 1e-4
optim_wrapper = dict(
    # optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-5),
    optimizer=dict(type='AdamW', lr=lr, weight_decay=1e-4),
    clip_grad=dict(max_norm=100),
    paramwise_cfg=dict(
        custom_keys={
            'feat_type_embeddings': dict(decay_mult=0.),
            'keypts_query_embeddings': dict(decay_mult=0.),
            })
)
max_epoch = 1000
param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=0.001,
    #     by_epoch=True,
    #     begin=0,
    #     end=5,
    #     # update by iter
    #     convert_to_iter_based=True),
    # main learning rate scheduler
    # dict(
    #     type='CosineAnnealingLR',
    #     T_max=max_epoch-5,
    #     eta_min=lr * 0.1,
    #     by_epoch=True,
    #     begin=5,
    #     end=max_epoch),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[800],
        gamma=0.1)
]

train_cfg = dict(by_epoch=True, max_epochs=max_epoch, val_interval=50)

default_hooks.update(dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=200, max_keep_ckpts=3),
))


# load_from = 'output/EquiBindOrigin_ternary_MG/epoch_1000.pth'
resume = True
