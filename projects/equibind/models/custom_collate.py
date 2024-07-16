import random
from typing import Tuple, List, Sequence

import dgl
import torch
from mmengine.registry import FUNCTIONS

from .geometry_utils import random_rotation_translation
from .process_mols import lig_rec_graphs_to_complex_graph


def graph_collate(batch):
    complex_graphs, ligs_coords, recs_coords, pockets_coords_lig, pockets_coords_rec,geometry_graph, complex_names, idx = map(list, zip(*batch))
    geometry_graph = dgl.batch(geometry_graph) if geometry_graph[0] != None else None
    return dgl.batch(complex_graphs), ligs_coords, recs_coords, pockets_coords_lig, pockets_coords_rec,geometry_graph, complex_names, idx


@FUNCTIONS.register_module()
def graph_collate_revised(batch):
    lig_graphs, rec_graphs, ligs_coords, recs_coords,all_rec_coords, pockets_coords_lig,geometry_graph, complex_names, idx = map(list, zip(*batch))
    geometry_graph = dgl.batch(geometry_graph) if geometry_graph[0] != None else None
    return dgl.batch(lig_graphs), dgl.batch(rec_graphs), ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig,geometry_graph, complex_names, idx



@FUNCTIONS.register_module()
def dgl_collate(data_batch: List[dict]) -> dict:
    """
    Args:
        data_batch (Sequence): Data sampled from dataset.

    Returns:
        Any: Data in the same format as the data_itement of ``data_batch``, of which
        tensors have been stacked, and ndarray, int, float have been
        converted to tensors.
    """
    batch_data = dict()
    for i in range(len(data_batch)):
        for k, v in data_batch[i].items():
            if k in batch_data:
                batch_data[k].append(v)
            else:
                batch_data[k] = [v]

    # batchlize graphs
    data = dict()
    for k, v in batch_data.items():
        if k == 'geometry_graph':
            if v[0] is not None:
                data[k] = dgl.batch(v)
            else:
                data[k] = None
        elif k.endswith('_graph'):
            data[k] = dgl.batch(v)
        else:
            data[k] = v
    return data



def torsion_collate(batch):
    lig_graphs, rec_graphs, angles, masks, ligs_coords, recs_coords,all_rec_coords, pockets_coords_lig,geometry_graph, complex_names, idx = map(list, zip(*batch))
    geometry_graph = torch.cat(geometry_graph,dim=0) if geometry_graph[0] != None else None
    return dgl.batch(lig_graphs), dgl.batch(rec_graphs), torch.cat(angles,dim=0), torch.cat(masks, dim=0),  ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig,geometry_graph, complex_names, idx


class AtomSubgraphCollate(object):
    def __init__(self, random_rec_atom_subgraph_radius=10):
        self.random_rec_atom_subgraph_radius = random_rec_atom_subgraph_radius
    def __call__(self, batch: List[Tuple]):
        lig_graphs, rec_graphs, ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig,geometry_graph, complex_names, idx = map(
            list, zip(*batch))

        rec_subgraphs =  []
        for i, (lig_graph, rec_graph) in enumerate(zip(lig_graphs,rec_graphs)):
            rot_T, rot_b = random_rotation_translation(translation_distance=2)
            translated_lig_coords = ligs_coords[i] + rot_b
            min_distances, _ = torch.cdist(rec_graph.ndata['x'], translated_lig_coords.to(rec_graph.ndata['x'].device)).min(dim=1)
            rec_subgraphs.append(dgl.node_subgraph(rec_graph, min_distances < self.random_rec_atom_subgraph_radius))

        geometry_graph = dgl.batch(geometry_graph) if geometry_graph[0] != None else None

        return dgl.batch(lig_graphs), dgl.batch(rec_subgraphs), ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig,geometry_graph, complex_names, idx

class SubgraphAugmentationCollate(object):
    def __init__(self, min_shell_thickness=2):
        self.min_shell_thickness = min_shell_thickness
    def __call__(self, batch: List[Tuple]):
        lig_graphs, rec_graphs, ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig,geometry_graph, complex_names, idx = map(
            list, zip(*batch))

        rec_subgraphs =  []
        for lig_graph, rec_graph in zip(lig_graphs,rec_graphs):
            lig_centroid = lig_graph.ndata['x'].mean(dim=0)
            distances = torch.norm(rec_graph.ndata['x'] - lig_centroid, dim=1)
            max_distance = torch.max(distances)
            min_distance = torch.min(distances)
            radius = min_distance + self.min_shell_thickness + random.random() * (max_distance - min_distance- self.min_shell_thickness).abs()
            rec_subgraphs.append(dgl.node_subgraph(rec_graph, distances <= radius))
        geometry_graph = dgl.batch(geometry_graph) if geometry_graph[0] != None else None

        return dgl.batch(lig_graphs), dgl.batch(rec_subgraphs), ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig,geometry_graph, complex_names, idx