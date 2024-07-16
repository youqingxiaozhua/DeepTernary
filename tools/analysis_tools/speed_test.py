from copy import deepcopy
from tqdm import tqdm
import torch
from mmpretrain import get_model
from projects.equibind.models.path import PREPROCESSED_PATH
from projects.equibind.models.process_mols import get_pocket_and_mask

config = 'output/PP_No4_poc_alpoc_poc_poc_poc_poc/protac.py'
ckpt = 'output/PP_No4_poc_alpoc_poc_poc_poc_poc/epoch_1000.pth'

model = get_model(config, pretrained=ckpt, device='cuda')
model.eval()

preprocessed_data = torch.load(f'{PREPROCESSED_PATH}/pdb2311_noHs/5T35_D_A_759.pth')

name = preprocessed_data['name']
lig_graph = preprocessed_data['lig_graphs'][0]
p1_graph = preprocessed_data['p1_graph']
p2_graph = preprocessed_data['p2_graph']
geometry_graph = preprocessed_data['geometry_graphs'][0]

protein1_pocket_coords = preprocessed_data['p1_pocket']
protein2_pocket_coords = preprocessed_data['p2_pocket']

p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = get_pocket_and_mask(
    lig_graph.ndata['x'], p1_graph.ndata['x'], cutoff=12)
p1lig_p1_pocket_coords = deepcopy(p1lig_lig_pocket_coords)
p2lig_lig_pocket_coords, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = get_pocket_and_mask(
    lig_graph.ndata['x'], p2_graph.ndata['x'], cutoff=12)

data = dict(
    lig_graph=lig_graph,
    rec_graph=p1_graph,
    rec2_graph=p2_graph,
    geometry_graph=geometry_graph,
    complex_name=[name],
    rec2_coords=[p2_graph.ndata['x']],
    rec2_coords_input=[p2_graph.ndata['x']],
    p1lig_p1_pocket_mask=[p1lig_p1_pocket_mask],
    p1lig_p1_pocket_coords=[p1lig_lig_pocket_coords],
    p1lig_lig_pocket_mask=[p1lig_lig_pocket_mask],
    p1lig_lig_pocket_coords=[p1lig_lig_pocket_coords],
    p2lig_p2_pocket_mask=[p2lig_p2_pocket_mask],
    p2lig_p2_pocket_coords=[p2lig_lig_pocket_coords],
    p2lig_lig_pocket_mask=[p2lig_lig_pocket_mask],
    p2lig_lig_pocket_coords=[p2lig_lig_pocket_coords],
)

# warmup
for _ in tqdm(range(50)):
    with torch.no_grad():
        model(**data, mode='predict')

# inference
N = 20
import time

start = time.time()
for _ in tqdm(range(N * 40)):  # every sample need 40 seeds
    with torch.no_grad():
        model(**data, mode='predict')
end = time.time()

print(f'Inference time: {(end - start) / N:.4f} s')
