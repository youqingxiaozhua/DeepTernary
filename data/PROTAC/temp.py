import os
import shutil
import json

from tqdm import tqdm
import torch

from Bio.PDB import PDBParser, PDBIO

from projects.equibind.models.path import PDB_PATH, PREPROCESSED_PATH
from projects.equibind.models.utils import read_strings_from_txt
from projects.equibind.models.process_mols import get_pocket_and_mask

SAVE_PATH = 'output/new_test'
os.makedirs(SAVE_PATH, exist_ok=True)


def convert_old_test(old_name_str, new_name_str):
    """
    convert old test data to new folder
    old: 6SIS_DA_D_LFE
    """
    old_name = (old_name_str[:4], old_name_str[5], old_name_str[6], old_name_str[-3:])
    new_name = new_name_str.split('_')
    assert len(old_name) == len(new_name)
    assert old_name[0] == new_name[0]
    assert set(old_name[1:3]) == set(new_name[1:3])
    assert old_name[3] == new_name[3]

    OLD_PATH = '/opt/data/DeepTernary/pdb1223/'

    if old_name[1] == new_name[1]:
        # same order, directly copy
        assert old_name[2] == new_name[2]
        shutil.copytree(
            os.path.join(OLD_PATH, old_name_str),
            os.path.join(SAVE_PATH, new_name_str)
        )
    else:
        # need to exchange p1 and p2
        os.makedirs(os.path.join(SAVE_PATH, new_name_str), exist_ok=True)
        same_names = ('complex.pdb', 'ligand.pdb')
        for same_name in same_names:
            shutil.copy(
                os.path.join(OLD_PATH, old_name_str, same_name),
                os.path.join(SAVE_PATH, new_name_str, same_name))
        shutil.copy(
            os.path.join(OLD_PATH, old_name_str, 'protein2.pdb '),
            os.path.join(SAVE_PATH, new_name_str, 'protein1.pdb'))
        shutil.copy(
            os.path.join(OLD_PATH, old_name_str, 'protein1.pdb'),
            os.path.join(SAVE_PATH, new_name_str, 'protein2.pdb'))


def remove_old_folder(old_name_str, new_name_str):
    old_name = (old_name_str[:4], old_name_str[5], old_name_str[6], old_name_str[-3:])
    new_name = new_name_str.split('_')
    new_name_str2 = f'{new_name[0]}_{new_name[2]}_{new_name[1]}_{new_name[3]}'
    for name in (new_name_str, new_name_str2):
        if os.path.exists(os.path.join(PDB_PATH, name)):
            # print()
            shutil.rmtree(os.path.join(PDB_PATH, name))


def update_all_list():
    """Remove old names in complex.txt"""
    new_names = read_strings_from_txt('data/PROTAC/protac22.txt')
    all_names = read_strings_from_txt('data/2311/complex.txt')
    n = len(all_names)
    all_names = set(all_names)
    print(len(all_names))
    assert len(all_names) == n
    for new_name_str in new_names:
        new_name = new_name_str.split('_')
        delete_name_str = f'{new_name[0]}_{new_name[2]}_{new_name[1]}_{new_name[3]}'
        if delete_name_str in all_names:
            all_names.remove(delete_name_str)
    print('After deleting: ', len(all_names))
    all_names = all_names.union(set(new_names))
    print('After adding: ', len(all_names))

    with open('data/PROTAC/complex.txt', 'w') as f:
        f.write('\n'.join(all_names))


def filter_few_pocket(min_poc_num=3):
    """delete complex if p1-lig-p2 pocket number < 3"""
    def check_pocket(name):
        data = torch.load(os.path.join(PREPROCESSED_PATH, 'pdb2311_noHs', f'{name}.pth'))
        lig_graph = data['lig_graphs'][0]
        p1_graph = data['p1_graph']
        p2_graph = data['p2_graph']
        p1_coord, _, _ = get_pocket_and_mask(lig_graph.ndata['x'], p1_graph.ndata['x'], cutoff=6)
        p2_coord, _, _ = get_pocket_and_mask(lig_graph.ndata['x'], p2_graph.ndata['x'], cutoff=6)
        if len(p1_coord) >= min_poc_num and len(p2_coord) >= min_poc_num:
            return True
        return False

    with open('data/PROTAC/train_clusters.json', 'r') as f:
        clusters = json.load(f)
    new_clusters = []
    for cluster in tqdm(clusters):
        represent = cluster['representative']
        items = cluster['items']
        represent = [i for i in represent if check_pocket(i)]
        items = [i for i in items if check_pocket(i)]
        if len(items):
            new_clusters.append({
                'representative': represent,
                'items': items
                })
    with open(f'data/PROTAC/train_clusters_poc{min_poc_num}.json', 'w') as f:
        json.dump(new_clusters, f, indent=2)
    
    print()


def check_chain_num():
    """check if chain number is 2 in gt_complex"""
    names = read_strings_from_txt('data/PROTAC/protac22.txt')
    for name in names:
        pdb_id, chain1, chain2, lig = name.split('_')
        pdb = PDBParser().get_structure('', os.path.join(PDB_PATH, name, 'gt_complex.pdb'))
        model = pdb[0]
        assert model.child_dict.keys() == {'A', 'B'}
        # save chain A as protein1.pdb
        io = PDBIO()
        io.set_structure(model['A'])
        io.save(os.path.join(PDB_PATH, name, 'protein1.pdb'))
        # save chain B as protein2.pdb
        io = PDBIO()
        io.set_structure(model['B'])
        io.save(os.path.join(PDB_PATH, name, 'protein2.pdb'))


def fake_test_cluster():
    with open('data/PROTAC/protac22.txt', 'r') as f:
        names = f.read().strip().split('\n')
    
    fake_test_cluster = []
    for name in names:
        d = {
            'representative': [name],
            'items': [name]
        }
        fake_test_cluster.append(d)
    
    with open('data/PROTAC/fake_test_cluster.json', 'w') as f:
        json.dump(fake_test_cluster, f, indent=2)

if __name__ == '__main__':
    fake_test_cluster()
    # filter_few_pocket(6)
    exit()
    check_chain_num()
    exit()
    old_names = read_strings_from_txt('data/DeepTernary/protac22.txt')
    new_names = read_strings_from_txt('data/PROTAC/protac22.txt')
    for old, new in zip(old_names, new_names):
        # convert_old_test(old, new)
        remove_old_folder(old, new)
