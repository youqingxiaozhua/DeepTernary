from typing import List
import numpy as np
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser, PDBIO


def read_pdb(pdb_path):
    return PandasPdb().read_pdb(pdb_path)


def get_pdb_coords(pdb, is_ligand) -> np.ndarray:
    """Get the coordinates of the atoms in a pdb file"""
    if isinstance(pdb, str):
        pdb = read_pdb(pdb)
    if not isinstance(pdb, PandasPdb):
        raise ValueError(
            "pdb must be a path to a pdb file or a PandasPdb object")
    category = "HETATM" if is_ligand else "ATOM"
    coords = pdb.df[category][["x_coord", "y_coord", "z_coord"]].to_numpy()
    return coords


def set_new_coords(pdb, coords, is_ligand):
    if isinstance(pdb, str):
        pdb = read_pdb(pdb)
    if not isinstance(pdb, PandasPdb):
        raise ValueError(
            "pdb must be a path to a pdb file or a PandasPdb object")
    original_coords = get_pdb_coords(pdb, is_ligand)
    assert original_coords.shape == coords.shape, f"Original coords shape: {original_coords.shape}, new coords shape: {coords.shape}"
    category = "HETATM" if is_ligand else "ATOM"
    pdb.df[category][["x_coord", "y_coord", "z_coord"]] = coords
    return pdb


def set_new_chain(pdb, chain_id, is_ligand):
    if isinstance(pdb, str):
        pdb = read_pdb(pdb)
    if not isinstance(pdb, PandasPdb):
        raise ValueError(
            "pdb must be a path to a pdb file or a PandasPdb object")
    category = "HETATM" if is_ligand else "ATOM"
    assert len(set(pdb.df[category]["chain_id"])) == 1, f"Only one chain is allowed, but got {set(pdb.df[category]['chain_id'])}"
    pdb.df[category]["chain_id"] = chain_id
    return pdb


def write_pdb(pdb, path):
    if not isinstance(pdb, PandasPdb):
        raise ValueError("pdb must be a PandasPdb object")
    pdb.to_pdb(path)


def merge_pdbs(pdb_files: List[str], path: str):
    complex_text = []
    for fname in pdb_files:
        with open(fname, 'r') as f:
            complex_text.extend(f.read().strip().split('\n'))
    complex_text = [i for i in complex_text if i[:4] in ('ATOM', 'HETA', 'CONE')]
    with open(path, 'w') as f:
        f.write('\n'.join(complex_text))


def combine_multiple_pdb(pdb_files: List[str], path: str):
    pdbs = [PDBParser().get_structure('random_id', p) for p in pdb_files]
    for structure in pdbs:
        model = structure[0]
        assert len(model) == 1, 'Should only have one chain'

    p1_structure = pdbs[0]
    p1_model = p1_structure[0]
    for structure in pdbs[1:]:
        model = structure[0]
        for chain in model:
            p1_model.add(chain)

    io = PDBIO()
    io.set_structure(p1_structure)
    io.save(path)
