from rdkit import Chem
from rdkit.Chem import AllChem



def gen_btk_crbn_protacs():
    with open('data/Affinity/protacs.txt', 'r') as f:
        data = f.read().strip().split('\n')
    for line in data:
        idx, smiles = line.split(',')
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        # remove Hs
        mol = Chem.RemoveHs(mol)
        Chem.MolToPDBFile(mol, f'data/Affinity/protacs/{idx}.pdb')


if __name__ == '__main__':
    gen_btk_crbn_protacs()
