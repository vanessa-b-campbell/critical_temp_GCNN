from rdkit import Chem
from rdkit.Chem import Draw
mol = Chem.MolFromSmiles('CC[Si](CC)(CC)O[Si](CC)(CC)CC')


mol = Chem.MolFromSmiles(smile)
img = Draw.MolToImage(mol)