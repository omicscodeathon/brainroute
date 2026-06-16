
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

# Read SMILES from data/smiles.txt
with open('data/smiles.txt', 'r') as f:
    smiles_list = [line.strip() for line in f.readlines()]

# Get all descriptor names and functions
descriptor_names = [desc[0] for desc in Descriptors._descList]
descriptor_funcs = [desc[1] for desc in Descriptors._descList]

# Calculate descriptors for each molecule
data = []
for smile in smiles_list:
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        values = [func(mol) for func in descriptor_funcs]
    else:
        values = [None] * len(descriptor_funcs)
    data.append(values)

# Convert to dataframe
df = pd.DataFrame(data, columns=descriptor_names)
df['smiles'] = smiles_list
df.to_csv('data/rdkit_descriptors.csv', index=False)