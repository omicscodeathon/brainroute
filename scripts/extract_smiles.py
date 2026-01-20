import pandas as pd


# Read the CSV file
df = pd.read_csv('data/combined_bbb_classification_fixed.csv')

# Extract the 'smiles' column
smiles_list = df['smiles'].tolist()

# Write the SMILES strings to a text file, one per line and save in data folder
with open('data/clean_smiles.smi', 'w') as f:
    for smiles in smiles_list:
        f.write(f"{smiles}\n")
    print("SMILES strings have been written to data/smiles.smi")