import requests

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def get_chembl_info(compound_input):
    """
    Given a compound name or SMILES, fetch information from ChEMBL.
    Returns a dictionary with compound info or None if not found.
    """

    search_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={compound_input}"
    response = requests.get(search_url, headers={"Accept": "application/json"})
    
    if response.status_code != 200:
        print("Error connecting to ChEMBL API")
        return None

    search_results = response.json()
    if not search_results['molecules']:
        print("Compound not found in ChEMBL")
        return None

    #taking the first molecule found - could do something better? 
    chembl_id = search_results['molecules'][0]['molecule_chembl_id']

    #get info
    molecule_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}"
    molecule_response = requests.get(molecule_url, headers={"Accept": "application/json"})
    
    if molecule_response.status_code != 200:
        print("Error fetching molecule details")
        return None

    molecule_data = molecule_response.json()

    info = {
        "ChEMBL ID": chembl_id,
        "Name": molecule_data.get("pref_name"),
        "Molecule Type": molecule_data.get("molecule_type"),
        "SMILES": molecule_data.get("molecule_structures", {}).get("canonical_smiles"),
        "Max Phase": molecule_data.get("max_phase"),  # clinical trial phase
        "Mechanism of Action": molecule_data.get("mechanism_of_action"),
        "Therapeutic Indications": molecule_data.get("therapeutic_indication"),
    }

    return info





def get_smiles(molecule_name):
    
    molecule_name = molecule_name.strip()
    
    #try pubchem - pug rest api 
    pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{molecule_name}/property/SMILES,CanonicalSMILES,IsomericSMILES,ConnectivitySMILES/JSON"
    response = requests.get(pubchem_url)
    
    if response.status_code == 200:
        try:
            props = response.json()['PropertyTable']['Properties'][0]
            for field in ["CanonicalSMILES", "IsomericSMILES", "ConnectivitySMILES", "SMILES"]:
                if field in props:
                    return props[field]
        except (KeyError, IndexError):
            pass  
    
    # try chembl
    chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={molecule_name}"
    response = requests.get(chembl_url)
    
    if response.status_code == 200:
        try:
            molecules = response.json().get("molecules", [])
            if molecules:
                # Get ChEMBL ID of the first match
                chembl_id = molecules[0]["molecule_chembl_id"]
                
                # Fetch details for that molecule
                details_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
                details = requests.get(details_url).json()
                
                # Extract SMILES (canonical preferred)
                smiles = details.get("molecule_structures", {}).get("canonical_smiles")
                if smiles:
                    print("used chembl")
                    return smiles
        except Exception:
            pass
    
    #if nothing found 
    raise ValueError(f"Couldn't find information on NCBI PubChem or ChEMBL for '{molecule_name}'.")


def get_name_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    name = rdMolDescriptors.CalcMolFormula(mol)
    
    return name

# # Example usage
# compound_name = "Aspirin"
# compound_info = get_chembl_info(compound_name)

# if compound_info:
#     for key, value in compound_info.items():
#         print(f"{key}: {value}")
