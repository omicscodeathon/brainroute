import os
import psycopg2
from dotenv import load_dotenv
import threading
import numpy as np
from rdkit.Chem import Descriptors

load_dotenv()

def convert_numpy_types(value):
  """Convert NumPy types to Python native types for database insertion."""
  if value is None:
    return None
  if isinstance(value, (np.integer, np.floating)):
    return value.item()  # Converts np.int64, np.float64, etc. to Python native types
  if isinstance(value, (list, tuple)):
    return [convert_numpy_types(v) for v in value]
  if isinstance(value, dict):
    return {k: convert_numpy_types(v) for k, v in value.items()}
  return value

def add_to_database(results):
  """Adds a compound and its prediction results to the Neon database.
  """
  try:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    
    prop = results.get('properties', {})
    info = results.get('info', {})
        
    # Extract data and convert NumPy types
    smiles = info.get('SMILES', results.get('smiles'))
    name = info.get('Name', results.get('name'))
    formula = results.get('formula')
    prediction = 1 if results.get('prediction')=='BBB+' else 0
    confidence = convert_numpy_types(results.get('confidence'))
    mw = convert_numpy_types(prop.get('mw'))
    hbd = convert_numpy_types(prop.get('hbd'))
    hba = convert_numpy_types(prop.get('hba'))
    tpsa = convert_numpy_types(prop.get('tpsa'))
    rotatable_bonds = convert_numpy_types(prop.get('rotatable_bonds'))
    heavy_atoms = convert_numpy_types(prop.get('heavy_atoms'))

    insert_query = """
    INSERT INTO molecules_and_predictions 
    ("Smiles", "Name", "Prediction", "Confidence", mw, hbd, hba, tpsa, rotatable_bonds, heavy_atoms, formula)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT ("Smiles") DO NOTHING
    """
    
    cur.execute(insert_query, (smiles, name, prediction, confidence, mw, hbd, hba, tpsa, rotatable_bonds, heavy_atoms, formula))
    
    conn.commit()
    cur.close()
    conn.close()
    print(f'added {name} to database')

  except Exception as e:
    print(f"Error adding to Neon DB: {e}")
    

def add_to_database_batch(results):
  """Adds a batch of compounds and prediction results to the Neon database.
  """
  try:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    
    insert_query = """
    INSERT INTO molecules_and_predictions 
    ("Smiles", "Name", "Prediction", "Confidence", mw, hbd, hba, tpsa, rotatable_bonds, heavy_atoms, formula)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT ("Smiles") DO NOTHING
    """
    for r in results:
      # neatly geting the dictionary items in order to avoid error in indexing
      prop = r.get('properties', {})
      info = r.get('info', {})
    
      # Prefer API-provided SMILES/Name; fall back to computed results
      smiles = convert_numpy_types(info.get('SMILES', r.get('smiles', 'Unknown')))
      name = convert_numpy_types(info.get('Name', r.get('name', smiles)))
      prediction = 1 if r.get('prediction')=='BBB+' else 0
      confidence = convert_numpy_types(r.get('confidence'))
      formula = convert_numpy_types(r.get('formula', None))
      mw = convert_numpy_types(prop.get('mw'))
      hbd = convert_numpy_types(prop.get('hbd'))
      hba = convert_numpy_types(prop.get('hba'))
      tpsa = convert_numpy_types(prop.get('tpsa'))
      rotatable_bonds = convert_numpy_types(prop.get('rotatable_bonds'))
      heavy_atoms = convert_numpy_types(prop.get('heavy_atoms'))
      
        
      cur.execute(insert_query, (smiles, name, prediction, confidence, mw, hbd, hba, tpsa, rotatable_bonds, heavy_atoms, formula))
      print(f'added {name} to database')
    cur.close()
    conn.close()
    print(f"Batch data added to DB successfully .")
  except Exception as e:
    print(f"Error adding batch to Neon DB: {e}")

# using a threaded function to enable asynchronous writing to database
def add_to_database_threaded(results):
    """
    Wrapper to run the database insertion in a background thread.
    """
    db_thread = threading.Thread(target=add_to_database, args=(results,))
    db_thread.start()
    
def add_to_database_batch_threaded(results):
    """
    Wrapper to run the database insertion in a background thread.
    """
    db_thread = threading.Thread(target=add_to_database_batch, args=(results,))
    db_thread.start()