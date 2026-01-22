import os
import psycopg2
from dotenv import load_dotenv
import threading
from rdkit.Chem import Descriptors

load_dotenv()

def add_to_database(results):
  """Adds a compound and its prediction results to the Neon database.
  """
  try:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    
    prop = results.get('properties')
    info = results.get('info')
    
    # Extract data
    smiles =info.get('SMILES', results.get('smiles') )
    name = info.get('Name', results.get('name'))
    formula = results['formula']
    prediction = 1 if results.get('prediction')=='BBB+' else 0
    confidence = results.get('confidence')
    mw = prop.get('mw')
    hbd = prop.get('hbd')
    hba = prop.get('hba')
    tpsa = prop.get('tpsa')
    rotatable_bonds = prop.get('rotatable_bonds')
    heavy_atoms = prop.get('heavy_atoms')
    formula = results.get('formula',  None)

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
    
      smiles = info.get('SMILES', 'Unknown')
      name = info.get('Name', 'Unknown')
      prediction = 1 if r.get('prediction')=='BBB+' else 0
      confidence =  r.get('confidence')
      formula = r.get('formula', None)
      mw = prop.get('mw')
      hbd = prop.get('hbd')
      hba = prop.get('hba')
      tpsa = prop.get('tpsa')
      rotatable_bonds = prop.get('rotatable_bonds')
      heavy_atoms = prop.get('heavy_atoms')
      
        
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