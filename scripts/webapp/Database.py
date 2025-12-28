"""
Google Sheets-based database for BBB predictions.
This persists data in the cloud and works on Streamlit Cloud!

Setup:
1. pip install gspread oauth2client
2. Create a Google Cloud project: https://console.cloud.google.com
3. Enable Google Sheets API
4. Create a service account and download JSON credentials
5. Share your Google Sheet with the service account email
"""
#should be using google_auth? oauth2client no longer supported


import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from datetime import datetime
import hashlib
from typing import Optional, Dict, List
import streamlit as st

class GoogleSheetsDatabase:
    """
    Database manager using Google Sheets as the backend.
    Free, persistent, and works on any platform.
    """
    
    def __init__(self, credentials_dict: Dict = None, sheet_name: str = "BBB_Predictions"):
        """
        Initialize connection to Google Sheets.
        
        Args:
            credentials_dict: Service account credentials (from Streamlit secrets)
            sheet_name: Name of the Google Sheet to use
        """
        self.sheet_name = sheet_name
        self.worksheet = None
        self._connect(credentials_dict)
        self._ensure_headers()
    
    def _connect(self, credentials_dict: Dict = None):
        """Establish connection to Google Sheets."""
        try:
            # Use Streamlit secrets if available
            if credentials_dict is None and hasattr(st, 'secrets'):
                credentials_dict = dict(st.secrets["gcp_service_account"])
            
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = ServiceAccountCredentials.from_json_keyfile_dict(
                credentials_dict, scope
            )
            
            client = gspread.authorize(credentials)
            
            # Try to open existing sheet, create if doesn't exist
            try:
                spreadsheet = client.open(self.sheet_name)
                self.worksheet = spreadsheet.sheet1
            except gspread.SpreadsheetNotFound:
                spreadsheet = client.create(self.sheet_name)
                self.worksheet = spreadsheet.sheet1
                # Share with your email so you can access it
                # spreadsheet.share('your-email@gmail.com', perm_type='user', role='writer')
            
            print(f"✅ Connected to Google Sheet: {self.sheet_name}")
            
        except Exception as e:
            print(f"❌ Error connecting to Google Sheets: {e}")
            raise
    
    def _ensure_headers(self):
        """Ensure the sheet has proper headers."""
        headers = [
            'id', 'smiles', 'smiles_hash', 'molecule_name', 'chembl_id',
            'prediction', 'prediction_probability', 'molecular_weight',
            'logp', 'h_bond_donors', 'h_bond_acceptors', 'rotatable_bonds',
            'tpsa', 'prediction_date', 'user_input_method', 'additional_info'
        ]
        
        try:
            existing_headers = self.worksheet.row_values(1)
            if not existing_headers or existing_headers != headers:
                self.worksheet.insert_row(headers, 1)
        except:
            self.worksheet.insert_row(headers, 1)
    
    def _get_smiles_hash(self, smiles: str) -> str:
        """Generate unique hash for SMILES."""
        return hashlib.md5(smiles.encode()).hexdigest()
    
    def _get_next_id(self) -> int:
        """Get the next available ID."""
        all_values = self.worksheet.get_all_values()
        if len(all_values) <= 1:  # Only headers
            return 1
        
        # Get last ID
        last_row = all_values[-1]
        try:
            return int(last_row[0]) + 1
        except:
            return len(all_values)  # Fallback to row count
    
    def check_molecule_exists(self, smiles: str) -> Optional[Dict]:
        """Check if molecule already exists in the sheet."""
        smiles_hash = self._get_smiles_hash(smiles)
        
        try:
            # Find cell with matching hash
            cell = self.worksheet.find(smiles_hash, in_column=3)  # Column 3 is smiles_hash
            
            if cell:
                # Get the entire row
                row_values = self.worksheet.row_values(cell.row)
                headers = self.worksheet.row_values(1)
                
                return dict(zip(headers, row_values))
        except:
            pass
        
        return None
    
    def add_prediction(self,
                      smiles: str,
                      prediction: int,
                      prediction_probability: float,
                      molecule_name: Optional[str] = None,
                      chembl_id: Optional[str] = None,
                      molecular_weight: Optional[float] = None,
                      logp: Optional[float] = None,
                      h_bond_donors: Optional[int] = None,
                      h_bond_acceptors: Optional[int] = None,
                      rotatable_bonds: Optional[int] = None,
                      tpsa: Optional[float] = None,
                      user_input_method: str = "single",
                      additional_info: Optional[str] = None) -> int:
        """Add a new prediction to the sheet."""
        
        # Check if exists
        existing = self.check_molecule_exists(smiles)
        if existing:
            print(f"Molecule already exists with ID: {existing.get('id')}")
            return int(existing.get('id', -1))
        
        # Create new row
        record_id = self._get_next_id()
        smiles_hash = self._get_smiles_hash(smiles)
        prediction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = [
            record_id,
            smiles,
            smiles_hash,
            molecule_name or '',
            chembl_id or '',
            prediction,
            prediction_probability,
            molecular_weight or '',
            logp or '',
            h_bond_donors or '',
            h_bond_acceptors or '',
            rotatable_bonds or '',
            tpsa or '',
            prediction_date,
            user_input_method,
            additional_info or ''
        ]
        
        # Append row
        self.worksheet.append_row(row)
        print(f"✅ Added prediction with ID: {record_id}")
        
        return record_id
    
    def get_all_predictions(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve all predictions as DataFrame."""
        all_values = self.worksheet.get_all_values()
        
        if len(all_values) <= 1:
            # No data, return empty DataFrame with headers
            return pd.DataFrame(columns=all_values[0] if all_values else [])
        
        df = pd.DataFrame(all_values[1:], columns=all_values[0])
        
        # Convert numeric columns
        numeric_cols = ['id', 'prediction', 'prediction_probability', 
                       'molecular_weight', 'logp', 'h_bond_donors', 
                       'h_bond_acceptors', 'rotatable_bonds', 'tpsa']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by ID descending (newest first)
        if 'id' in df.columns:
            df = df.sort_values('id', ascending=False)
        
        if limit:
            df = df.head(limit)
        
        return df
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        df = self.get_all_predictions()
        
        if df.empty:
            return {
                'total_predictions': 0,
                'bbb_permeable': 0,
                'bbb_non_permeable': 0,
                'avg_molecular_weight': None,
                'first_prediction': None,
                'latest_prediction': None
            }
        
        stats = {
            'total_predictions': len(df),
            'bbb_permeable': len(df[df['prediction'] == 1]),
            'bbb_non_permeable': len(df[df['prediction'] == 0]),
            'avg_molecular_weight': df['molecular_weight'].mean() if 'molecular_weight' in df.columns else None,
            'first_prediction': df['prediction_date'].min() if 'prediction_date' in df.columns else None,
            'latest_prediction': df['prediction_date'].max() if 'prediction_date' in df.columns else None
        }
        
        return stats
    
    def search_molecules(self,
                        query: str = None,
                        bbb_permeable: Optional[bool] = None,
                        min_mw: Optional[float] = None,
                        max_mw: Optional[float] = None) -> pd.DataFrame:
        """Search molecules with filters."""
        df = self.get_all_predictions()
        
        if df.empty:
            return df
        
        # Apply filters
        if query:
            mask = (
                df['smiles'].str.contains(query, case=False, na=False) |
                df['molecule_name'].str.contains(query, case=False, na=False) |
                df['chembl_id'].str.contains(query, case=False, na=False)
            )
            df = df[mask]
        
        if bbb_permeable is not None:
            df = df[df['prediction'] == (1 if bbb_permeable else 0)]
        
        if min_mw is not None and 'molecular_weight' in df.columns:
            df = df[df['molecular_weight'] >= min_mw]
        
        if max_mw is not None and 'molecular_weight' in df.columns:
            df = df[df['molecular_weight'] <= max_mw]
        
        return df
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bbb_predictions_export_{timestamp}.csv"
        
        df = self.get_all_predictions()
        df.to_csv(filename, index=False)
        return filename


# Singleton instance
_gsheets_db_instance = None

def get_database() -> GoogleSheetsDatabase:
    """Get or create Google Sheets database instance."""
    global _gsheets_db_instance
    if _gsheets_db_instance is None:
        _gsheets_db_instance = GoogleSheetsDatabase()
    return _gsheets_db_instance


# ============================================
# STREAMLIT SECRETS SETUP INSTRUCTIONS
# ============================================
"""
In Streamlit Cloud, add this to your secrets.toml:

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYour-Private-Key\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@project-id.iam.gserviceaccount.com"
client_id = "1234567890"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."

Steps:
1. Go to https://console.cloud.google.com
2. Create new project
3. Enable Google Sheets API
4. Create service account
5. Download JSON key
6. Copy contents to Streamlit secrets
7. Create a Google Sheet named "BBB_Predictions"
8. Share sheet with service account email (give Editor access)
"""