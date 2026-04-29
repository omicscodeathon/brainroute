import os
import threading
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem

TABLE_NAME = "molecules"
OPTIONAL_INSERT_FIELDS = {
    "created_at",
    "tags",
    "prediction_confidence",
    "logd",
    "cns_mpo",
}

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except Exception:
    pass


def _get_secret(name, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def _get_supabase_config():
    url = _get_secret("SUPABASE_URL")
    key = (
        _get_secret("SUPABASE_SERVICE_ROLE_KEY")
        or _get_secret("SUPABASE_KEY")
        or _get_secret("SUPABASE_ANON_KEY")
    )
    table = _get_secret("SUPABASE_MOLECULES_TABLE", TABLE_NAME)
    return url, key, table


def _headers(key):
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _native(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    return value


def _json_value(value):
    value = _native(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def _get_property(results, name):
    props = results.get("properties") or {}
    return _native(props.get(name, results.get(name)))


def _build_molecule_row(results):
    info = results.get("info") or {}
    smiles = _canonical_smiles(info.get("SMILES") or results.get("smiles"))
    prediction = results.get("prediction")
    bbb_tag = "BBB+" if prediction == "BBB+" else "BBB-" if prediction == "BBB-" else prediction

    row = {
        "name": info.get("Name") or results.get("name") or smiles,
        "smiles": smiles,
        "tpsa": _get_property(results, "tpsa"),
        "logp": _get_property(results, "logp"),
        "mw": _get_property(results, "mw") or _get_property(results, "molecular_weight"),
        "hbd": _get_property(results, "hbd"),
        "hba": _get_property(results, "hba"),
        "rotatable_bonds": _get_property(results, "rotatable_bonds"),
        "ring_count": _get_property(results, "ring_count"),
        "molar_refractivity": _get_property(results, "molar_refractivity"),
        "heterocycle_present": _get_property(results, "heterocycle_present"),
        "peptide_like": _get_property(results, "peptide_like"),
        "lipid_like": _get_property(results, "lipid_like"),
        "aromatic": _get_property(results, "aromatic"),
        "tpsa_bin": _get_property(results, "tpsa_bin"),
        "logp_bin": _get_property(results, "logp_bin"),
        "mw_bin": _get_property(results, "mw_bin"),
        "logd": _get_property(results, "logd"),
        "cns_mpo": _get_property(results, "cns_mpo"),
        "bbb_tag": bbb_tag,
        "tags": ["br_predicted"],
        "prediction_confidence": _native(results.get("confidence")),
        "lipinski_pass": _get_property(results, "lipinski_pass"),
        "veber_pass": _get_property(results, "veber_pass"),
        "egan_pass": _get_property(results, "egan_pass"),
        "ghose_pass": _get_property(results, "ghose_pass"),
        "pains_flag": _get_property(results, "pains_flag"),
        "profile_json": _json_value(_get_property(results, "profile_json")),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {key: _native(value) for key, value in row.items() if value is not None}


def _molecule_exists(url, key, table, smiles):
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
    response = requests.get(
        endpoint,
        headers=_headers(key),
        params={"select": "id", "smiles": f"eq.{smiles}", "limit": "1"},
        timeout=20,
    )
    response.raise_for_status()
    return bool(response.json())


def _insert_row(url, key, table, row):
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
    payload = dict(row)
    for _ in range(len(OPTIONAL_INSERT_FIELDS) + 1):
        response = requests.post(endpoint, headers=_headers(key), json=payload, timeout=20)
        if response.status_code != 400:
            response.raise_for_status()
            return

        message = response.text
        missing_column = re.search(r"Could not find the '([^']+)' column", message)
        if missing_column and missing_column.group(1) in OPTIONAL_INSERT_FIELDS:
            payload.pop(missing_column.group(1), None)
            continue

        removable = [field for field in OPTIONAL_INSERT_FIELDS if field in payload]
        if not removable:
            response.raise_for_status()
        payload.pop(removable[0], None)

    response.raise_for_status()


def add_prediction_to_supabase(results):
    if results.get("status") == "Error":
        return False
    url, key, table = _get_supabase_config()
    if not url or not key:
        print("SUPABASE_URL and SUPABASE_KEY/SUPABASE_SERVICE_ROLE_KEY are not configured")
        return False

    row = _build_molecule_row(results)
    smiles = row.get("smiles")
    if not smiles:
        return False
    raw_smiles = (results.get("info") or {}).get("SMILES") or results.get("smiles")
    raw_smiles = raw_smiles.strip() if isinstance(raw_smiles, str) else raw_smiles
    if _molecule_exists(url, key, table, smiles) or (
        raw_smiles and raw_smiles != smiles and _molecule_exists(url, key, table, raw_smiles)
    ):
        print(f"Skipped existing molecule: {smiles}")
        return False
    _insert_row(url, key, table, row)
    print(f"Added {row.get('name', smiles)} to Supabase")
    return True


def add_predictions_to_supabase(results):
    added = 0
    for result in results or []:
        try:
            if add_prediction_to_supabase(result):
                added += 1
        except Exception as exc:
            print(f"Error adding molecule to Supabase: {exc}")
    return added


def add_prediction_to_supabase_threaded(results):
    thread = threading.Thread(target=add_prediction_to_supabase, args=(results,), daemon=True)
    thread.start()


def add_predictions_to_supabase_threaded(results):
    thread = threading.Thread(target=add_predictions_to_supabase, args=(results,), daemon=True)
    thread.start()
