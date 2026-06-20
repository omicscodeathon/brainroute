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
AUTH_HANDOFFS_TABLE = "auth_handoffs"
PREDICTION_LOGS_TABLE = "user_prediction_runs"
PREDICTION_BATCHES_TABLE = "prediction_batches"
BRAINROUTE_DB_URL = "https://omicscodeathon.github.io/brainroutedb"
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


BRAINROUTE_DB_URL = _get_secret("BRAINROUTE_DB_URL", BRAINROUTE_DB_URL)


def _get_supabase_config():
    url = _get_secret("SUPABASE_URL")
    key = (
        _get_secret("SUPABASE_SERVICE_ROLE_KEY")
        or _get_secret("SUPABASE_KEY")
        or _get_secret("SUPABASE_ANON_KEY")
    )
    table = _get_secret("SUPABASE_MOLECULES_TABLE", TABLE_NAME)
    return url, key, table


def _get_service_role_config():
    return _get_secret("SUPABASE_URL"), _get_secret("SUPABASE_SERVICE_ROLE_KEY")


def has_service_role_config():
    url, service_key = _get_service_role_config()
    return bool(url and service_key)


def _get_table(name, default):
    return _get_secret(name, default)


def _headers(key, prefer="return=minimal"):
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": prefer,
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


def _safe_json(value):
    value = _native(value)
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items() if k != "mol"}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


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


def redeem_auth_handoff(code):
    if not code:
        return None

    url, service_key = _get_service_role_config()
    if not url or not service_key:
        return None

    table = _get_table("SUPABASE_AUTH_HANDOFFS_TABLE", AUTH_HANDOFFS_TABLE)
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"

    try:
        response = requests.get(
            endpoint,
            headers=_headers(service_key),
            params={
                "select": "id,user_id,expires_at,used_at",
                "code": f"eq.{code}",
                "limit": "1",
            },
            timeout=20,
        )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return None

        row = rows[0]
        if row.get("used_at"):
            return None

        expires_at = _parse_datetime(row.get("expires_at"))
        if not expires_at or expires_at <= datetime.now(timezone.utc):
            return None

        update_response = requests.patch(
            endpoint,
            headers=_headers(service_key),
            params={"id": f"eq.{row['id']}"},
            json={"used_at": datetime.now(timezone.utc).isoformat()},
            timeout=20,
        )
        update_response.raise_for_status()

        return {"user_id": row.get("user_id"), "handoff_id": row.get("id")}
    except Exception as exc:
        print(f"Unable to redeem BrainRoute handoff: {exc}")
        return None


def _prediction_probability(results):
    value = results.get("prediction_probability") or results.get("probability")
    if value is not None:
        return _native(value)
    confidence = results.get("confidence")
    if confidence is None:
        return None
    try:
        confidence = float(confidence)
        return confidence / 100 if confidence > 1 else confidence
    except (TypeError, ValueError):
        return None


def _build_prediction_log_row(results, user_id, batch_id=None, input_mode=None):
    info = results.get("info") or {}
    raw_smiles = info.get("SMILES") or results.get("smiles")
    smiles = raw_smiles.strip() if isinstance(raw_smiles, str) else raw_smiles
    canonical_smiles = _canonical_smiles(smiles)
    properties = results.get("properties") or {}

    row = {
        "user_id": user_id,
        "source_app": "brainroute_streamlit",
        "input_mode": input_mode,
        "batch_id": batch_id,
        "molecule_name": info.get("Name") or results.get("name") or canonical_smiles,
        "smiles": smiles or canonical_smiles,
        "canonical_smiles": canonical_smiles,
        "prediction_label": results.get("prediction"),
        "prediction_probability": _prediction_probability(results),
        "confidence": _native(results.get("confidence")),
        "uncertainty": _native(results.get("uncertainty")),
        "model_name": results.get("model_name") or "strict_validation_ensemble",
        "feature_set": results.get("feature_set") or "padel_descriptors",
        "molecular_properties": _safe_json(properties),
        "model_outputs": _safe_json({
            "padel_preds": results.get("padel_preds"),
            "padel_confs": results.get("padel_confs"),
        }),
        "raw_result": _safe_json(results),
    }
    return {key: value for key, value in row.items() if value is not None}


def log_user_prediction(results, user_id, batch_id=None, input_mode=None):
    if not user_id or not results or results.get("status") == "Error":
        return False

    url, service_key = _get_service_role_config()
    if not url or not service_key:
        return False

    row = _build_prediction_log_row(results, user_id, batch_id=batch_id, input_mode=input_mode)
    if not row.get("smiles"):
        return False

    table = _get_table("SUPABASE_PREDICTION_LOGS_TABLE", PREDICTION_LOGS_TABLE)
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"

    try:
        response = requests.post(endpoint, headers=_headers(service_key), json=row, timeout=20)
        response.raise_for_status()
        return True
    except Exception as exc:
        print(f"Unable to log user prediction: {exc}")
        return False


def log_user_prediction_batch(results, user_id, input_type=None, batch_name=None):
    if not user_id:
        return None

    url, service_key = _get_service_role_config()
    if not url or not service_key:
        return None

    results = results or []
    successful = [result for result in results if result.get("status") == "Success"]
    failed = len(results) - len(successful)
    row = {
        "user_id": user_id,
        "source_app": "brainroute_streamlit",
        "batch_name": batch_name,
        "input_type": input_type,
        "total_molecules": len(results),
        "successful_molecules": len(successful),
        "failed_molecules": failed,
        "summary_json": _safe_json({
            "prediction_counts": {
                label: sum(1 for result in successful if result.get("prediction") == label)
                for label in sorted({result.get("prediction") for result in successful if result.get("prediction")})
            }
        }),
    }

    table = _get_table("SUPABASE_PREDICTION_BATCHES_TABLE", PREDICTION_BATCHES_TABLE)
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"

    try:
        response = requests.post(
            endpoint,
            headers=_headers(service_key, prefer="return=representation"),
            json=row,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0].get("id")
        if isinstance(data, dict):
            return data.get("id")
    except Exception as exc:
        print(f"Unable to log user prediction batch: {exc}")
    return None


def log_user_predictions(results, user_id, batch_id=None):
    added = 0
    for result in results or []:
        try:
            if log_user_prediction(result, user_id, batch_id=batch_id, input_mode="batch"):
                added += 1
        except Exception as exc:
            print(f"Unable to log one user prediction: {exc}")
    return added
