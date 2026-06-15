#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.utils import LOGGER, ensure_dirs, load_config, project_path, read_table, script_arg_parser, set_global_seed, write_csv


def main() -> None:
    args = script_arg_parser("Calculate optional pretrained SMILES embeddings.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    emb_cfg = cfg.get("pretrained_embeddings", {})
    if not emb_cfg.get("use_pretrained_embeddings", False):
        LOGGER.info("Pretrained embeddings disabled in config; skipping.")
        return
    # This workflow only needs PyTorch. Some environments have TensorFlow installed
    # but broken; force Transformers to avoid TensorFlow imports.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        LOGGER.warning("Embedding dependencies unavailable; skipping embeddings: %s", exc)
        return

    df = read_table(project_path(cfg, "data/processed/standardized_molecules.csv"))
    model_name = emb_cfg.get("model_name", "DeepChem/ChemBERTa-77M-MLM")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        model.eval()
    except Exception as exc:
        LOGGER.warning("Could not load pretrained embedding model %s; skipping embeddings: %s", model_name, exc)
        return

    embeddings = []
    batch_size = int(emb_cfg.get("batch_size", 32))
    max_length = int(emb_cfg.get("max_length", 256)) 
    pooling = emb_cfg.get("pooling", "cls")
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch = df["canonical_smiles"].iloc[start : start + batch_size].tolist()
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            outputs = model(**tokens)
            hidden = outputs.last_hidden_state
            if pooling == "mean":
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden[:, 0, :]
            embeddings.append(pooled.cpu().numpy())
    arr = np.vstack(embeddings)
    base = project_path(cfg, "data/processed")
    emb_path = base / "pretrained_smiles_embeddings.npy"
    if not emb_path.exists() or cfg.get("overwrite", False):
        np.save(emb_path, arr)
    write_csv(df, base / "pretrained_embedding_index.csv", cfg)
    info = {"model_name": model_name, "pooling": pooling, "shape": list(arr.shape), "frozen_feature_extractor": True}
    with (project_path(cfg, "reports/pretrained_embedding_model_info.json")).open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


if __name__ == "__main__":
    main()
