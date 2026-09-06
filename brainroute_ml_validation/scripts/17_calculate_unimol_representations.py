#!/usr/bin/env python3
"""Generate resumable frozen Uni-Mol v1 molecular representations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from unimol_tools import UniMolRepr
from unimol_tools.data.conformer import inner_smi2coords


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1] / "data/benchmarks/matched_3d_qm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.workspace / "artifacts/benchmark_manifest.csv")
    stop = len(manifest) if args.limit <= 0 else min(len(manifest), args.start + args.limit)
    selected = manifest.iloc[args.start:stop].copy()
    chunk_root = args.workspace / "cache/unimol_v1_audited/chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    encoder = UniMolRepr(
        data_type="molecule",
        batch_size=args.batch_size,
        remove_hs=False,
        model_name="unimolv1",
        use_cuda=False,
        max_atoms=256,
    )
    initialization_seconds = time.perf_counter() - started
    failures = []
    for start in range(args.start, stop, args.batch_size):
        end = min(stop, start + args.batch_size)
        path = chunk_root / f"rows_{start:06d}_{end:06d}.npz"
        if path.exists():
            print(f"cached rows {start}:{end}", flush=True)
            continue
        frame = manifest.iloc[start:end]
        batch_started = time.perf_counter()
        try:
            atoms_list = []
            coordinates_list = []
            conformer_is_3d = []
            for value in frame["calculation_smiles"]:
                atoms, coordinates, _ = inner_smi2coords(
                    value,
                    seed=42,
                    mode="fast",
                    remove_hs=False,
                )
                atoms_list.append(atoms)
                coordinates_list.append(coordinates)
                coordinate_array = np.asarray(coordinates)
                conformer_is_3d.append(
                    bool(coordinate_array.size and not np.allclose(coordinate_array[:, 2], 0.0))
                )
            representations = np.asarray(
                encoder.get_repr({"atoms": atoms_list, "coordinates": coordinates_list}),
                dtype=np.float32,
            )
            identifiers = frame["benchmark_id"].astype(str).to_numpy(dtype=str)
        except Exception as exc:
            batch_rows = []
            identifiers = []
            conformer_is_3d = []
            for _, row in frame.iterrows():
                try:
                    atoms, coordinates, _ = inner_smi2coords(
                        row["calculation_smiles"],
                        seed=42,
                        mode="fast",
                        remove_hs=False,
                    )
                    coordinate_array = np.asarray(coordinates)
                    is_3d = bool(coordinate_array.size and not np.allclose(coordinate_array[:, 2], 0.0))
                    representation = np.asarray(
                        encoder.get_repr({"atoms": [atoms], "coordinates": [coordinates]})[0],
                        dtype=np.float32,
                    )
                    batch_rows.append(representation)
                    identifiers.append(str(row["benchmark_id"]))
                    conformer_is_3d.append(is_3d)
                except Exception as inner:
                    failures.append(
                        {
                            "benchmark_id": row["benchmark_id"],
                            "batch_error": str(exc),
                            "individual_error": str(inner),
                        }
                    )
            representations = np.vstack(batch_rows) if batch_rows else np.empty((0, 512), dtype=np.float32)
            identifiers = np.asarray(identifiers, dtype=str)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            benchmark_id=np.asarray(identifiers, dtype=str),
            representation=representations,
            conformer_is_3d=np.asarray(conformer_is_3d, dtype=bool),
        )
        temporary.replace(path)
        print(
            f"rows {start}:{end} retained={len(identifiers)} seconds={time.perf_counter() - batch_started:.1f}",
            flush=True,
        )

    if failures:
        pd.DataFrame(failures).to_csv(args.workspace / "artifacts/unimol_failures.csv", index=False)

    all_ids = []
    all_repr = []
    all_is_3d = []
    for path in sorted(chunk_root.glob("rows_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            all_ids.extend(payload["benchmark_id"].astype(str).tolist())
            all_repr.append(payload["representation"].astype(np.float32))
            all_is_3d.extend(payload["conformer_is_3d"].astype(bool).tolist())
    combined = np.vstack(all_repr)
    index = pd.DataFrame({"benchmark_id": all_ids, "unimol_3d_conformer": all_is_3d})
    if index["benchmark_id"].duplicated().any():
        raise ValueError("Duplicate Uni-Mol identifiers were found across cached chunks")
    order = manifest[["benchmark_id"]].merge(index.reset_index(names="cache_position"), on="benchmark_id", how="inner")
    combined = combined[order["cache_position"].to_numpy()]
    index = order[["benchmark_id", "unimol_3d_conformer"]].copy()
    np.save(args.workspace / "artifacts/unimol_v1_representations.npy", combined)
    index.to_csv(args.workspace / "artifacts/unimol_v1_index.csv", index=False)
    metadata = {
        "model": "Uni-Mol v1 molecular pretraining model",
        "package": "unimol_tools",
        "package_version": "0.1.6",
        "device": "cpu",
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "max_atoms": 256,
        "n_representations": len(index),
        "n_three_dimensional_conformers": int(index["unimol_3d_conformer"].sum()),
        "n_two_dimensional_fallbacks_excluded_from_benchmark": int((~index["unimol_3d_conformer"]).sum()),
        "representation_dimension": int(combined.shape[1]),
        "initialization_seconds": initialization_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "frozen_representation": True,
    }
    with (args.workspace / "artifacts/unimol_run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
