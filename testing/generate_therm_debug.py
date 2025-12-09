# Utility script to generate Data Debugger style outputs (therm_merged_engine.csv,
# therm_debug_bundle.json, therm_raw_prephysics.csv, therm_readme.txt) using the
# same sample files and profiles that the regression runner uses.
#
# Usage (from repo root):
#   $env:PYTHONPATH='.'; .\.venv\Scripts\python.exe testing\generate_therm_debug.py
#
# Outputs are written under testing/artifacts/therm_debug_<mode>_<timestamp>/

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

import ha_loader
import data_loader
import processing
from testing.regression_runner import SamplePaths


def _ts_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _write_outputs(
    out_dir: Path,
    merged_df: pd.DataFrame,
    raw_df: pd.DataFrame | None,
    debug_bundle: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_dir / "therm_merged_engine.csv", index=True)
    if raw_df is not None:
        raw_df.to_csv(out_dir / "therm_raw_prephysics.csv", index=True)
    (out_dir / "therm_debug_bundle.json").write_text(
        json.dumps(debug_bundle, indent=2, default=str), encoding="utf-8"
    )

    readme = (
        "THERM Debug Export README\n"
        f"Generated at: {datetime.now(timezone.utc).isoformat()}\n\n"
        "Files in this directory:\n"
        "- therm_merged_engine.csv: processed engine dataframe (post-physics, tariff, runs)\n"
        "- therm_debug_bundle.json: metadata/config/runs summary\n"
        "- therm_raw_prephysics.csv: pre-physics dataframe (if available)\n"
    )
    (out_dir / "therm_readme.txt").write_text(readme, encoding="utf-8")


def _build_debug_bundle(
    dataset_source: str,
    merged_df: pd.DataFrame,
    raw_df: pd.DataFrame | None,
    config: Dict[str, Any],
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stats = processing.compute_global_stats(merged_df)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        "config": config,
        "global_stats": stats,
        "runs": runs,
        "raw_present": raw_df is not None,
    }


def run_ha(paths: SamplePaths) -> None:
    with open(paths.ha_profile, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # ha_loader expects file-like objects; open in binary mode to mimic uploads
    with open(paths.ha_csv, "rb") as f_ha:
        res = ha_loader.process_ha_files([f_ha], profile, progress_cb=lambda *args, **kwargs: None)

    df = res["df"]
    raw = res.get("raw_history")
    runs = res.get("runs", [])
    bundle = _build_debug_bundle("ha", df, raw, profile, runs)

    out_dir = paths.artifacts / f"therm_debug_ha_{_ts_label()}"
    _write_outputs(out_dir, df, raw, bundle)
    print(f"[ha] written to {out_dir}")


def run_grafana(paths: SamplePaths) -> None:
    with open(paths.grafana_profile, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # data_loader expects file-like objects (numeric + state)
    with open(paths.grafana_numeric, "rb") as f_num, open(paths.grafana_state, "rb") as f_state:
        res = data_loader.load_and_clean_data(
            [f_num, f_state], profile, progress_cb=lambda *args, **kwargs: None
        )

    df_raw = res["df"]
    df = processing.apply_gatekeepers(df_raw, profile)
    runs = processing.detect_runs(df, profile)
    raw_history = res.get("raw_history")
    bundle = _build_debug_bundle("grafana", df, raw_history, profile, runs)

    out_dir = paths.artifacts / f"therm_debug_grafana_{_ts_label()}"
    _write_outputs(out_dir, df, raw_history, bundle)
    print(f"[grafana] written to {out_dir}")


def main() -> None:
    base = Path(__file__).resolve().parent
    paths = SamplePaths(base=base)
    run_ha(paths)
    run_grafana(paths)


if __name__ == "__main__":
    main()
