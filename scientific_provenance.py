"""Actual runtime and source identities, not guessed dependency versions."""
from importlib.metadata import version, PackageNotFoundError
import hashlib
from pathlib import Path
import platform
import json

from model import MODEL_VERSION, DEFAULT_N_CELLS, SOLVER_RTOL, SOLVER_ATOL

ROOT = Path(__file__).resolve().parent
CONTRACT_VERSION = "scientific_selection_contract_v1"
INITIAL_CONDITIONS = "zero_liquid_and_adsorbed_stock"
DATAFRAME_HASH_ALGORITHM = "SHA-256(pandas.hash_pandas_object(index=False).uint64_bytes)"


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def current_provenance(n_cells=DEFAULT_N_CELLS):
    dependencies = {}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "joblib", "streamlit", "optuna"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    import_manifest_path = ROOT / "experimental_import_manifest.json"
    import_manifest = json.loads(import_manifest_path.read_text(encoding="utf-8")) if import_manifest_path.exists() else {}
    return {
        "model_version": MODEL_VERSION, "n_cells": int(n_cells),
        "boundary_conditions": "total_inlet_flux_uC0_outlet_zero_dispersive_gradient",
        "solver": "BDF", "solver_rtol": SOLVER_RTOL, "solver_atol": SOLVER_ATOL,
        "python_version": platform.python_version(), "dependencies": dependencies,
        "laboratory_csv_sha256": file_hash(Path(__file__).parent / "cleaned_dynamic_sorption_data.csv"),
        "breakthrough_ratio": 0.05, "random_seed": 42,
        "small_signal_policy": "retain_algebraic_unclipped_no_inferred_blank_LOD_LOQ",
        "initial_conditions_policy": INITIAL_CONDITIONS,
        "effective_parameters_source": "effective_parameter_comparison.csv",
        "effective_parameters_sha256": file_hash(ROOT / "effective_parameter_comparison.csv"),
        "primary_excel_sha256": import_manifest.get("excel_sha256"),
        "primary_measurements_csv_sha256": file_hash(ROOT / "laboratory_individual_runs.csv"),
        "analytical_calibration_sha256": file_hash(ROOT / "analytical_calibration_points.csv"),
        "code_sha256": {name: file_hash(ROOT / name) for name in
                        ("model.py", "mass_balance.py", "import_experimental_excel.py",
                         "ml_adsorber.py", "ml_validation.py", "thesis_ml_workflow.py", "recommended_ranges.py")},
    }


def provenance_is_current(metadata):
    return not compatibility_errors(metadata)


def dataframe_hash(df):
    import pandas as pd
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def compatibility_errors(metadata):
    """Exact scientific contract; no untested alternative grid is permitted."""
    metadata = metadata or {}
    current = current_provenance()
    errors = []
    for key in ("model_version", "boundary_conditions", "n_cells", "breakthrough_ratio",
                "solver", "solver_rtol", "solver_atol", "initial_conditions_policy",
                "laboratory_csv_sha256", "effective_parameters_source", "effective_parameters_sha256"):
        if key not in metadata or metadata[key] != current[key]:
            errors.append(f"{key}: expected {current[key]!r}, received {metadata.get(key)!r}")
    if metadata.get("compatibility_contract") != CONTRACT_VERSION:
        errors.append("compatibility_contract: missing or unsupported")
    generation = metadata.get("generation_source") or {}
    manifest_path = generation.get("manifest_path")
    if not manifest_path:
        errors.append("generation_source: verified manifest is missing")
        return errors
    path = Path(manifest_path)
    path = path if path.is_absolute() else ROOT / path
    if not path.exists() or file_hash(path) != generation.get("manifest_sha256"):
        errors.append("generation_source: manifest file missing or SHA-256 mismatch")
        return errors
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return errors + ["generation_source: unreadable manifest"]
    for key in ("model_version", "boundary_conditions", "n_cells", "breakthrough_ratio",
                "solver", "solver_rtol", "solver_atol", "initial_conditions_policy",
                "laboratory_csv_sha256", "effective_parameters_source", "effective_parameters_sha256"):
        if manifest.get(key) != metadata.get(key):
            errors.append(f"generation_source: {key} differs from model metadata")
    if manifest.get("compatibility_contract") != CONTRACT_VERSION:
        errors.append("generation_source: unsupported manifest contract")
    for key in ("dataset_csv_sha256", "dataset_csv_sha256_algorithm", "generation_method"):
        if not manifest.get(key) or manifest.get(key) != generation.get(key):
            errors.append(f"generation_source: {key} missing or inconsistent")
    dataset_path = manifest.get("dataset_csv_path")
    if not dataset_path:
        errors.append("generation_source: dataset CSV reference is missing")
    else:
        dataset_path = Path(dataset_path)
        dataset_path = dataset_path if dataset_path.is_absolute() else ROOT / dataset_path
        if file_hash(dataset_path) != manifest.get("dataset_csv_sha256"):
            errors.append("generation_source: dataset CSV bytes changed or unavailable")
    return errors


def record_generation_manifest(df, dataset_path, manifest_path, generation_method, **details):
    """Record an actual generation, not an inference from a MODEL_VERSION cell."""
    import pandas as pd
    from ml_adsorber import canonicalize_ml_columns
    dataset_path, manifest_path = Path(dataset_path), Path(manifest_path)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    n_values = pd.to_numeric(df.get("n_cells", pd.Series(dtype=float)), errors="coerce").dropna().unique()
    if len(n_values) != 1:
        raise ValueError("Generation manifest requires a single recorded n_cells.")
    if n_values[0] != int(n_values[0]):
        raise ValueError("Recorded n_cells must be an integer.")
    n_cells = int(n_values[0])
    model_versions = df.get("mechanistic_model_id", pd.Series(dtype=str)).dropna().unique().tolist()
    if model_versions != [MODEL_VERSION]:
        raise ValueError("Generation manifest requires the actual current mechanistic model identity.")
    thresholds = pd.to_numeric(df.get("C_breakthrough_ratio", pd.Series(dtype=float)), errors="coerce").dropna().unique().tolist()
    if not thresholds:
        raise ValueError("Generation manifest requires recorded per-case breakthrough thresholds.")
    manifest = {**details, **current_provenance(n_cells),
                "compatibility_contract": CONTRACT_VERSION,
                "generation_method": generation_method,
                "dataset_csv_path": str(dataset_path.relative_to(ROOT)),
                "dataset_csv_sha256": file_hash(dataset_path),
                "dataset_csv_sha256_algorithm": "SHA-256(file_bytes)",
                "dataset_dataframe_sha256": dataframe_hash(df),
                "dataset_dataframe_sha256_algorithm": DATAFRAME_HASH_ALGORITHM,
                "dataset_canonical_dataframe_sha256": dataframe_hash(canonicalize_ml_columns(df)),
                "dataframe_hash_columns": list(df.columns)}
    manifest["generation_breakthrough_ratios"] = sorted(thresholds)
    manifest["breakthrough_ratio"] = thresholds[0] if len(thresholds) == 1 else "per_case_C_breakthrough_ratio"
    # CSV parsing can change last binary digits/dtypes without changing file bytes.
    read_df = pd.read_csv(dataset_path)
    manifest["dataset_csv_read_dataframe_sha256"] = dataframe_hash(read_df)
    manifest["dataset_csv_read_canonical_dataframe_sha256"] = dataframe_hash(canonicalize_ml_columns(read_df))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return generation_source_from_manifest(manifest_path)


def generation_source_from_manifest(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    fields = {key: manifest.get(key) for key in current_provenance()}
    fields.update({"compatibility_contract": manifest.get("compatibility_contract"),
                  "generation_source": {"manifest_path": str(path.relative_to(ROOT)),
                    "manifest_sha256": file_hash(path),
                    **{key: manifest.get(key) for key in ("dataset_csv_sha256", "dataset_csv_sha256_algorithm", "generation_method")}}})
    return fields


def resolve_dataset_provenance(df, supplied=None):
    """Verify dataframe identity against a real local generation manifest."""
    from ml_adsorber import canonicalize_ml_columns
    supplied = supplied or df.attrs.get("generation_provenance") or {}
    paths = []
    if supplied.get("generation_source", {}).get("manifest_path"):
        paths.append(ROOT / supplied["generation_source"]["manifest_path"])
    paths.extend([ROOT / "thesis_final_results/synthetic_generation_manifest.json",
                  *sorted((ROOT / "generated_datasets").glob("*_manifest.json"))])
    raw_hash, canonical_hash = dataframe_hash(df), dataframe_hash(canonicalize_ml_columns(df))
    for path in dict.fromkeys(paths):
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if raw_hash not in {manifest.get("dataset_dataframe_sha256"), manifest.get("dataset_csv_read_dataframe_sha256")} and canonical_hash not in {manifest.get("dataset_canonical_dataframe_sha256"), manifest.get("dataset_csv_read_canonical_dataframe_sha256")}:
            continue
        verified = generation_source_from_manifest(path)
        if not compatibility_errors(verified):
            return verified
    return {"dataset_provenance_status": "unverified_origin", "generation_source": None}
