"""One uncached mechanistic verification contract for ML selections and exports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import model as current_core
from scientific_provenance import ROOT, compatibility_errors, file_hash

# These compare two implementations of the SAME equations, not grid convergence.
# Chosen before evaluation: near-roundoff agreement, well below solver tolerances.
CONTROL_CONCENTRATION_ATOL = 1e-10
CONTROL_TIME_ATOL_MIN = 1e-8
CONTROL_INPUTS = ((8.5, 8.0, 18.0, .40, 3., 7.56, .045, .017, .0117),
                  (7.8, 7.2, 15.0, .38, 6., 7.2, .09, .014, 2.7),
                  (9.2, 9.0, 22.0, .42, 2.5, 8.1, .08, .03, .02))


@dataclass
class VerificationConfig:
    times: np.ndarray
    n_cells: int
    rtol: float
    atol: float
    ratio: float
    identity: dict


def resolve_verification_config(metadata):
    """Missing source/grid metadata is an explicit failure, never a default grid."""
    if metadata.get("task_type") != "breakthrough" or metadata.get("target_column") != "breakthrough_time_min":
        raise ValueError("A breakthrough / breakthrough_time_min model is required")
    errors = compatibility_errors(metadata)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(metadata["generation_source"]["manifest_path"])
    path = path if path.is_absolute() else ROOT / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    profile = manifest.get("profile") or {}
    keys = ("time_start_min", "time_end_min", "time_step_min")
    if any(key not in profile for key in keys):
        raise ValueError("Verified generation manifest has no explicit time_start/end/step_min profile")
    start, end, step = (float(profile[key]) for key in keys)
    if start != 0 or not np.isfinite([start, end, step]).all() or end <= start or step <= 0:
        raise ValueError("Unsupported or invalid source time grid")
    if metadata["breakthrough_ratio"] != .05 or profile.get("breakthrough_ratio") != .05:
        raise ValueError("Source breakthrough ratio must be 0.05")
    times = np.arange(start, end + .5 * step, step)
    if abs(times[-1] - end) > 1e-10:
        raise ValueError("Source time horizon is not an integer multiple of its step")
    if profile.get("n_cells") != metadata["n_cells"]:
        raise ValueError("Source profile and model spatial grids differ")
    expected = (metadata.get("code_sha256") or {}).get("model.py")
    if not expected:
        raise ValueError("Source mechanistic model SHA-256 is missing")
    actual = file_hash(ROOT / "model.py")
    identity = {"verification_contract": "selection_source_grid_v1", "source_core_sha256": expected,
                "executed_core_sha256": actual, "source_manifest_sha256": file_hash(path),
                "model_id": metadata.get("model_id"), "n_cells": int(metadata["n_cells"]),
                "solver": metadata["solver"], "rtol": metadata["solver_rtol"], "atol": metadata["solver_atol"],
                "time_start_min": start, "time_end_min": end, "time_step_min": step,
                "breakthrough_ratio": .05, "boundary_conditions": metadata["boundary_conditions"],
                "initial_conditions_policy": metadata["initial_conditions_policy"],
                "parameter_policy": "candidate_effective_KH_kLDF_Dax_no_second_temperature_correction"}
    config = VerificationConfig(times, int(metadata["n_cells"]), float(metadata["solver_rtol"]),
                                float(metadata["solver_atol"]), .05, identity)
    if actual == expected:
        identity["core_compatibility"] = "identical_source"
    else:
        candidates = [ROOT / "equilibrium_audit_2026_10_20/before/model.py",
                      ROOT / "backups/laboratory_update_2026_10_13/model.py"]
        source = next((p for p in candidates if file_hash(p) == expected), None)
        if source is None:
            raise ValueError("Current core differs; exact training core is unavailable for compatibility controls")
        name = "_selection_source_core"
        spec = importlib.util.spec_from_file_location(name, source)
        original = importlib.util.module_from_spec(spec)
        sys.modules[name] = original
        spec.loader.exec_module(original)
        differences = []
        for values in CONTROL_INPUTS:
            column = current_core.ColumnParams(*values[:6])
            params = current_core.SorptionParams(*values[6:])
            a = _solve(column, params, config, current_core)
            b = _solve(column, params, config, original)
            concentration_error = float(np.max(np.abs(a[0] - b[0])))
            time_error = abs(a[1].breakthrough_time_min - b[1].breakthrough_time_min)
            if (concentration_error > CONTROL_CONCENTRATION_ATOL or time_error > CONTROL_TIME_ATOL_MIN
                    or a[1].breakthrough_bound_type != b[1].breakthrough_bound_type):
                raise ValueError("Current mechanistic core failed the predefined source compatibility controls")
            differences.append({"max_concentration_difference_mg_dm3": concentration_error,
                                "breakthrough_time_difference_min": time_error})
        identity.update(core_compatibility="different_source_matches_three_fixed_controls",
                        compatibility_controls=differences, control_inputs=CONTROL_INPUTS,
                        concentration_atol=CONTROL_CONCENTRATION_ATOL, time_atol_min=CONTROL_TIME_ATOL_MIN)
    return config


def _solve(column, params, config, core=current_core):
    concentration = core.simulate_breakthrough(config.times, column, params, n_cells=config.n_cells,
                                              rtol=config.rtol, atol=config.atol)
    interpreted = core.interpret_breakthrough_curve(config.times, concentration / column.c0_mg_l,
        c0_mg_l=column.c0_mg_l, breakthrough_ratio=config.ratio)
    return concentration, interpreted


def candidate_inputs(candidate, metadata):
    """Read stored effective candidate inputs, not current mutable UI controls."""
    from ml_adsorber import normalize_units_for_ml, normalize_scale_mode
    row = dict(candidate)
    prefixed = {key[8:]: value for key, value in row.items() if key.startswith("feature_")} if "feature_C0_mg_L" in row else {}
    raw = prefixed or row
    if row.get("model_id") not in (None, metadata.get("model_id")):
        raise ValueError("Candidate belongs to another ML model; generate candidates again")
    ratio = raw.get("C_breakthrough_ratio", row.get("breakthrough_ratio", .05))
    if float(ratio) != .05:
        raise ValueError("Candidate breakthrough ratio differs from its training source")
    scale = normalize_scale_mode(raw.get("scale_mode", "laboratory"))
    normalized, _ = normalize_units_for_ml(pd.DataFrame([raw]), metadata, scale)
    values = normalized.iloc[0]
    units = metadata.get("feature_units") or {}

    def value(name, aliases=(), factors=None):
        for alias in aliases:
            if alias in raw and pd.notna(raw[alias]):
                return float(raw[alias])
        if name not in values or not np.isfinite(float(values[name])):
            raise ValueError(f"Missing physical candidate input: {name}")
        unit = units.get(name)
        if factors is not None and unit not in factors:
            raise ValueError(f"Unknown physical unit for {name}: {unit}")
        return float(values[name]) * (factors[unit] if factors else 1.)

    height = value("height", ("bed_height_cm", "height_cm"), {"cm": 1., "m": 100.})
    diameter = value("diameter", ("column_diameter_cm", "diameter_cm"), {"cm": 1., "m": 100.})
    flow = value("flow", ("flow_ml_min",), {"ml/min": 1., "m3/h": 1e6 / 60.})
    mass = value("carbon_mass", ("carbon_mass_g",), {"g": 1., "kg": 1000.})
    column = current_core.ColumnParams(height, np.pi * diameter ** 2 / 4., mass,
        value("epsilon"), flow, value("C0_mg_L"))
    params = current_core.SorptionParams(value("KH_used_L_g", ("KH_L_g",)),
        value("kLDF_used_1_min", ("kLDF_1_min",)), value("Dax_used_cm2_min", ("Dax_cm2_min",)))
    return column, params


def verify_candidate(candidate, metadata, config=None):
    config = config or resolve_verification_config(metadata)
    if config.identity["model_id"] != metadata.get("model_id"):
        raise ValueError("Verification configuration belongs to another model")
    column, params = candidate_inputs(candidate, metadata)
    concentration, interpreted = _solve(column, params, config)
    required = float(candidate["required_breakthrough_time_min"] if "required_breakthrough_time_min" in candidate
                     else float(candidate["required_time_h"]) * 60.)
    exact_time = interpreted.breakthrough_time_min if interpreted.is_exact else None
    predicted = candidate.get("predicted_breakthrough_time_min", np.nan)
    discrepancy = float(predicted) - exact_time if exact_time is not None else None
    record = {key: candidate[key] for key in ("candidate_id", "model_id", "bundle_source", "rank",
              "source_experiment_id", "sorption_parameter_source", "parameter_provenance_json") if key in candidate}
    for key in ("candidate_id", "rank"):
        if key in record:
            record[key] = int(record[key])
    record["parameter_origin_status"] = "recorded" if candidate.get("parameter_provenance_json") else "not_recorded_in_legacy_candidate"
    record.update(mechanistic_breakthrough_status=interpreted.breakthrough_status,
        mechanistic_bound_type=interpreted.breakthrough_bound_type,
        mechanistic_breakthrough_time_min=exact_time,
        mechanistic_observation_bound_min=interpreted.breakthrough_time_min if not interpreted.is_exact else None,
        ml_breakthrough_time_min=float(predicted), ML_minus_mechanistic_min=discrepancy,
        absolute_discrepancy_min=abs(discrepancy) if discrepancy is not None else None,
        relative_discrepancy_percent=100. * discrepancy / exact_time if exact_time and discrepancy is not None else None,
        required_breakthrough_time_min=required,
        mechanistic_passes_required_time=bool(interpreted.is_exact and exact_time >= required),
        verification_config_json=json.dumps(config.identity, sort_keys=True),
        effective_parameters_json=json.dumps(asdict(params), sort_keys=True),
        physical_inputs_json=json.dumps(asdict(column), sort_keys=True))
    return {"record": record, "time_min": config.times.copy(), "C_out_mg_L": concentration,
            "C_out_C0": concentration / column.c0_mg_l, "breakthrough": interpreted.to_dict(),
            "configuration": config.identity, "column": column, "sorption": params}


def verification_export_bytes(result):
    """Export the already computed result; never invoke another simulation."""
    return pd.DataFrame([result["record"]]).to_csv(index=False).encode("utf-8-sig")
