"""Non-numerical, per-component lineage. Never infer evidence from a value."""
from __future__ import annotations

from copy import deepcopy
import json
import math


PARAMETER_UNITS = {
    "KH_L_g": "L/g", "kLDF_1_min": "1/min", "Dax_cm2_min": "cm2/min",
    "qmax_mg_g": "mg/g", "b_L_mg": "L/mg", "deltaH_ads_kJ_mol": "kJ/mol",
    "Ea_kLDF_kJ_mol": "kJ/mol", "D_m_ref_m2_s": "m2/s",
    "T_ref_C": "degC", "T_exp_C": "degC", "T_sim_C": "degC",
}
ORIGINS = {"fitted", "profile", "literature", "user", "derived", "default", "unknown"}


def provenance_record(origin, *, method=None, **evidence):
    if origin not in ORIGINS:
        raise ValueError(f"Unsupported parameter origin: {origin}")
    if origin == "derived" and method == "range_sampling":
        evidence.setdefault("formula", "value = sample(configured_range, sampling_method, random_seed)")
    if origin == "derived" and method == "training_statistic":
        evidence.setdefault("formula", "value = feature_ranges[parameter][statistic]")
    return {"origin": origin, **({"method": method} if method else {}),
            **{key: deepcopy(value) for key, value in evidence.items() if value is not None}}


def parameter_origins(payload):
    """Read legacy JSON/CSV mappings without changing the input or its identity."""
    records = payload.get("parameter_provenance", {})
    if isinstance(records, str):
        try:
            records = json.loads(records)
        except (ValueError, TypeError):
            records = {}
    if not isinstance(records, dict):
        records = {}
    return {key: deepcopy(record) for key, record in records.items()
            if isinstance(record, dict) and record.get("origin") in ORIGINS}


def parameter_origin(payload, parameter):
    return parameter_origins(payload).get(parameter, provenance_record("unknown", method="legacy_unknown"))


def derived_record(formula, dependencies, records, *, method="calculated", dependency_values=None, **evidence):
    return provenance_record("derived", method=method, formula=formula,
                             dependencies=list(dependencies),
                             dependency_provenance={key: deepcopy(records.get(key, provenance_record("unknown", method="legacy_unknown")))
                                                    for key in dependencies},
                             dependency_values=dependency_values, **evidence)


def declared_profile_ratio(profile_key, kh_literal, qmax_literal):
    inputs = {"profile_definition.KH_literal": kh_literal, "profile_definition.qmax_literal": qmax_literal}
    return derived_record(f"b_L_mg = {kh_literal} / {qmax_literal}", inputs,
                          {key: provenance_record("profile", method="profile_declared", profile_key=profile_key,
                                                  declared_literal=value) for key, value in inputs.items()},
                          method="profile_expression", profile_key=profile_key, dependency_values=inputs)


def geometry_records(scale_mode, input_origin):
    records = {key: provenance_record(input_origin, method="user_input" if input_origin == "user" else "range_sampling")
               for key in ("flow", "diameter", "dp_mm")}
    laboratory = "lab" in str(scale_mode).lower() or "лаборатор" in str(scale_mode).lower()
    formula = ("(flow_ml_min / (pi * diameter_cm**2 / 4)) * 0.01 / 60" if laboratory
               else "(flow_m3_h / 3600) / (pi * diameter_m**2 / 4)")
    records["superficial_velocity_m_s"] = derived_record(formula, ("flow", "diameter"), records, method="geometry_conversion")
    records["dp_m"] = derived_record("dp_mm / 1000", ("dp_mm",), records, method="geometry_conversion")
    return records


def effective_input_payload(base, values, user_parameters=(), inherited=None):
    """Annotate actual widget values, not a mode's untouched starting defaults."""
    result = deepcopy(base)
    records = parameter_origins(base)
    for key, value in values.items():
        if key in user_parameters:
            records[key] = provenance_record("user", method="user_input", input_field=key)
        elif inherited and key in inherited:
            records[key] = deepcopy(inherited[key])
        elif value != base.get(key):
            records[key] = provenance_record("user", method="user_input", input_field=key)
        else:
            records[key] = parameter_origin(base, key)
        result[key] = value
    result["parameter_provenance"] = records
    return result


def temperature_lineage(adaptation, reference_payload):
    """Attach lineage to already calculated numbers; no correction is rerun."""
    result = deepcopy(adaptation)
    records = parameter_origins(reference_payload)
    aliases = {"KH_ref_L_g": "KH_L_g", "kLDF_ref_1_min": "kLDF_1_min", "Dax_ref_cm2_min": "Dax_cm2_min"}
    for alias, key in aliases.items():
        records[alias] = parameter_origin(reference_payload, key)
    for key in ("T_ref_C", "T_sim_C", "deltaH_ads_kJ_mol", "Ea_kLDF_kJ_mol", "D_m_ref_m2_s"):
        records[key] = parameter_origin(reference_payload, key)
    formulas = {
        "KH_used_L_g": ("KH_ref_L_g * exp(-deltaH_ads_kJ_mol*1000/R * (1/T_sim_K - 1/T_ref_K))", ("KH_ref_L_g", "deltaH_ads_kJ_mol", "T_ref_C", "T_sim_C")),
        "kLDF_used_1_min": ("kLDF_ref_1_min * exp(-Ea_kLDF_kJ_mol*1000/R * (1/T_sim_K - 1/T_ref_K))", ("kLDF_ref_1_min", "Ea_kLDF_kJ_mol", "T_ref_C", "T_sim_C")),
        "D_m_used_m2_s": ("D_m_ref_m2_s * (T_sim_K/T_ref_K) * (mu_ref/mu_T)", ("D_m_ref_m2_s", "T_ref_C", "T_sim_C")),
    }
    for key, (formula, dependencies) in formulas.items():
        records[key] = derived_record(formula, dependencies, records, method="temperature_adaptation")
    mode = adaptation.get("Dax_temperature_mode")
    if mode == "no_correction":
        formula, dependencies = "Dax_ref_cm2_min", ("Dax_ref_cm2_min",)
    elif mode == "engineering_estimate" and reference_payload.get("engineering_estimate_available"):
        formula, dependencies = "(0.5 * superficial_velocity_m_s * dp_m + D_m_used_m2_s) * 600000", ("superficial_velocity_m_s", "dp_m", "D_m_used_m2_s")
    else:
        formula, dependencies = "Dax_ref_cm2_min * D_m_used_m2_s / max(D_m_ref_m2_s, 1e-30)", ("Dax_ref_cm2_min", "D_m_used_m2_s", "D_m_ref_m2_s")
    records["Dax_used_cm2_min"] = derived_record(formula, dependencies, records, method="temperature_adaptation")
    result["parameter_provenance"] = records
    return result


def sampled_parameter_payload(values, ranges, profile_payload, method, seed):
    """Dataset values are sampled from configured ranges, not experimental fits."""
    result = {key: values[key] for key in PARAMETER_UNITS if key in values}
    records = {}
    for key in result:
        if key not in ranges:
            records[key] = parameter_origin(profile_payload, key)
            continue
        records[key] = provenance_record("derived", method="range_sampling", sampling_method=method,
                                         configured_range=list(ranges[key]), random_seed=int(seed),
                                         dependencies=[f"configured_range.{key}"],
                                         starting_value_provenance=parameter_origin(profile_payload, key))
    result["parameter_provenance"] = records
    return result


def training_stat_origins(metadata, chosen_values, fallback_values):
    """Mirror metadata_range_value's exact existing median/min/max preference."""
    records = {}
    for key in chosen_values:
        if key not in fallback_values:
            continue
        ranges = (metadata or {}).get("feature_ranges", {}).get(key, {})
        if not isinstance(ranges, dict):
            ranges = {}
        chosen_stat = None
        for name in ("median", "min", "max"):
            try:
                if math.isfinite(float(ranges.get(name))):
                    chosen_stat = name
                    break
            except (TypeError, ValueError):
                pass
        records[key] = (provenance_record("derived", method="training_statistic", statistic=chosen_stat,
                         model_id=(metadata or {}).get("model_id"), dependencies=[f"feature_ranges.{key}.{chosen_stat}"],
                         underlying_parameter_provenance=(metadata or {}).get("sorption_parameter_provenance", {"origin": "unknown"}))
                        if chosen_stat else provenance_record("default", method="selection_fallback", default_value=fallback_values[key]))
    return records


def dataset_case_parameter_provenance(df, documented_by_case=None):
    """Read a capsule only from its own case; never infer it from numeric cells."""
    if "case_id" not in df:
        return {}
    documented_by_case = documented_by_case or {}
    result = {}
    for case_id, rows in df.groupby("case_id", sort=False):
        records = {}
        if "parameter_provenance_json" in rows:
            for value in rows["parameter_provenance_json"].dropna():
                parsed = parameter_origins({"parameter_provenance": value})
                if parsed:
                    records[json.dumps(parsed, sort_keys=True)] = parsed
        if len(records) == 1:
            item = {"status": "recorded", "parameter_provenance": next(iter(records.values()))}
        elif len(records) > 1:
            item = {"status": "unknown", "reason": "conflicting_case_records", "parameter_provenance": {}}
        else:
            documented = parameter_origins({"parameter_provenance": documented_by_case.get(case_id, {})})
            item = ({"status": "documented_recovery", "parameter_provenance": documented}
                    if documented else {"status": "unknown", "reason": "missing_or_empty_case_record", "parameter_provenance": {}})
        result[case_id] = item
    return result


def dataset_parameter_provenance(df):
    """Keep per-case metadata separate from features; old datasets stay unknown."""
    if "parameter_provenance_json" not in df:
        return {"status": "unknown", "reason": "legacy_dataset_without_component_provenance"}
    cases = dataset_case_parameter_provenance(df)
    recipes = {}
    unknown = []
    for case_id, item in cases.items():
        origin = item["parameter_provenance"]
        if not origin:
            unknown.append(str(case_id))
            continue
        key = json.dumps(origin, sort_keys=True)
        recipes.setdefault(key, {"parameter_provenance": origin, "number_of_cases": 0, "case_ids": []})
        recipes[key]["number_of_cases"] += 1
        recipes[key]["case_ids"].append(str(case_id))
    status = "unknown" if not recipes else "partial" if unknown else "recorded_per_case"
    return {"status": status, "recipes": list(recipes.values()),
            "unknown_case_ids": unknown, "number_of_unknown_cases": len(unknown),
            "per_case_location": "dataset.parameter_provenance_json"}
