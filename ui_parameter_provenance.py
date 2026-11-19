"""Display backend lineage, including unknown legacy origins, without guesses."""
import pandas as pd

from parameter_provenance import PARAMETER_UNITS, parameter_origin
from translations import tr
from display_units import display_volume_units


def origin_description(record, lang):
    origin = record.get("origin", "unknown")
    label = tr(lang, "provenance_origin_" + origin)
    method = record.get("method", "legacy_unknown")
    comment = tr(lang, "provenance_method_" + method)
    if comment == "provenance_method_" + method:
        comment = tr(lang, "provenance_method_recorded")
    reference = record.get("experiment_id") or record.get("profile_key") or record.get("reference")
    if reference:
        comment += " " + str(reference)
    if record.get("formula"):
        comment += " " + record["formula"]
    dependencies = record.get("dependency_provenance", {})
    defaults = [key for key, item in dependencies.items() if item.get("origin") == "default"]
    if defaults:
        comment += " " + tr(lang, "provenance_default_dependencies").format(parameters=", ".join(defaults))
    if record.get("statistic"):
        comment += " " + str(record["statistic"])
    return label, comment


def dependency_snapshot_description(record, params, lang, formatter):
    if record.get("origin") != "derived" or record.get("method") not in {"calculated", "profile_expression"}:
        return ""
    snapshot = record.get("dependency_values")
    if not isinstance(snapshot, dict):
        return tr(lang, "provenance_snapshot_unknown")
    used = ", ".join(f"{key}={formatter(value)}" for key, value in snapshot.items())
    current = ", ".join(f"{key}={formatter(params[key])}" for key in ("KH_L_g", "qmax_mg_g") if key in params)
    return tr(lang, "provenance_snapshot_values").format(used=used, current=current)


def parameter_usage_description(key, context, lang):
    if not context or context.get("solver") != "henry_ldf":
        return tr(lang, "provenance_usage_unspecified")
    roles = {
        "KH_L_g": "kh", "kLDF_1_min": "kldf", "Dax_cm2_min": "dax",
        "qmax_mg_g": "qmax", "b_L_mg": "b", "deltaH_ads_kJ_mol": "deltaH",
        "Ea_kLDF_kJ_mol": "ea", "D_m_ref_m2_s": "diffusion",
        "T_ref_C": "temperature", "T_sim_C": "temperature", "T_exp_C": "experimental_temperature",
    }
    role = roles.get(key, "unspecified")
    if key == "Dax_cm2_min" and context.get("Dax_temperature_mode") == "engineering_estimate":
        role = "dax_engineering"
    if key == "D_m_ref_m2_s" and context.get("Dax_temperature_mode") == "no_correction":
        role = "diffusion_no_dax"
    return tr(lang, "provenance_usage_" + role)


def parameter_source_table(params, lang, formatter=str, usage_context=None):
    names = {"KH_L_g": "K_H", "kLDF_1_min": "k_LDF", "Dax_cm2_min": "D_ax",
             "qmax_mg_g": "qmax", "b_L_mg": "b", "deltaH_ads_kJ_mol": "ΔH_ads",
             "Ea_kLDF_kJ_mol": "Ea_kLDF", "D_m_ref_m2_s": "D_m_ref",
             "T_ref_C": "T_ref", "T_exp_C": "T_exp", "T_sim_C": "T_sim"}
    rows = []
    for key, name in names.items():
        if params.get(key) is None:
            continue
        origin, comment = origin_description(parameter_origin(params, key), lang)
        snapshot = dependency_snapshot_description(parameter_origin(params, key), params, lang, formatter)
        if snapshot:
            comment += " " + snapshot
        rows.append({tr(lang, "col_parameter"): name, tr(lang, "col_value"): formatter(params[key]),
                     tr(lang, "col_unit"): display_volume_units(PARAMETER_UNITS[key]), tr(lang, "col_source"): origin,
                     tr(lang, "col_comment"): comment,
                     tr(lang, "provenance_col_usage"): parameter_usage_description(key, usage_context, lang)})
    return pd.DataFrame(rows)
