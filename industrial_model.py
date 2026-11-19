# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any

import numpy as np

from unit_conversions import cm2_min_to_m2_s, m2_s_to_cm2_min


R_GAS_J_MOL_K = 8.314


try:
    from CoolProp.CoolProp import PropsSI
except Exception:  # pragma: no cover - optional dependency fallback
    PropsSI = None


def water_properties(T_C: float, P_bar: float = 1.01325) -> dict[str, float | str]:
    T_K = float(T_C) + 273.15
    P_Pa = float(P_bar) * 100_000.0
    if PropsSI is not None:
        try:
            return {
                "rho_water_kg_m3": float(PropsSI("D", "T", T_K, "P", P_Pa, "Water")),
                "mu_water_Pa_s": float(PropsSI("V", "T", T_K, "P", P_Pa, "Water")),
                "source": "CoolProp",
                "P_bar": float(P_bar),
                "note": "",
            }
        except Exception:
            pass

    temp_table = np.array([5, 10, 15, 20, 25, 30, 40, 50, 60], dtype=float)
    rho_table = np.array([999.96, 999.70, 999.10, 998.21, 997.05, 995.65, 992.22, 988.03, 983.20])
    mu_table = np.array([1.519e-3, 1.307e-3, 1.139e-3, 1.002e-3, 0.890e-3, 0.797e-3, 0.653e-3, 0.547e-3, 0.467e-3])
    T_clip = float(np.clip(T_C, temp_table.min(), temp_table.max()))
    return {
        "rho_water_kg_m3": float(np.interp(T_clip, temp_table, rho_table)),
        "mu_water_Pa_s": float(np.interp(T_clip, temp_table, mu_table)),
        "source": "fallback",
        "P_bar": float(P_bar),
        "note": "fallback не учитывает влияние давления на свойства воды",
    }


def henry_temperature_correction(
    KH_ref: float,
    T_ref_C: float,
    T_C: float,
    deltaH_ads_kJ_mol: float,
) -> float:
    """Van't Hoff style correction. For exothermic adsorption deltaH_ads < 0,
    increasing temperature usually decreases KH.
    """
    T_ref_K = float(T_ref_C) + 273.15
    T_K = float(T_C) + 273.15
    deltaH_ads_J_mol = float(deltaH_ads_kJ_mol) * 1000.0
    return float(KH_ref) * math.exp(-(deltaH_ads_J_mol / R_GAS_J_MOL_K) * (1.0 / T_K - 1.0 / T_ref_K))


def arrhenius_correction(
    value_ref: float,
    T_ref_C: float,
    T_C: float,
    Ea_kJ_mol: float,
) -> float:
    T_ref_K = float(T_ref_C) + 273.15
    T_K = float(T_C) + 273.15
    Ea_J_mol = float(Ea_kJ_mol) * 1000.0
    return float(value_ref) * math.exp(-(Ea_J_mol / R_GAS_J_MOL_K) * (1.0 / T_K - 1.0 / T_ref_K))


def henry_pressure_correction(
    KH_ref: float,
    T_C: float,
    P_bar: float,
    P_ref_bar: float,
    deltaV_ads_cm3_mol: float,
) -> float:
    """Optional experimental correction. Disabled by default in the app."""
    T_K = float(T_C) + 273.15
    deltaV_ads_m3_mol = float(deltaV_ads_cm3_mol) * 1e-6
    deltaP_Pa = (float(P_bar) - float(P_ref_bar)) * 100_000.0
    return float(KH_ref) * math.exp(-(deltaV_ads_m3_mol / (R_GAS_J_MOL_K * T_K)) * deltaP_Pa)


def molecular_diffusion_temperature_correction(
    D_ref_m2_s: float,
    T_ref_C: float,
    T_C: float,
    mu_ref_Pa_s: float,
    mu_T_Pa_s: float,
) -> float:
    T_ref_K = float(T_ref_C) + 273.15
    T_K = float(T_C) + 273.15
    return float(D_ref_m2_s) * (T_K / T_ref_K) * (float(mu_ref_Pa_s) / float(mu_T_Pa_s))


def packed_bed_hydrodynamics(
    Q_m3_s: float,
    D_col_m: float,
    dp_m: float,
    epsilon: float,
    T_C: float,
    D_m_m2_s: float,
    P_bar: float = 1.01325,
) -> dict[str, Any]:
    props = water_properties(T_C, P_bar=P_bar)
    rho = float(props["rho_water_kg_m3"])
    mu = float(props["mu_water_Pa_s"])
    A_m2 = math.pi * float(D_col_m) ** 2 / 4.0
    superficial_velocity_m_s = float(Q_m3_s) / A_m2
    interstitial_velocity_m_s = superficial_velocity_m_s / float(epsilon)
    Re_p = rho * superficial_velocity_m_s * float(dp_m) / mu
    Sc = mu / (rho * float(D_m_m2_s))
    Sh = 2.0 + 1.1 * (max(Re_p, 0.0) ** 0.6) * (max(Sc, 0.0) ** (1.0 / 3.0))
    k_f_m_s = Sh * float(D_m_m2_s) / float(dp_m)
    Pe_p = interstitial_velocity_m_s * float(dp_m) / float(D_m_m2_s)
    warnings: list[str] = []
    if Re_p < 3.0 or Re_p > 10_000:
        warnings.append("Re_p вне области применимости корреляции Wakao-Funazkri.")
    return {
        "A_m2": A_m2,
        "superficial_velocity_m_s": superficial_velocity_m_s,
        "superficial_velocity_m_h": superficial_velocity_m_s * 3600.0,
        "interstitial_velocity_m_s": interstitial_velocity_m_s,
        "interstitial_velocity_m_h": interstitial_velocity_m_s * 3600.0,
        "Re_p": Re_p,
        "Sc": Sc,
        "Sh": Sh,
        "k_f_m_s": k_f_m_s,
        "Pe_p": Pe_p,
        "warnings": warnings,
        **props,
    }


def packed_bed_dimensionless_numbers(
    Q_m3_s: float,
    D_col_m: float,
    dp_m: float,
    epsilon: float,
    T_C: float,
    D_m_m2_s: float,
    P_bar: float = 1.01325,
) -> dict[str, Any]:
    return packed_bed_hydrodynamics(Q_m3_s, D_col_m, dp_m, epsilon, T_C, D_m_m2_s, P_bar=P_bar)


def ergun_pressure_drop(
    Q_m3_s: float,
    D_col_m: float,
    H_m: float,
    dp_m: float,
    epsilon: float,
    T_C: float,
    particle_sphericity: float = 1.0,
    P_bar: float = 1.01325,
) -> dict[str, float]:
    props = water_properties(T_C, P_bar=P_bar)
    rho = float(props["rho_water_kg_m3"])
    mu = float(props["mu_water_Pa_s"])
    A_m2 = math.pi * float(D_col_m) ** 2 / 4.0
    v_s = float(Q_m3_s) / A_m2
    eps = float(epsilon)
    phi = float(particle_sphericity)
    viscous = 150.0 * mu * (1.0 - eps) ** 2 / (float(dp_m) ** 2 * eps**3 * phi**2) * v_s
    inertial = 1.75 * rho * (1.0 - eps) / (float(dp_m) * eps**3 * phi) * v_s**2
    deltaP_per_m = viscous + inertial
    deltaP = deltaP_per_m * float(H_m)
    return {
        "DeltaP_Pa": deltaP,
        "DeltaP_bar": deltaP / 100_000.0,
        "DeltaP_per_m_Pa_m": deltaP_per_m,
    }


def ebct(Q_m3_s: float, D_col_m: float, H_m: float) -> dict[str, float]:
    A_m2 = math.pi * float(D_col_m) ** 2 / 4.0
    bed_volume_m3 = A_m2 * float(H_m)
    EBCT_s = bed_volume_m3 / float(Q_m3_s)
    return {
        "bed_volume_m3": bed_volume_m3,
        "EBCT_s": EBCT_s,
        "EBCT_min": EBCT_s / 60.0,
    }


def validate_industrial_inputs(
    Q_m3_h: float,
    C0_mg_L: float,
    C_breakthrough_mg_L: float,
    T_C: float,
    D_col_m: float,
    H_m: float,
    dp_mm: float,
    epsilon: float,
    bulk_density_kg_m3: float,
    particle_sphericity: float,
) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if Q_m3_h <= 0:
        errors.append("Q должен быть больше 0.")
    if D_col_m <= 0:
        errors.append("Диаметр колонки должен быть больше 0.")
    if H_m <= 0:
        errors.append("Высота слоя должна быть больше 0.")
    if dp_mm <= 0:
        errors.append("Размер частиц dp должен быть больше 0.")
    if not (0.2 < epsilon < 0.7):
        errors.append("Порозность epsilon должна быть в диапазоне 0.2–0.7.")
    if bulk_density_kg_m3 <= 0:
        errors.append("Насыпная плотность должна быть больше 0.")
    if C0_mg_L <= 0:
        errors.append("C0 должен быть больше 0.")
    if C_breakthrough_mg_L >= C0_mg_L:
        errors.append("C прорыва должен быть меньше C0.")
    if not (5.0 <= T_C <= 60.0):
        warnings.append("T_C выходит за разумный диапазон 5–60 °C.")
    if not (0.5 <= particle_sphericity <= 1.0):
        errors.append("Сферичность частиц должна быть в диапазоне 0.5–1.0.")
    return {"errors": errors, "warnings": warnings}


def industrial_adsorber_diagnostics(
    Q_m3_h: float,
    D_col_m: float,
    H_m: float,
    dp_mm: float,
    epsilon: float,
    bulk_density_kg_m3: float,
    T_C: float,
    D_m_ref_m2_s: float,
    T_ref_C: float,
    KH_ref_L_g: float,
    kLDF_ref_1_min: float,
    Dax_ref_cm2_min: float,
    deltaH_ads_kJ_mol: float,
    Ea_kLDF_kJ_mol: float,
    Dax_mode: str,
    kLDF_mode: str = "из калибровки",
    particle_sphericity: float = 1.0,
    C0_mg_L: float | None = None,
    pressure_mode: str = "Атмосферное",
    P_in_bar: float = 1.01325,
    P_out_bar: float = 1.01325,
    deltaP_available_bar: float | None = None,
    apply_pressure_KH_correction: bool = False,
    deltaV_ads_cm3_mol: float = 0.0,
    P_ref_bar: float = 1.01325,
) -> dict[str, Any]:
    Q_m3_s = float(Q_m3_h) / 3600.0
    dp_m = float(dp_mm) / 1000.0
    if pressure_mode == "Задать давление на входе":
        P_in_bar = float(P_in_bar)
        P_out_bar = float(P_out_bar)
        available_deltaP_bar = max(P_in_bar - P_out_bar, 0.0)
        properties_pressure_bar = max((P_in_bar + P_out_bar) / 2.0, 0.001)
    elif pressure_mode == "Задать перепад давления":
        P_out_bar = float(P_out_bar)
        available_deltaP_bar = max(float(deltaP_available_bar or 0.0), 0.0)
        P_in_bar = P_out_bar + available_deltaP_bar
        properties_pressure_bar = max((P_in_bar + P_out_bar) / 2.0, 0.001)
    else:
        pressure_mode = "Атмосферное"
        P_in_bar = 1.01325
        P_out_bar = 1.01325
        available_deltaP_bar = None
        properties_pressure_bar = 1.01325

    mu_ref = float(water_properties(T_ref_C, P_bar=P_ref_bar)["mu_water_Pa_s"])
    mu_T = float(water_properties(T_C, P_bar=properties_pressure_bar)["mu_water_Pa_s"])
    D_m_T = molecular_diffusion_temperature_correction(D_m_ref_m2_s, T_ref_C, T_C, mu_ref, mu_T)
    hyd = packed_bed_hydrodynamics(Q_m3_s, D_col_m, dp_m, epsilon, T_C, D_m_T, P_bar=properties_pressure_bar)
    pressure = ergun_pressure_drop(
        Q_m3_s,
        D_col_m,
        H_m,
        dp_m,
        epsilon,
        T_C,
        particle_sphericity=particle_sphericity,
        P_bar=properties_pressure_bar,
    )
    ebct_data = ebct(Q_m3_s, D_col_m, H_m)
    bed_volume_m3 = ebct_data["bed_volume_m3"]
    KH_T = henry_temperature_correction(KH_ref_L_g, T_ref_C, T_C, deltaH_ads_kJ_mol)
    KH_pressure_correction_enabled = bool(apply_pressure_KH_correction)
    if KH_pressure_correction_enabled:
        KH_T = henry_pressure_correction(KH_T, T_C, properties_pressure_bar, P_ref_bar, deltaV_ads_cm3_mol)
    kLDF_arrhenius = arrhenius_correction(kLDF_ref_1_min, T_ref_C, T_C, Ea_kLDF_kJ_mol)
    R_p_m = dp_m / 2.0
    kLDF_film = 3.0 * hyd["k_f_m_s"] / R_p_m * 60.0

    if kLDF_mode == "температурная поправка Arrhenius":
        kLDF_used = kLDF_arrhenius
    elif kLDF_mode == "оценка через внешний массообмен k_f":
        kLDF_used = kLDF_film
    elif kLDF_mode == "комбинированная оценка":
        kLDF_used = min(kLDF_arrhenius, kLDF_film)
    else:
        kLDF_mode = "из калибровки"
        kLDF_used = float(kLDF_ref_1_min)

    if Dax_mode == "температурная поправка к D_ax_ref":
        Dax_T = m2_s_to_cm2_min(
            molecular_diffusion_temperature_correction(
                cm2_min_to_m2_s(Dax_ref_cm2_min),
                T_ref_C,
                T_C,
                mu_ref,
                mu_T,
            )
        )
    elif Dax_mode == "инженерная оценка через скорость и размер частиц":
        dispersion_m2_s = 0.5 * hyd["superficial_velocity_m_s"] * dp_m + D_m_T
        Dax_T = m2_s_to_cm2_min(dispersion_m2_s)
    else:
        Dax_T = float(Dax_ref_cm2_min)

    Dax_m2_s = cm2_min_to_m2_s(Dax_T)
    if Dax_m2_s > 0:
        Pe_L_superficial = hyd["superficial_velocity_m_s"] * float(H_m) / Dax_m2_s
        Pe_L_interstitial = hyd["interstitial_velocity_m_s"] * float(H_m) / Dax_m2_s
    else:
        Pe_L_superficial = float("inf")
        Pe_L_interstitial = float("inf")

    warnings = list(hyd["warnings"])
    if ebct_data["EBCT_min"] < 5.0:
        warnings.append("EBCT < 5 мин: слишком малое время контакта.")
    if ebct_data["EBCT_min"] > 60.0:
        warnings.append("EBCT > 60 мин: очень большое время контакта.")
    if pressure["DeltaP_bar"] > 2.0:
        warnings.append("pressure_drop_bar > 2 бар: высокая потеря давления.")
    pressure_warning = ""
    if available_deltaP_bar is not None and pressure["DeltaP_bar"] > available_deltaP_bar:
        pressure_warning = "Заданный расход Q не обеспечивается выбранным перепадом давления."
        warnings.append(pressure_warning)
    if T_C > 40.0:
        warnings.append("T_C > 40 °C: высокая температура для текущей валидации.")
    if not (5.0 <= T_C <= 60.0):
        warnings.append("T_C выходит за диапазон 5–60 °C.")
    if C0_mg_L is not None and C0_mg_L > 10.0:
        warnings.append("C0 > 10 мг/л: Henry isotherm может быть недостаточной.")
    if kLDF_mode in {"оценка через внешний массообмен k_f", "комбинированная оценка"}:
        warnings.append(
            "Оценка k_LDF через k_f является приближённой и требует проверки по экспериментальным данным."
        )
    ratio = max(kLDF_film, float(kLDF_ref_1_min)) / max(
        min(kLDF_film, float(kLDF_ref_1_min)),
        1e-30,
    )
    if ratio > 10.0:
        warnings.append("kLDF_film_1_min отличается от kLDF_ref_1_min больше чем в 10 раз.")
    if KH_pressure_correction_enabled:
        warnings.append("Экспериментальная поправка K_H по давлению требует обоснованного ΔV_ads. Не используйте без экспериментального подтверждения.")
    if hyd.get("note"):
        warnings.append(str(hyd["note"]))
    warnings.append("industrial mode: результаты являются инженерной экстраполяцией.")

    return {
        "bed_volume_m3": bed_volume_m3,
        "carbon_mass_kg": bed_volume_m3 * float(bulk_density_kg_m3),
        "EBCT_s": ebct_data["EBCT_s"],
        "EBCT_min": ebct_data["EBCT_min"],
        "superficial_velocity_m_s": hyd["superficial_velocity_m_s"],
        "superficial_velocity_m_h": hyd["superficial_velocity_m_h"],
        "interstitial_velocity_m_s": hyd["interstitial_velocity_m_s"],
        "interstitial_velocity_m_h": hyd["interstitial_velocity_m_h"],
        "Re_p": hyd["Re_p"],
        "Sc": hyd["Sc"],
        "Sh": hyd["Sh"],
        "Pe_p": hyd["Pe_p"],
        "Pe_L_superficial": Pe_L_superficial,
        "Pe_L_interstitial": Pe_L_interstitial,
        "k_f_m_s": hyd["k_f_m_s"],
        "pressure_drop_bar": pressure["DeltaP_bar"],
        "required_deltaP_bar": pressure["DeltaP_bar"],
        "available_deltaP_bar": available_deltaP_bar,
        "pressure_feasible": True if available_deltaP_bar is None else pressure["DeltaP_bar"] <= available_deltaP_bar,
        "pressure_mode": pressure_mode,
        "pressure_warning": pressure_warning,
        "P_in_bar": P_in_bar,
        "P_out_bar": P_out_bar,
        "P_properties_bar": properties_pressure_bar,
        "pressure_drop_Pa": pressure["DeltaP_Pa"],
        "pressure_drop_per_m_Pa_m": pressure["DeltaP_per_m_Pa_m"],
        "rho_water_kg_m3": hyd["rho_water_kg_m3"],
        "mu_water_Pa_s": hyd["mu_water_Pa_s"],
        "water_properties_source": hyd["source"],
        "water_properties_note": hyd.get("note", ""),
        "D_m_T_m2_s": D_m_T,
        "KH_T_L_g": KH_T,
        "KH_pressure_correction_enabled": KH_pressure_correction_enabled,
        "deltaV_ads_cm3_mol": float(deltaV_ads_cm3_mol),
        "kLDF_ref_1_min": float(kLDF_ref_1_min),
        "kLDF_Arrhenius_1_min": kLDF_arrhenius,
        "kLDF_film_1_min": kLDF_film,
        "kLDF_used_1_min": kLDF_used,
        "kLDF_mode": kLDF_mode,
        "kLDF_T_1_min": kLDF_used,
        "Dax_T_cm2_min": Dax_T,
        "warnings": warnings,
    }
