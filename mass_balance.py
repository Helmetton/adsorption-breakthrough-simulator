# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from model import ColumnParams, SorptionParams


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    if len(y) <= 1:
        return out
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out[1:] = np.cumsum(increments)
    return out


def calculate_mass_balance(
    time_min,
    C_profile_mg_L,
    q_profile_mg_g,
    C_out_mg_L,
    column: ColumnParams,
    params: SorptionParams,
) -> pd.DataFrame:
    time = np.asarray(time_min, dtype=float)
    C_profile = np.asarray(C_profile_mg_L, dtype=float)
    q_profile = np.asarray(q_profile_mg_g, dtype=float)
    C_out = np.asarray(C_out_mg_L, dtype=float)
    if len(time) == 0 or np.any(np.diff(time) <= 0):
        raise ValueError("Mass balance requires strictly increasing observation times.")

    if C_profile.ndim != 2 or q_profile.ndim != 2:
        raise ValueError("C_profile_mg_L и q_profile_mg_g должны быть двумерными массивами.")
    if C_profile.shape != q_profile.shape:
        raise ValueError("C_profile_mg_L и q_profile_mg_g должны иметь одинаковую форму.")
    if C_profile.shape[0] != len(time):
        raise ValueError("Число временных точек профилей должно совпадать с time_min.")

    n_cells = C_profile.shape[1]
    Q_L_min = column.flow_ml_min / 1000.0
    V_bed_L = column.area_cm2 * column.length_cm / 1000.0
    V_liquid_cell_L = column.epsilon * V_bed_L / n_cells
    mass_carbon_cell_g = column.mass_g / n_cells

    inlet_rate_mg_min = np.full_like(time, column.c0_mg_l * Q_L_min, dtype=float)
    outlet_rate_mg_min = C_out * Q_L_min

    M_in_mg = _cumulative_trapezoid(inlet_rate_mg_min, time)
    M_out_mg = _cumulative_trapezoid(outlet_rate_mg_min, time)
    M_liquid_mg = np.sum(C_profile * V_liquid_cell_L, axis=1)
    M_adsorbed_mg = np.sum(q_profile * mass_carbon_cell_g, axis=1)

    # Balance the measured interval, including a nonzero first returned time.
    M_initial_liquid_mg = M_liquid_mg[0]
    M_initial_adsorbed_mg = M_adsorbed_mg[0]
    M_initial_total_mg = M_initial_liquid_mg + M_initial_adsorbed_mg

    mass_balance_error_mg = (
        M_initial_total_mg
        + M_in_mg
        - M_out_mg
        - M_liquid_mg
        - M_adsorbed_mg
    )
    denominator = np.abs(M_initial_total_mg + M_in_mg)
    mass_balance_error_percent = np.full_like(time, np.nan)
    np.divide(100.0 * mass_balance_error_mg, denominator,
              out=mass_balance_error_percent, where=denominator > 1e-10)

    return pd.DataFrame(
        {
            "time_min": time,
            "M_in_mg": M_in_mg,
            "M_out_mg": M_out_mg,
            "M_liquid_mg": M_liquid_mg,
            "M_adsorbed_mg": M_adsorbed_mg,
            "M_initial_total_mg": np.full_like(time, M_initial_total_mg, dtype=float),
            "mass_balance_error_mg": mass_balance_error_mg,
            "mass_balance_error_percent": mass_balance_error_percent,
            "balance_error_mg": mass_balance_error_mg,
            "balance_error_percent": mass_balance_error_percent,
        }
    )


def summarize_mass_balance(balance_df: pd.DataFrame) -> dict:
    if balance_df.empty:
        return {
            "final_balance_error_mg": None,
            "final_balance_error_percent": None,
            "max_abs_balance_error_percent": None,
            "mass_balance_ok": False,
        }

    final = balance_df.iloc[-1]
    max_abs = float(balance_df["mass_balance_error_percent"].abs().max())
    return {
        "final_balance_error_mg": float(final["mass_balance_error_mg"]),
        "final_balance_error_percent": float(final["mass_balance_error_percent"]),
        "max_abs_balance_error_percent": max_abs,
        "mass_balance_ok": bool(max_abs <= 5.0),
    }
