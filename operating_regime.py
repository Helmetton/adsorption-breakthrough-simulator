# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd

from encoding_utils import safe_read_csv


LABORATORY_SCALE_LABELS = {"Лабораторный"}


def is_laboratory_scale(scale_mode: str) -> bool:
    text = str(scale_mode).lower()
    return (
        str(scale_mode) in LABORATORY_SCALE_LABELS
        or "labor" in text
        or "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440" in text
    )


def load_calibration_ranges(
    fitted_path: str = "fitted_experiment_parameters.csv",
    cleaned_path: str = "cleaned_dynamic_sorption_data.csv",
) -> dict[str, tuple[float, float]]:
    source_path = fitted_path if os.path.exists(fitted_path) else cleaned_path
    if not os.path.exists(source_path):
        return {}

    df = safe_read_csv(source_path)
    ranges: dict[str, tuple[float, float]] = {}

    def add_range(output_key: str, candidates: list[str]) -> None:
        for col in candidates:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if not values.empty:
                    ranges[output_key] = (float(values.min()), float(values.max()))
                    return

    add_range("flow_ml_min", ["flow_ml_min", "flow_ml_min_assumed"])
    add_range("bed_height_cm", ["bed_height_cm"])
    add_range("C0_mg_L", ["C0_mg_L"])
    add_range("temperature_C", ["temperature_C"])
    if "column_equivalent_diameter_cm" in df.columns:
        add_range("column_diameter_cm", ["column_equivalent_diameter_cm"])
    elif "column_area_cm2" in df.columns:
        areas = pd.to_numeric(df["column_area_cm2"], errors="coerce").dropna()
        if not areas.empty:
            diameters = np.sqrt(4.0 * areas / math.pi)
            ranges["column_diameter_cm"] = (float(diameters.min()), float(diameters.max()))
    return ranges


def _range_warning(
    label: str,
    value: float,
    range_value: tuple[float, float] | None,
    strong_factor: float = 2.0,
) -> tuple[list[str], str | None]:
    if range_value is None:
        return [], None
    lo, hi = range_value
    tolerance = 0.01 * max(abs(lo), abs(hi), 1e-12)
    if lo - tolerance <= value <= hi + tolerance:
        return [], None
    if value > hi * strong_factor or value < lo / max(strong_factor, 1e-12):
        return [f"{label} сильно выходит за диапазон лабораторной калибровки ({lo:.4g}–{hi:.4g})."], "strong_extrapolation"
    return [f"{label} находится вне диапазона лабораторной калибровки ({lo:.4g}–{hi:.4g})."], "moderate_extrapolation"


def check_operating_regime(
    scale_mode: str,
    Q: float,
    H: float,
    D: float,
    epsilon: float,
    C0: float,
    C_breakthrough: float,
    bulk_density: float,
    calibration_ranges: dict[str, tuple[float, float]] | None = None,
    temperature_C: float | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    info: list[str] = []
    calibration_ranges = calibration_ranges or {}
    lab = is_laboratory_scale(scale_mode)

    if lab:
        area = math.pi * float(D) ** 2 / 4.0
        bed_volume = area * float(H)
        void_volume = float(epsilon) * bed_volume
        EBCT_min = bed_volume / max(float(Q), 1e-30)
        void_residence_time_min = void_volume / max(float(Q), 1e-30)
        superficial_velocity = float(Q) / max(area, 1e-30)
        interstitial_velocity = superficial_velocity / max(float(epsilon), 1e-30)
        bed_volumes_per_min = float(Q) / max(bed_volume, 1e-30)
        estimated_carbon_mass = float(bulk_density) * bed_volume
        dye_loading_mg_min = float(C0) * float(Q) / 1000.0
        result = {
            "column_area_cm2": area,
            "bed_volume_ml": bed_volume,
            "void_volume_ml": void_volume,
            "EBCT_min": EBCT_min,
            "void_residence_time_min": void_residence_time_min,
            "superficial_velocity_cm_min": superficial_velocity,
            "interstitial_velocity_cm_min": interstitial_velocity,
            "bed_volumes_per_min": bed_volumes_per_min,
            "bed_volumes_per_hour": bed_volumes_per_min * 60.0,
            "estimated_carbon_mass_g": estimated_carbon_mass,
            "dye_loading_mg_min": dye_loading_mg_min,
            "recommended_Q_for_EBCT_10_min": bed_volume / 10.0,
            "recommended_Q_for_EBCT_20_min": bed_volume / 20.0,
            "required_bed_volume_for_current_Q_EBCT_20_min": float(Q) * 20.0,
        }
    else:
        Q_m3_min = float(Q) / 60.0
        area = math.pi * float(D) ** 2 / 4.0
        bed_volume = area * float(H)
        void_volume = float(epsilon) * bed_volume
        EBCT_min = bed_volume / max(Q_m3_min, 1e-30)
        void_residence_time_min = void_volume / max(Q_m3_min, 1e-30)
        superficial_velocity_m_h = float(Q) / max(area, 1e-30)
        interstitial_velocity_m_h = superficial_velocity_m_h / max(float(epsilon), 1e-30)
        bed_volumes_per_hour = float(Q) / max(bed_volume, 1e-30)
        estimated_carbon_mass_kg = float(bulk_density) * bed_volume
        dye_loading_g_h = float(C0) * float(Q)
        result = {
            "column_area_m2": area,
            "bed_volume_m3": bed_volume,
            "void_volume_m3": void_volume,
            "EBCT_min": EBCT_min,
            "void_residence_time_min": void_residence_time_min,
            "superficial_velocity_m_h": superficial_velocity_m_h,
            "interstitial_velocity_m_h": interstitial_velocity_m_h,
            "bed_volumes_per_min": bed_volumes_per_hour / 60.0,
            "bed_volumes_per_hour": bed_volumes_per_hour,
            "estimated_carbon_mass_kg": estimated_carbon_mass_kg,
            "dye_loading_g_h": dye_loading_g_h,
            "recommended_Q_for_EBCT_10_min": bed_volume / (10.0 / 60.0),
            "recommended_Q_for_EBCT_20_min": bed_volume / (20.0 / 60.0),
            "required_bed_volume_for_current_Q_EBCT_20_min": float(Q) * (20.0 / 60.0),
        }

    if EBCT_min < 1.0:
        warnings.append(
            "EBCT < 1 мин: раствор проходит через слой почти мгновенно. Для сорбционной очистки такой режим обычно слишком быстрый. Прорыв может наступить практически сразу."
        )
    elif EBCT_min < 5.0:
        warnings.append("EBCT < 5 мин: очень малое время контакта. Результат следует считать экстремальным режимом.")
    elif EBCT_min <= 30.0:
        info.append("EBCT находится в рабочем инженерном диапазоне для предварительных расчётов.")
    elif EBCT_min > 60.0:
        warnings.append("EBCT > 60 мин: очень большое время контакта. Расчёт может быть долгим, а датасет большим.")

    level = "validated_range"
    def raise_level(new_level: str | None) -> None:
        nonlocal level
        order = {"validated_range": 0, "moderate_extrapolation": 1, "strong_extrapolation": 2}
        if new_level is not None and order[new_level] > order[level]:
            level = new_level

    if lab:
        range_checks = [
            ("Введённый расход Q", float(Q), calibration_ranges.get("flow_ml_min")),
            ("Высота слоя H", float(H), calibration_ranges.get("bed_height_cm")),
            ("Диаметр колонки D", float(D), calibration_ranges.get("column_diameter_cm")),
            ("C0", float(C0), calibration_ranges.get("C0_mg_L")),
        ]
        if temperature_C is not None:
            range_checks.append(("Температура", float(temperature_C), calibration_ranges.get("temperature_C")))
        for label, value, rng in range_checks:
            msgs, new_level = _range_warning(label, value, rng)
            warnings.extend(msgs)
            raise_level(new_level)

        flow_range = calibration_ranges.get("flow_ml_min")
        if flow_range is not None and float(Q) > flow_range[1] * 2.0:
            warnings.append(
                "Введённый расход Q значительно превышает диапазон лабораторной калибровки. Результат является экстраполяцией."
            )
            raise_level("strong_extrapolation")

        if float(Q) >= 100.0 and EBCT_min < 1.0:
            warnings.append(
                f"Q = {float(Q):.3g} мл/мин для лабораторной колонки H = {float(H):.3g} см и D = {float(D):.3g} см даёт EBCT меньше 1 минуты. Это не режим, близкий к экспериментам."
            )
            raise_level("strong_extrapolation")

    if float(C0) > 10.0:
        warnings.append("C0 > 10 мг/л: линейная изотерма Генри может быть недостаточной. Расчёт является экстраполяцией.")
        raise_level("moderate_extrapolation")

    if EBCT_min < 1.0:
        raise_level("strong_extrapolation")
    elif EBCT_min < 5.0:
        raise_level("moderate_extrapolation")

    result.update(
        {
            "warnings": warnings,
            "info": info,
            "regime_warning": " | ".join(warnings),
            "extrapolation_level": level,
            "C_breakthrough": float(C_breakthrough),
        }
    )
    return result
