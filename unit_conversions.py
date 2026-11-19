# -*- coding: utf-8 -*-
from __future__ import annotations

import math

from model import ColumnParams

try:
    import pint
except Exception:  # pragma: no cover - optional dependency
    pint = None


_UREG = pint.UnitRegistry() if pint is not None else None


def m_to_cm(value: float) -> float:
    return float(value) * 100.0


def m2_to_cm2(value: float) -> float:
    return float(value) * 10_000.0


def m3_h_to_ml_min(value: float) -> float:
    return float(value) * 1_000_000.0 / 60.0


def kg_to_g(value: float) -> float:
    return float(value) * 1000.0


def kg_m3_to_g_cm3(value: float) -> float:
    return float(value) / 1000.0


def m2_s_to_cm2_min(value: float) -> float:
    return float(value) * 10_000.0 * 60.0


def cm2_min_to_m2_s(value: float) -> float:
    return float(value) / (10_000.0 * 60.0)


def _close(a: float, b: float, rel_tol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=rel_tol)


def check_flow_conversion() -> bool:
    if _UREG is None:
        return False
    pint_value = (1.0 * _UREG.meter**3 / _UREG.hour).to(_UREG.milliliter / _UREG.minute).magnitude
    return _close(m3_h_to_ml_min(1.0), pint_value)


def check_length_conversion() -> bool:
    if _UREG is None:
        return False
    pint_value = (1.0 * _UREG.meter).to(_UREG.centimeter).magnitude
    return _close(m_to_cm(1.0), pint_value)


def check_area_conversion() -> bool:
    if _UREG is None:
        return False
    pint_value = (1.0 * _UREG.meter**2).to(_UREG.centimeter**2).magnitude
    return _close(m2_to_cm2(1.0), pint_value)


def check_density_conversion() -> bool:
    if _UREG is None:
        return False
    pint_value = (1.0 * _UREG.kilogram / _UREG.meter**3).to(_UREG.gram / _UREG.centimeter**3).magnitude
    return _close(kg_m3_to_g_cm3(1.0), pint_value)


def check_diffusivity_conversion() -> bool:
    if _UREG is None:
        return False
    pint_value = (1.0 * _UREG.meter**2 / _UREG.second).to(_UREG.centimeter**2 / _UREG.minute).magnitude
    return _close(m2_s_to_cm2_min(1.0), pint_value)


def verify_unit_conversions() -> dict:
    warnings: list[str] = []
    if _UREG is None:
        warnings.append("Pint не установлен. Используются встроенные функции перевода единиц.")
        return {
            "pint_available": False,
            "m3_h_to_ml_min_ok": None,
            "kg_m3_to_g_cm3_ok": None,
            "m2_s_to_cm2_min_ok": None,
            "all_ok": None,
            "warnings": warnings,
        }

    checks = {
        "length_ok": check_length_conversion(),
        "area_ok": check_area_conversion(),
        "m3_h_to_ml_min_ok": check_flow_conversion(),
        "kg_m3_to_g_cm3_ok": check_density_conversion(),
        "m2_s_to_cm2_min_ok": check_diffusivity_conversion(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        warnings.append("Проверка единиц через Pint выявила расхождения: " + ", ".join(failed))
    return {
        "pint_available": True,
        **checks,
        "all_ok": not failed,
        "warnings": warnings,
    }


def industrial_to_column_params(
    Q_m3_h: float,
    D_col_m: float,
    H_m: float,
    bulk_density_kg_m3: float,
    epsilon: float,
    C0_mg_L: float,
) -> ColumnParams:
    area_m2 = math.pi * float(D_col_m) ** 2 / 4.0
    bed_volume_m3 = area_m2 * float(H_m)
    carbon_mass_kg = bed_volume_m3 * float(bulk_density_kg_m3)
    return ColumnParams(
        length_cm=m_to_cm(H_m),
        area_cm2=m2_to_cm2(area_m2),
        mass_g=kg_to_g(carbon_mass_kg),
        epsilon=float(epsilon),
        flow_ml_min=m3_h_to_ml_min(Q_m3_h),
        c0_mg_l=float(C0_mg_L),
    )
