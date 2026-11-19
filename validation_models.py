# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any


try:
    from pydantic import BaseModel, Field, ValidationError, model_validator
except Exception:  # pragma: no cover - optional dependency
    BaseModel = None
    Field = None
    ValidationError = Exception
    model_validator = None


PYDANTIC_AVAILABLE = BaseModel is not None


if PYDANTIC_AVAILABLE:

    class SimulationInput(BaseModel):
        C0_mg_L: float = Field(gt=0)
        C_breakthrough_mg_L: float = Field(gt=0)
        flow: float = Field(gt=0)
        height: float = Field(gt=0)
        diameter: float = Field(gt=0)
        temperature_C: float
        epsilon: float = Field(gt=0, lt=1)
        bulk_density: float = Field(gt=0)
        KH_L_g: float = Field(gt=0)
        kLDF_1_min: float = Field(gt=0)
        Dax_cm2_min: float = Field(ge=0)
        n_cells: int = Field(ge=3)
        n_points: int = Field(ge=3)

        @model_validator(mode="after")
        def check_breakthrough(self):
            if self.C_breakthrough_mg_L >= self.C0_mg_L:
                raise ValueError("C_breakthrough_mg_L должен быть меньше C0_mg_L.")
            return self


    class IndustrialInput(BaseModel):
        Q_m3_h: float = Field(gt=0)
        D_col_m: float = Field(gt=0)
        H_m: float = Field(gt=0)
        dp_mm: float = Field(gt=0)
        epsilon: float = Field(gt=0.2, lt=0.7)
        bulk_density_kg_m3: float = Field(gt=0)
        particle_sphericity: float = Field(ge=0.5, le=1.0)
        T_C: float

else:
    SimulationInput = None
    IndustrialInput = None


def _pydantic_errors(exc: Exception) -> list[str]:
    if PYDANTIC_AVAILABLE and isinstance(exc, ValidationError):
        return [f"{'.'.join(str(x) for x in err.get('loc', []))}: {err.get('msg')}" for err in exc.errors()]
    return [str(exc)]


def validate_simulation_input(**kwargs: Any) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if PYDANTIC_AVAILABLE:
        try:
            SimulationInput(**kwargs)
        except Exception as exc:
            errors.extend(_pydantic_errors(exc))
    else:
        if kwargs.get("C0_mg_L", 0) <= 0:
            errors.append("C0 должен быть больше 0.")
        if kwargs.get("C_breakthrough_mg_L", 0) <= 0:
            errors.append("C прорыва должен быть больше 0.")
        if kwargs.get("C_breakthrough_mg_L", 0) >= kwargs.get("C0_mg_L", 0):
            errors.append("C прорыва должен быть меньше C0.")
        for key in ["flow", "height", "diameter", "bulk_density", "KH_L_g", "kLDF_1_min"]:
            if kwargs.get(key, 0) <= 0:
                errors.append(f"{key} должен быть больше 0.")
        if not (0 < kwargs.get("epsilon", 0) < 1):
            errors.append("epsilon должен быть в диапазоне (0, 1).")
        if kwargs.get("Dax_cm2_min", -1) < 0:
            errors.append("Dax_cm2_min должен быть >= 0.")
        if kwargs.get("n_cells", 0) < 3:
            errors.append("n_cells должен быть >= 3.")
        if kwargs.get("n_points", 0) < 3:
            errors.append("n_points должен быть >= 3.")

    if kwargs.get("C0_mg_L", 0) > 10:
        warnings.append(
            "C0 > 10 мг/л: линейная изотерма Генри может быть недостаточной; расчёт помечается как extrapolation."
        )
    return {"errors": errors, "warnings": warnings}


def validate_industrial_input(**kwargs: Any) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if PYDANTIC_AVAILABLE:
        try:
            IndustrialInput(**kwargs)
        except Exception as exc:
            errors.extend(_pydantic_errors(exc))
    else:
        if kwargs.get("Q_m3_h", 0) <= 0:
            errors.append("Q должен быть больше 0.")
        if kwargs.get("D_col_m", 0) <= 0:
            errors.append("Диаметр должен быть больше 0.")
        if kwargs.get("H_m", 0) <= 0:
            errors.append("Высота слоя должна быть больше 0.")
        if kwargs.get("dp_mm", 0) <= 0:
            errors.append("dp должен быть больше 0.")
        if not (0.2 < kwargs.get("epsilon", 0) < 0.7):
            errors.append("epsilon должен быть в диапазоне 0.2-0.7.")
        if kwargs.get("bulk_density_kg_m3", 0) <= 0:
            errors.append("Насыпная плотность должна быть больше 0.")
        if not (0.5 <= kwargs.get("particle_sphericity", 0) <= 1.0):
            errors.append("Сферичность частиц должна быть в диапазоне 0.5-1.0.")

    T_C = float(kwargs.get("T_C", 25.0))
    if not (5.0 <= T_C <= 60.0):
        warnings.append("T_C выходит за рекомендуемый диапазон 5-60 °C.")
    return {"errors": errors, "warnings": warnings}
