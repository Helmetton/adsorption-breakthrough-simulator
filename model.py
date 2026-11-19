# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from encoding_utils import CSV_ENCODINGS, fix_mojibake


DEFAULT_BREAKTHROUGH_RATIO = 0.05
MODEL_VERSION = "henry_ldf_conservative_fvm_bdf_v2"
LANGMUIR_MODEL_VERSION = "langmuir_kh_b_ldf_conservative_fvm_bdf_v1"
DEFAULT_N_CELLS = 320
SOLVER_RTOL = 1e-5
SOLVER_ATOL = 1e-7
CALIBRATION_MIN_POINTS = 4
CALIBRATION_INITIAL_GUESS = (0.03, 0.03, 0.01)
CALIBRATION_LOWER_BOUNDS = (1e-5, 1e-4, 1e-6)
CALIBRATION_UPPER_BOUNDS = (1.0, 10.0, 10.0)
CALIBRATION_MAX_NFEV = 120
CALIBRATION_XTOL = 1e-7
CALIBRATION_FTOL = 1e-7
CALIBRATION_GTOL = 1e-7


@dataclass(frozen=True)
class BreakthroughInterpretation:
    """Canonical scientific interpretation of a measured or simulated curve."""

    breakthrough_ratio_used: float
    breakthrough_concentration_mg_L: Optional[float]
    breakthrough_status: str
    breakthrough_time_min: Optional[float]
    breakthrough_bound_type: str
    startup_transient_detected: bool
    startup_end_time_min: Optional[float]
    first_below_threshold_time_min: Optional[float]
    rising_breakthrough_time_min: Optional[float]
    last_observation_time_min: Optional[float]
    original_breakthrough_concentration_mg_L: Optional[float] = None
    original_breakthrough_ratio: Optional[float] = None
    n_valid_points: int = 0
    duplicate_time_points_detected: bool = False
    first_observation_time_min: Optional[float] = None
    initially_at_or_above_threshold: bool = False

    @property
    def is_exact(self) -> bool:
        return self.breakthrough_bound_type == "exact"

    @property
    def label(self) -> str:
        if self.breakthrough_time_min is None:
            return "прорыв не определён"
        if self.breakthrough_bound_type == "upper_bound":
            return f"t_b <= {self.breakthrough_time_min:.2f} мин"
        if self.breakthrough_bound_type == "lower_bound":
            return f"t_b > {self.breakthrough_time_min:.2f} мин"
        return f"t_b = {self.breakthrough_time_min:.2f} мин"

    def to_dict(self) -> dict:
        result = asdict(self)
        # Compatibility aliases for existing plotting and reporting code.
        result.update(
            {
                "status": self.breakthrough_status,
                "time_min": self.breakthrough_time_min,
                "is_exact": self.is_exact,
                "label": self.label,
                "crossing_interpretation": (
                    "late_upward_crossing_after_initial_exceedance"
                    if self.startup_transient_detected and self.is_exact
                    else self.breakthrough_status
                ),
                "continuous_protection_time_established": bool(
                    self.is_exact and not self.initially_at_or_above_threshold
                    and self.first_observation_time_min == 0.0
                ),
                "continuous_protection_upper_bound_min": (
                    self.first_observation_time_min if self.initially_at_or_above_threshold else None
                ),
            }
        )
        return result


@dataclass(frozen=True)
class CalibrationStartResult:
    fit_start_time_min: float
    fit_start_source: str
    startup_transient_detected: bool
    startup_end_time_min: Optional[float]
    limited_prebreakthrough_information: bool
    limited_postbreakthrough_information: bool


@dataclass(frozen=True)
class ColumnParams:
    length_cm: float
    area_cm2: float
    mass_g: float
    epsilon: float
    flow_ml_min: float
    c0_mg_l: float


@dataclass(frozen=True)
class SorptionParams:
    KH_L_g: float
    kLDF_1_min: float
    Dax_cm2_min: float
    c_initial_mg_l: float = 0.0
    q_initial_mg_g: float = 0.0
    equilibrium_model: str = "Henry"
    b_L_mg: float = 0.0


def equilibrium_loading(concentration, params: SorptionParams):
    """Equilibrium closure only; transport and LDF remain time dependent.

    getattr preserves instances unpickled from the pre-extension dataclass.
    KH is dm3/g, C mg/dm3 and b dm3/mg, giving q* in mg/g.
    """
    name = getattr(params, "equilibrium_model", "Henry")
    linear = params.KH_L_g * concentration
    if name == "Henry":
        return linear
    if name != "Langmuir":
        raise ValueError(f"Unknown equilibrium model: {name}")
    b = getattr(params, "b_L_mg", 0.0)
    if not np.isfinite(b) or b < 0:
        raise ValueError("Langmuir b must be finite and nonnegative")
    return linear / (1.0 + b * concentration)


def sorption_model_version(params: SorptionParams) -> str:
    return LANGMUIR_MODEL_VERSION if getattr(params, "equilibrium_model", "Henry") == "Langmuir" else MODEL_VERSION


def column_area_from_diameter(diameter_cm: float) -> float:
    return math.pi * (diameter_cm / 2.0) ** 2


def load_experiments_csv(path: str) -> Dict[str, List[dict]]:
    experiments: Dict[str, List[dict]] = {}

    last_error: UnicodeDecodeError | None = None
    for encoding in CSV_ENCODINGS:
        experiments = {}
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row = {
                        fix_mojibake(key): fix_mojibake(value) if isinstance(value, str) else value
                        for key, value in row.items()
                    }
                    for key in [
                        "temperature_C",
                        "bed_height_cm",
                        "carbon_mass_g",
                        "column_area_cm2",
                        "flow_ml_min_assumed",
                        "C0_mg_L",
                        "C_break_mg_L",
                        "time_min",
                        "C_out_mg_L",
                        "C_out_C0",
                        "C1_mg_L",
                        "C2_mg_L",
                    ]:
                        if row.get(key, "") == "":
                            row[key] = None
                        else:
                            row[key] = float(row[key])

                    experiments.setdefault(row["experiment_id"], []).append(row)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        if last_error is not None:
            raise last_error

    for rows in experiments.values():
        rows.sort(key=lambda item: item["time_min"])

    return experiments


def make_column_from_experiment(rows: List[dict], epsilon: float = 0.40) -> ColumnParams:
    first = rows[0]
    if first["flow_ml_min_assumed"] is None:
        raise ValueError("Для эксперимента не задан расход flow_ml_min_assumed.")

    return ColumnParams(
        length_cm=float(first["bed_height_cm"]),
        area_cm2=float(first["column_area_cm2"]),
        mass_g=float(first["carbon_mass_g"]),
        epsilon=float(epsilon),
        flow_ml_min=float(first["flow_ml_min_assumed"]),
        c0_mg_l=float(first["C0_mg_L"]),
    )


def column_rhs(_t: float, y: np.ndarray, column: ColumnParams,
               params: SorptionParams, n_cells: int) -> np.ndarray:
    """Conservative cell-centred FVM; Dax is pore-liquid dispersion (cm2/min).

    epsilon*dC/dt = -div(J) - rho_bulk*dq/dt.
    J_in=u*C0, J_out=u*C_last; internal J=u*C_left-epsilon*Dax*grad(C).
    Thus dispersion redistributes solute but never adds an inlet source.
    """
    concentration, loading = y[:n_cells], y[n_cells:]
    dz = column.length_cm / n_cells
    u = column.flow_ml_min / column.area_cm2
    rho = 1000.0 * column.mass_g / (column.area_cm2 * column.length_cm)
    dq = params.kLDF_1_min * (equilibrium_loading(concentration, params) - loading)
    flux = np.empty(n_cells + 1)
    flux[0] = u * column.c0_mg_l
    flux[-1] = u * concentration[-1]
    flux[1:-1] = u * concentration[:-1] - column.epsilon * params.Dax_cm2_min * np.diff(concentration) / dz
    dc = (-np.diff(flux) / dz - rho * dq) / column.epsilon
    return np.concatenate([dc, dq])


def _solve_column(times_min, column, params, n_cells, rtol, atol):
    times = np.asarray(list(times_min), dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times_min must contain finite observations.")
    if times.min() < 0 or np.any(np.diff(times) < 0):
        raise ValueError("Times must be nonnegative and sorted.")
    if not 0 < column.epsilon < 1 or n_cells < 3:
        raise ValueError("Invalid porosity or spatial grid.")
    if min(column.length_cm, column.area_cm2, column.mass_g, column.c0_mg_l) <= 0 or column.flow_ml_min < 0:
        raise ValueError("Invalid column dimensions, mass, flow or inlet concentration.")
    if min(params.KH_L_g, params.kLDF_1_min, params.Dax_cm2_min,
           params.c_initial_mg_l, params.q_initial_mg_g) < 0:
        raise ValueError("Physical model parameters must be nonnegative.")
    physical_values = (*asdict(column).values(), params.KH_L_g, params.kLDF_1_min,
                       params.Dax_cm2_min, params.c_initial_mg_l, params.q_initial_mg_g)
    if not np.all(np.isfinite(physical_values)):
        raise ValueError("Physical model parameters must be finite.")
    equilibrium_loading(np.array([0.0]), params)
    unique_times, inverse = np.unique(times, return_inverse=True)
    y0 = np.r_[np.full(n_cells, params.c_initial_mg_l), np.full(n_cells, params.q_initial_mg_g)]
    if unique_times[-1] == 0:
        state = np.repeat(y0[:, None], len(times), axis=1)
    else:
        # Tridiagonal liquid coupling plus local liquid/solid exchange.
        from scipy.sparse import diags, eye, bmat
        liquid = diags([np.ones(n_cells-1), np.ones(n_cells), np.ones(n_cells-1)], [-1, 0, 1])
        local = eye(n_cells)
        sparsity = bmat([[liquid, local], [local, local]], format="csc")
        solution = solve_ivp(
            lambda t, y: column_rhs(t, y, column, params, n_cells),
            (0.0, float(unique_times[-1])), y0, t_eval=unique_times,
            method="BDF", rtol=rtol, atol=atol, jac_sparsity=sparsity,
        )
        if not solution.success:
            raise RuntimeError(f"BDF failed: {solution.message}")
        if solution.y.shape != (2 * n_cells, len(unique_times)) or not np.all(np.isfinite(solution.y)):
            raise RuntimeError("BDF returned incomplete or nonfinite state despite success status")
        state = solution.y[:, inverse]
    return times, state[:n_cells].T, state[n_cells:].T


def simulate_breakthrough(times_min: Iterable[float], column: ColumnParams,
                          params: SorptionParams, n_cells: int = DEFAULT_N_CELLS,
                          rtol: float = SOLVER_RTOL, atol: float = SOLVER_ATOL) -> np.ndarray:
    return _solve_column(times_min, column, params, n_cells, rtol, atol)[1][:, -1]


def simulate_breakthrough_full(times_min: Iterable[float], column: ColumnParams,
                               params: SorptionParams, n_cells: int = DEFAULT_N_CELLS,
                               rtol: float = SOLVER_RTOL, atol: float = SOLVER_ATOL) -> dict:
    times, concentration, loading = _solve_column(times_min, column, params, n_cells, rtol, atol)
    return {
        "time_min": times, "C_out_mg_L": concentration[:, -1],
        "C_out_C0": concentration[:, -1] / column.c0_mg_l,
        "C_profile_mg_L": concentration, "q_profile_mg_g": loading,
        "column": column, "params": params, "n_cells": n_cells,
        "model_version": sorption_model_version(params), "solver_success": True,
        "equilibrium_model": getattr(params, "equilibrium_model", "Henry"),
    }


def first_breakthrough_time(
    times_min: np.ndarray,
    concentration_ratio: np.ndarray,
    breakthrough_ratio: float,
) -> Optional[float]:
    """
    Возвращает первое время достижения границы прорыва C/C0.

    Важно: используется линейная интерполяция между двумя соседними
    точками, а не просто первая экспериментальная/расчётная точка выше
    порога. Поэтому t_b получается ближе к реальному пересечению линии
    прорыва на графике.
    """
    result = interpret_breakthrough_curve(
        times_min,
        concentration_ratio,
        c0_mg_l=1.0,
        breakthrough_ratio=breakthrough_ratio,
    )
    return result.breakthrough_time_min if result.is_exact else None


def interpret_breakthrough_curve(
    times_min: Iterable[float],
    concentration_ratio: Iterable[float],
    c0_mg_l: Optional[float] = None,
    breakthrough_ratio: float = DEFAULT_BREAKTHROUGH_RATIO,
    original_breakthrough_concentration_mg_l: Optional[float] = None,
    comparison_tolerance: float = 1e-12,
) -> BreakthroughInterpretation:
    """Interpret breakthrough without smoothing or altering measured values.

    Invalid pairs are ignored. Duplicate times are accepted only when their
    concentration ratios agree within floating-point tolerance; conflicting
    measurements at the same time are rejected as ambiguous.
    """

    times = np.asarray(list(times_min), dtype=float)
    ratio = np.asarray(list(concentration_ratio), dtype=float)
    threshold = float(breakthrough_ratio)
    if times.ndim != 1 or ratio.ndim != 1 or len(times) != len(ratio):
        raise ValueError("times_min и concentration_ratio должны быть одномерными массивами одинаковой длины.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("breakthrough_ratio должен находиться в диапазоне (0, 1).")

    valid = np.isfinite(times) & np.isfinite(ratio)
    times = times[valid]
    ratio = ratio[valid]
    if len(times):
        order = np.argsort(times, kind="mergesort")
        times = times[order]
        ratio = ratio[order]

    duplicate_times = False
    if len(times) > 1:
        keep = np.ones(len(times), dtype=bool)
        for index in range(1, len(times)):
            if abs(float(times[index]) - float(times[index - 1])) <= comparison_tolerance:
                duplicate_times = True
                if abs(float(ratio[index]) - float(ratio[index - 1])) > comparison_tolerance:
                    raise ValueError(
                        "Для одного времени обнаружены противоречащие значения C/C0; "
                        "автоматическая интерпретация прорыва невозможна."
                    )
                keep[index] = False
        times = times[keep]
        ratio = ratio[keep]

    original_ratio = None
    if (
        original_breakthrough_concentration_mg_l is not None
        and c0_mg_l is not None
        and float(c0_mg_l) > 0.0
    ):
        original_ratio = float(original_breakthrough_concentration_mg_l) / float(c0_mg_l)
    scientific_concentration = (
        None if c0_mg_l is None else threshold * float(c0_mg_l)
    )

    common = {
        "breakthrough_ratio_used": threshold,
        "breakthrough_concentration_mg_L": scientific_concentration,
        "original_breakthrough_concentration_mg_L": original_breakthrough_concentration_mg_l,
        "original_breakthrough_ratio": original_ratio,
        "n_valid_points": int(len(times)),
        "duplicate_time_points_detected": bool(duplicate_times),
        "first_observation_time_min": float(times[0]) if len(times) else None,
        "initially_at_or_above_threshold": bool(len(times) and ratio[0] >= threshold - comparison_tolerance),
    }
    if len(times) == 0:
        return BreakthroughInterpretation(
            breakthrough_status="not_available",
            breakthrough_time_min=None,
            breakthrough_bound_type="not_available",
            startup_transient_detected=False,
            startup_end_time_min=None,
            first_below_threshold_time_min=None,
            rising_breakthrough_time_min=None,
            last_observation_time_min=None,
            **common,
        )

    last_time = float(times[-1])
    first_is_at_or_above = float(ratio[0]) >= threshold - comparison_tolerance
    first_below_index = next(
        (
            index
            for index, value in enumerate(ratio)
            if float(value) < threshold - comparison_tolerance
        ),
        None,
    )

    if first_is_at_or_above and first_below_index is None:
        return BreakthroughInterpretation(
            breakthrough_status="already_above_at_first_sample",
            breakthrough_time_min=float(times[0]),
            breakthrough_bound_type="upper_bound",
            startup_transient_detected=False,
            startup_end_time_min=None,
            first_below_threshold_time_min=None,
            rising_breakthrough_time_min=None,
            last_observation_time_min=last_time,
            **common,
        )

    startup_detected = bool(first_is_at_or_above and first_below_index is not None)
    if startup_detected:
        below_index = int(first_below_index)
        t1 = float(times[below_index - 1])
        y1 = float(ratio[below_index - 1])
        t2 = float(times[below_index])
        y2 = float(ratio[below_index])
        if abs(y1 - threshold) <= comparison_tolerance:
            startup_end = t1
        else:
            startup_end = float(t1 + (threshold - y1) * (t2 - t1) / (y2 - y1))
        search_start_index = below_index
        first_below_time = t2
    else:
        startup_end = None
        search_start_index = 0
        first_below_time = (
            float(times[first_below_index]) if first_below_index is not None else None
        )

    for index in range(max(1, search_start_index + 1), len(times)):
        y1 = float(ratio[index - 1])
        y2 = float(ratio[index])
        if y1 >= threshold - comparison_tolerance:
            continue
        if y2 < threshold - comparison_tolerance:
            continue
        t1 = float(times[index - 1])
        t2 = float(times[index])
        if abs(y2 - threshold) <= comparison_tolerance:
            t_break = t2
            status = "reached_at_measured_point"
        else:
            t_break = float(t1 + (threshold - y1) * (t2 - t1) / (y2 - y1))
            status = "reached_by_interpolation"
        return BreakthroughInterpretation(
            breakthrough_status=status,
            breakthrough_time_min=t_break,
            breakthrough_bound_type="exact",
            startup_transient_detected=startup_detected,
            startup_end_time_min=startup_end,
            first_below_threshold_time_min=first_below_time,
            rising_breakthrough_time_min=t_break,
            last_observation_time_min=last_time,
            **common,
        )

    return BreakthroughInterpretation(
        breakthrough_status="not_reached",
        breakthrough_time_min=last_time,
        breakthrough_bound_type="lower_bound",
        startup_transient_detected=startup_detected,
        startup_end_time_min=startup_end,
        first_below_threshold_time_min=first_below_time,
        rising_breakthrough_time_min=None,
        last_observation_time_min=last_time,
        **common,
    )


def determine_calibration_start(
    times_min: Iterable[float],
    breakthrough: BreakthroughInterpretation,
    manual_fit_start_time_min: Optional[float] = None,
) -> CalibrationStartResult:
    valid_times = np.asarray(list(times_min), dtype=float)
    valid_times = np.sort(valid_times[np.isfinite(valid_times)])
    if len(valid_times) == 0:
        raise ValueError("Невозможно определить начало калибровки без валидных значений времени.")
    if manual_fit_start_time_min is not None:
        fit_start = float(manual_fit_start_time_min)
        if fit_start < 0.0:
            raise ValueError("Ручное время начала калибровки должно быть >= 0.")
        source = "manual"
    elif breakthrough.startup_transient_detected:
        if breakthrough.first_below_threshold_time_min is None:
            raise ValueError("Startup transient обнаружен, но его окончание не определено.")
        fit_start = float(breakthrough.first_below_threshold_time_min)
        source = "automatic"
    else:
        fit_start = float(valid_times[0])
        source = "automatic"
    return CalibrationStartResult(
        fit_start_time_min=fit_start,
        fit_start_source=source,
        startup_transient_detected=breakthrough.startup_transient_detected,
        startup_end_time_min=breakthrough.startup_end_time_min,
        limited_prebreakthrough_information=(
            breakthrough.breakthrough_bound_type == "upper_bound"
        ),
        limited_postbreakthrough_information=(
            breakthrough.breakthrough_bound_type == "lower_bound"
        ),
    )


def breakthrough_status(
    times_min: np.ndarray,
    concentration_ratio: np.ndarray,
    breakthrough_ratio: float,
) -> dict:
    return interpret_breakthrough_curve(
        times_min,
        concentration_ratio,
        c0_mg_l=1.0,
        breakthrough_ratio=breakthrough_ratio,
    ).to_dict()


def breakthrough_ratio_from_concentration(
    c_breakthrough_mg_l: Optional[float],
    c0_mg_l: float,
) -> float:
    """Return the active scientific criterion.

    ``c_breakthrough_mg_l`` remains in the signature for backward-compatible
    provenance loading. Historical absolute thresholds must not silently alter
    the canonical five-percent criterion.
    """

    _ = c_breakthrough_mg_l, c0_mg_l
    return DEFAULT_BREAKTHROUGH_RATIO


def experiment_breakthrough_concentration(rows: List[dict]) -> Optional[float]:
    for row in rows:
        value = row.get("C_break_mg_L")
        if value is not None and value > 0:
            return float(value)
    return None


def fit_model_to_experiment(
    rows: List[dict],
    epsilon: float = 0.40,
    min_time_for_fit: Optional[float] = None,
    n_cells: int = DEFAULT_N_CELLS,
    fit_initial_concentration: bool = False,
    breakthrough_ratio: float = DEFAULT_BREAKTHROUGH_RATIO,
    initial_parameters: Optional[Tuple[float, float, float]] = None,
    equilibrium_model: str = "Henry",
    initial_b_L_mg: float = 0.01,
) -> Tuple[SorptionParams, dict]:
    if equilibrium_model not in {"Henry", "Langmuir"}:
        raise ValueError(f"Unknown equilibrium model: {equilibrium_model}")
    column = make_column_from_experiment(rows, epsilon=epsilon)
    times = np.array([row["time_min"] for row in rows], dtype=float)
    observed = np.array([row["C_out_mg_L"] for row in rows], dtype=float)
    valid_observations = np.isfinite(times) & np.isfinite(observed)
    original_c_break = experiment_breakthrough_concentration(rows)
    interpretation = interpret_breakthrough_curve(
        times,
        observed / column.c0_mg_l,
        c0_mg_l=column.c0_mg_l,
        breakthrough_ratio=breakthrough_ratio,
        original_breakthrough_concentration_mg_l=original_c_break,
    )
    calibration_start = determine_calibration_start(
        times,
        interpretation,
        manual_fit_start_time_min=min_time_for_fit,
    )
    mask = valid_observations & (times >= calibration_start.fit_start_time_min)

    if np.sum(mask) < CALIBRATION_MIN_POINTS:
        raise ValueError(
            "После научно обоснованного исключения startup-участка осталось "
            f"{int(np.sum(mask))} точек; для калибровки требуется минимум "
            f"{CALIBRATION_MIN_POINTS}. Исключённые точки не были возвращены в fit."
        )

    calibration_mode = (
        "diagnostic_fit_C_initial" if fit_initial_concentration else "clean_bed"
    )

    if fit_initial_concentration:
        initial_guess = (*(initial_parameters or CALIBRATION_INITIAL_GUESS), 0.001)
        lower_bounds = (*CALIBRATION_LOWER_BOUNDS, 1e-6)
        upper_bounds = (*CALIBRATION_UPPER_BOUNDS, max(column.c0_mg_l, 1e-3))
    else:
        initial_guess = initial_parameters or CALIBRATION_INITIAL_GUESS
        lower_bounds = CALIBRATION_LOWER_BOUNDS
        upper_bounds = CALIBRATION_UPPER_BOUNDS
    x0 = np.log(initial_guess)
    lower = np.log(lower_bounds)
    upper = np.log(upper_bounds)
    # Log-transform positive transport coefficients, but not b: b=0 is the
    # physical Henry boundary and must be admissible, not replaced by epsilon.
    if equilibrium_model == "Langmuir":
        x0 = np.r_[x0, float(initial_b_L_mg)]
        lower = np.r_[lower, 0.0]
        upper = np.r_[upper, 10.0]

    def decode(x):
        base = x[:-1] if equilibrium_model == "Langmuir" else x
        values = np.exp(base)
        kh, kldf, dax = values[:3]
        c_init = values[3] if fit_initial_concentration else 0.0
        return SorptionParams(kh, kldf, dax, c_initial_mg_l=c_init,
                              q_initial_mg_g=0.0, equilibrium_model=equilibrium_model,
                              b_L_mg=float(x[-1]) if equilibrium_model == "Langmuir" else 0.0)

    def residual(log_x: np.ndarray) -> np.ndarray:
        params = decode(log_x)
        predicted = simulate_breakthrough(times, column, params, n_cells=n_cells)
        return (predicted[mask] - observed[mask]) / column.c0_mg_l

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=CALIBRATION_MAX_NFEV,
        xtol=CALIBRATION_XTOL,
        ftol=CALIBRATION_FTOL,
        gtol=CALIBRATION_GTOL,
    )

    fitted = decode(result.x)
    predicted = simulate_breakthrough(times, column, fitted, n_cells=n_cells)
    residuals = predicted[mask] - observed[mask]
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((observed[mask] - np.mean(observed[mask])) ** 2))
    r2 = None if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    metrics = {
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimizer_status": int(result.status),
        "optimizer_nfev": int(result.nfev),
        "optimizer_optimality": float(result.optimality),
        "optimizer_active_bounds": result.active_mask.tolist(),
        "parameters_at_bounds": bool(np.any(result.active_mask)),
        "n_cells": int(n_cells),
        "small_signal_policy": "retain_algebraic_concentrations_unclipped",
        "rmse_mg_L": rmse,
        "rmse_C0": float(rmse / column.c0_mg_l),
        "r2": r2,
        "n_fit_points": int(np.sum(mask)),
        "fit_observation_indices": np.flatnonzero(mask).tolist(),
        "fit_start_time_min": calibration_start.fit_start_time_min,
        "fit_start_source": calibration_start.fit_start_source,
        "startup_transient_detected": calibration_start.startup_transient_detected,
        "startup_end_time_min": calibration_start.startup_end_time_min,
        "limited_prebreakthrough_information": calibration_start.limited_prebreakthrough_information,
        "limited_postbreakthrough_information": calibration_start.limited_postbreakthrough_information,
        "breakthrough_interpretation": interpretation.to_dict(),
        "breakthrough_ratio_used": float(breakthrough_ratio),
        "initial_guesses": list(map(float, initial_guess)),
        "optimizer_lower_bounds": list(map(float, lower_bounds)),
        "optimizer_upper_bounds": list(map(float, upper_bounds)),
        "optimizer_max_nfev": CALIBRATION_MAX_NFEV,
        "optimizer_xtol": CALIBRATION_XTOL,
        "optimizer_ftol": CALIBRATION_FTOL,
        "optimizer_gtol": CALIBRATION_GTOL,
        "solver_method": "BDF",
        "solver_rtol": 1e-5,
        "solver_atol": 1e-7,
        "model_version": sorption_model_version(fitted),
        "equilibrium_model": equilibrium_model,
        "b_L_mg": fitted.b_L_mg,
        "b_initial_L_mg": float(initial_b_L_mg) if equilibrium_model == "Langmuir" else None,
        "b_bounds_L_mg": [0.0, 10.0] if equilibrium_model == "Langmuir" else None,
        "calibration_mode": calibration_mode,
        "fit_initial_concentration": bool(fit_initial_concentration),
        "times_min": times,
        "observed_mg_L": observed,
        "predicted_mg_L": predicted,
    }
    return fitted, metrics


def bootstrap_fit_uncertainty(
    rows: List[dict],
    epsilon: float = 0.40,
    min_time_for_fit: Optional[float] = None,
    n_cells: int = DEFAULT_N_CELLS,
    n_bootstrap: int = 30,
    fit_initial_concentration: bool = False,
    seed: int = 42,
) -> dict:
    """Estimate parameter uncertainty by bootstrap resampling experimental points.

    The function repeatedly refits the same model to resampled breakthrough points.
    It is intended as an optional diagnostic and does not alter the calibrated model.
    """
    rng = np.random.default_rng(seed)
    if len(rows) < 4:
        return {
            "samples": [],
            "confidence_intervals": {},
            "correlation_matrix": {},
            "n_success": 0,
            "n_failed": int(n_bootstrap),
        }

    samples: list[dict] = []
    failures = 0
    metadata_row = dict(rows[0])
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, len(rows), size=len(rows))
        sampled_rows = [dict(rows[int(i)]) for i in indices]
        sampled_rows.sort(key=lambda item: item["time_min"])
        for key in metadata_row:
            if key not in {"time_min", "C_out_mg_L", "C_out_C0"}:
                for row in sampled_rows:
                    row[key] = metadata_row[key]
        try:
            fitted, metrics = fit_model_to_experiment(
                sampled_rows,
                epsilon=epsilon,
                min_time_for_fit=min_time_for_fit,
                n_cells=n_cells,
                fit_initial_concentration=fit_initial_concentration,
            )
        except Exception:
            failures += 1
            continue
        if not metrics["success"]:
            failures += 1
            continue

        samples.append(
            {
                "K_H_L_g": fitted.KH_L_g,
                "kLDF_1_min": fitted.kLDF_1_min,
                "Dax_cm2_min": fitted.Dax_cm2_min,
                "C_initial_mg_L": fitted.c_initial_mg_l,
                "RMSE_C0": metrics.get("rmse_C0"),
                "R2": metrics.get("r2"),
            }
        )

    if not samples:
        return {
            "samples": [],
            "confidence_intervals": {},
            "correlation_matrix": {},
            "n_success": 0,
            "n_failed": failures,
        }

    sample_matrix = {
        key: np.array([row[key] for row in samples], dtype=float)
        for key in ["K_H_L_g", "kLDF_1_min", "Dax_cm2_min"]
    }
    confidence_intervals = {
        key: {
            "p2_5": float(np.nanpercentile(values, 2.5)),
            "p50": float(np.nanpercentile(values, 50.0)),
            "p97_5": float(np.nanpercentile(values, 97.5)),
        }
        for key, values in sample_matrix.items()
    }
    param_names = list(sample_matrix.keys())
    corr_values = np.corrcoef(np.vstack([sample_matrix[name] for name in param_names]))
    correlation_matrix = {
        row_name: {
            col_name: float(corr_values[i, j])
            for j, col_name in enumerate(param_names)
        }
        for i, row_name in enumerate(param_names)
    }
    return {
        "samples": samples,
        "confidence_intervals": confidence_intervals,
        "correlation_matrix": correlation_matrix,
        "n_success": len(samples),
        "n_failed": failures,
    }


def run_simulation(
    column: ColumnParams,
    params: SorptionParams,
    c_breakthrough_mg_l: Optional[float],
    t_end_min: float,
    n_points: int,
    n_cells: int = DEFAULT_N_CELLS,
) -> dict:
    times = np.linspace(0.0, float(t_end_min), int(n_points))
    concentrations = simulate_breakthrough(times, column, params, n_cells=n_cells)
    ratio = concentrations / column.c0_mg_l
    breakthrough_result = interpret_breakthrough_curve(
        times,
        ratio,
        c0_mg_l=column.c0_mg_l,
        breakthrough_ratio=DEFAULT_BREAKTHROUGH_RATIO,
        original_breakthrough_concentration_mg_l=c_breakthrough_mg_l,
    )
    breakthrough_ratio = breakthrough_result.breakthrough_ratio_used
    breakthrough_time = breakthrough_result.breakthrough_time_min
    bed_volume = None if breakthrough_time is None else column.flow_ml_min * breakthrough_time

    return {
        "time_min": times,
        "C_out_mg_L": concentrations,
        "C_out_C0": ratio,
        "breakthrough_ratio": breakthrough_ratio,
        "breakthrough_concentration_mg_L": breakthrough_result.breakthrough_concentration_mg_L,
        "breakthrough_time_min": breakthrough_time,
        "breakthrough_status": breakthrough_result.breakthrough_status,
        "breakthrough_bound_type": breakthrough_result.breakthrough_bound_type,
        "breakthrough_interpretation": breakthrough_result.to_dict(),
        "breakthrough_volume_ml": bed_volume,
    }


def generate_ml_dataset(
    n_cases: int,
    time_grid_min: Iterable[float],
    seed: int = 42,
) -> List[dict]:
    rng = random.Random(seed)
    rows: List[dict] = []
    area_cm2 = math.pi * 3.2**2 / 4.0
    times = np.array(list(time_grid_min), dtype=float)

    for case_id in range(1, n_cases + 1):
        height = rng.uniform(4.0, 12.0)
        bulk_density_g_cm3 = rng.uniform(0.22, 0.32)
        mass = bulk_density_g_cm3 * area_cm2 * height
        flow = rng.choice([2.0, 3.0, 4.5, 6.0, 9.0, 12.0])
        c0 = rng.uniform(2.0, 10.0)
        temp = rng.uniform(10.0, 30.0)
        epsilon = rng.uniform(0.35, 0.48)

        kh_20 = rng.uniform(0.01, 0.08)
        kh = kh_20 * math.exp(-0.015 * (temp - 20.0))
        kldf_20 = rng.uniform(0.005, 0.08)
        kldf = kldf_20 * math.exp(0.025 * (temp - 20.0))
        dax = rng.uniform(1e-4, 0.05)

        column = ColumnParams(height, area_cm2, mass, epsilon, flow, c0)
        params = SorptionParams(kh, kldf, dax)
        c_out = simulate_breakthrough(times, column, params, n_cells=DEFAULT_N_CELLS)

        for time_min, concentration in zip(times, c_out):
            rows.append(
                {
                    "case_id": case_id,
                    "mechanistic_model_id": MODEL_VERSION,
                    "n_cells": DEFAULT_N_CELLS,
                    "time_min": float(time_min),
                    "C0_mg_L": c0,
                    "temperature_C": temp,
                    "bed_height_cm": height,
                    "carbon_mass_g": mass,
                    "column_area_cm2": area_cm2,
                    "flow_ml_min": flow,
                    "epsilon": epsilon,
                    "KH_L_g": kh,
                    "kLDF_1_min": kldf,
                    "Dax_cm2_min": dax,
                    "C_out_mg_L": float(concentration),
                    "C_out_C0": float(concentration / c0),
                }
            )

    return rows
