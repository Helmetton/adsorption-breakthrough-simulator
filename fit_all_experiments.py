# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import platform
import random
from pathlib import Path
from typing import Optional

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from model import (
    DEFAULT_BREAKTHROUGH_RATIO,
    MODEL_VERSION,
    determine_calibration_start,
    experiment_breakthrough_concentration,
    fit_model_to_experiment,
    interpret_breakthrough_curve,
    load_experiments_csv,
    make_column_from_experiment,
    simulate_breakthrough,
)


DATA_PATH = "cleaned_dynamic_sorption_data.csv"
OUTPUT_CSV = "fitted_experiment_parameters.csv"
FIGURES_DIR = Path("figures") / "fitted_experiments"
COMPARISON_PATH = Path("figures") / "comparison_all_experiments.png"
SUMMARY_CSV = "dynamic_sorption_experiment_summary.csv"
SCIENTIFIC_RANDOM_SEED = 42


def experiment_replicate_count(rows: list[dict]) -> int:
    return 2 if any(row.get("C2_mg_L") is not None for row in rows) else 1


def build_experiment_summary(
    data_path: str = DATA_PATH,
    output_csv: str = SUMMARY_CSV,
) -> pd.DataFrame:
    """Regenerate the experiment summary directly from the cleaned observations."""

    experiments = load_experiments_csv(data_path)
    summary_rows: list[dict] = []
    for experiment_id, rows in experiments.items():
        column = make_column_from_experiment(rows)
        times = np.asarray([row["time_min"] for row in rows], dtype=float)
        concentrations = np.asarray([row["C_out_mg_L"] for row in rows], dtype=float)
        ratios = concentrations / column.c0_mg_l
        original_c_break = experiment_breakthrough_concentration(rows)
        breakthrough = interpret_breakthrough_curve(
            times,
            ratios,
            c0_mg_l=column.c0_mg_l,
            breakthrough_ratio=DEFAULT_BREAKTHROUGH_RATIO,
            original_breakthrough_concentration_mg_l=original_c_break,
        )
        fit_start = determine_calibration_start(times, breakthrough)
        first_above = next(
            (
                float(time)
                for time, ratio in zip(times, ratios)
                if float(ratio) >= DEFAULT_BREAKTHROUGH_RATIO
            ),
            None,
        )
        first = rows[0]
        replicate_count = experiment_replicate_count(rows)
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "source_sheet": first.get("source_sheet"),
                "temperature_C": first.get("temperature_C"),
                "bed_height_cm": first.get("bed_height_cm"),
                "carbon_mass_g": first.get("carbon_mass_g"),
                "speed_label": first.get("speed_label"),
                "flow_ml_min_assumed": first.get("flow_ml_min_assumed"),
                "C0_mg_L": column.c0_mg_l,
                # Backward-compatible provenance column: original laboratory value.
                "C_break_mg_L": original_c_break,
                "C_break_original_mg_L": original_c_break,
                "original_breakthrough_ratio": breakthrough.original_breakthrough_ratio,
                "C_break_scientific_mg_L": breakthrough.breakthrough_concentration_mg_L,
                "breakthrough_ratio_used": breakthrough.breakthrough_ratio_used,
                "n_points": len(rows),
                "n_replicates": replicate_count,
                "calibration_curve_representation": (
                    "mean_of_parallel_C1_C2" if replicate_count > 1 else "single_run"
                ),
                "time_start_min": float(np.min(times)),
                "time_end_min": float(np.max(times)),
                "C_min_mg_L": float(np.min(concentrations)),
                "C_max_mg_L": float(np.max(concentrations)),
                "C_first_mg_L": float(concentrations[0]),
                "C_last_mg_L": float(concentrations[-1]),
                "C_min_C0": float(np.min(ratios)),
                "C_max_C0": float(np.max(ratios)),
                "first_C_ge_Cbreak_min": first_above,
                "rising_cross_after_below_Cbreak_min": breakthrough.rising_breakthrough_time_min,
                "breakthrough_status": breakthrough.breakthrough_status,
                "breakthrough_time_min": breakthrough.breakthrough_time_min,
                "breakthrough_bound_type": breakthrough.breakthrough_bound_type,
                "startup_transient_detected": breakthrough.startup_transient_detected,
                "startup_end_time_min": breakthrough.startup_end_time_min,
                "first_below_threshold_time_min": breakthrough.first_below_threshold_time_min,
                "automatic_fit_start_time_min": fit_start.fit_start_time_min,
                "limited_prebreakthrough_information": fit_start.limited_prebreakthrough_information,
                "limited_postbreakthrough_information": fit_start.limited_postbreakthrough_information,
            }
        )
    result = pd.DataFrame(summary_rows)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def fit_quality_label(
    r2: Optional[float],
    rmse_c0: Optional[float],
    breakthrough_status_exp: str,
) -> str:
    if r2 is None or rmse_c0 is None:
        return "нет оценки"
    if breakthrough_status_exp == "not_reached" and rmse_c0 <= 0.01:
        return "низкая RMSE, но прорыв не достигнут: не независимая валидация"
    if r2 >= 0.90 and rmse_c0 <= 0.03:
        return "хорошая подгонка"
    if 0.70 <= r2 < 0.90 and rmse_c0 <= 0.05:
        return "средняя подгонка"
    return "плохая подгонка / проверить эксперимент"


def calibration_mode_label(calibration_mode: str) -> str:
    if calibration_mode == "diagnostic_fit_C_initial":
        return "диагностический режим"
    return "режим: чистый слой"


def classify_experiment(
    calibration_mode: str,
    breakthrough_status_exp: str,
    breakthrough_status_model: str,
    r2: Optional[float],
    rmse_c0: Optional[float],
) -> dict:
    r2_value = float(r2) if r2 is not None and np.isfinite(r2) else None
    rmse_value = float(rmse_c0) if rmse_c0 is not None and np.isfinite(rmse_c0) else None

    use_for_calibration = (
        calibration_mode == "clean_bed"
        and breakthrough_status_exp in {"reached_by_interpolation", "reached_at_measured_point"}
        and breakthrough_status_model in {"reached_by_interpolation", "reached_at_measured_point"}
        and r2_value is not None
        and rmse_value is not None
        and r2_value >= 0.90
        and rmse_value <= 0.02
    )
    exclude_from_ml_ranges = (
        breakthrough_status_exp == "already_above_at_first_sample"
        or (r2_value is not None and r2_value < 0.70)
        or (rmse_value is not None and rmse_value > 0.03)
    )
    exact_statuses = {"reached_by_interpolation", "reached_at_measured_point"}
    use_for_validation = (
        not use_for_calibration
        and breakthrough_status_exp in exact_statuses
        and breakthrough_status_model in exact_statuses
        and r2_value is not None
        and rmse_value is not None
        and r2_value >= 0.70
        and rmse_value <= 0.03
    )

    if use_for_calibration:
        exclude_reason = "подходит для основной калибровки"
    elif breakthrough_status_exp == "already_above_at_first_sample":
        exclude_reason = "первая точка уже выше границы прорыва"
    elif breakthrough_status_exp == "not_reached":
        exclude_reason = "прорыв не достигнут"
    elif (r2_value is not None and r2_value < 0.70) or (
        rmse_value is not None and rmse_value > 0.03
    ):
        exclude_reason = "низкое качество подгонки"
    elif use_for_validation:
        exclude_reason = "подходит только для валидации"
    else:
        exclude_reason = "низкое качество подгонки"

    return {
        "use_for_calibration": bool(use_for_calibration),
        "use_for_validation": bool(use_for_validation),
        "exclude_from_ml_ranges": bool(exclude_from_ml_ranges),
        "exclude_reason": exclude_reason,
        "successful_model_validation": False,
    }


def plot_fit(
    experiment_id: str,
    exp_time: np.ndarray,
    exp_ratio: np.ndarray,
    model_time: np.ndarray,
    model_ratio: np.ndarray,
    breakthrough_ratio: float,
    breakthrough_exp: dict,
    breakthrough_model: dict,
    fit_start_time: float,
    startup_end_time: Optional[float],
    calibration_mode: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        exp_time,
        exp_ratio,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="экспериментальная кривая через точки",
    )
    ax.plot(exp_time, exp_ratio, "o", markersize=4, label="экспериментальные точки")
    ax.plot(
        model_time,
        model_ratio,
        "-",
        linewidth=2,
        label="модельная кривая",
    )
    ax.axhline(
        breakthrough_ratio,
        color="tab:red",
        linestyle="--",
        linewidth=1.3,
        label=f"граница прорыва C/C0 = {breakthrough_ratio:.3f}",
    )
    if breakthrough_exp["time_min"] is not None:
        ax.axvline(
            breakthrough_exp["time_min"],
            color="tab:green",
            linestyle=":",
            linewidth=1.5,
            label=f"t прорыва, эксперимент: {breakthrough_exp['label']}",
        )
    elif breakthrough_exp["status"] == "not_reached":
        ax.text(
            0.02,
            0.93,
            "эксперимент: прорыв не достигнут",
            transform=ax.transAxes,
            fontsize=8,
            color="tab:green",
            va="top",
        )
    if breakthrough_model["time_min"] is not None:
        ax.axvline(
            breakthrough_model["time_min"],
            color="tab:orange",
            linestyle="-.",
            linewidth=1.5,
            label=f"t прорыва, модель: {breakthrough_model['label']}",
        )
    elif breakthrough_model["status"] == "not_reached":
        ax.text(
            0.02,
            0.86,
            "модель: прорыв не достигнут",
            transform=ax.transAxes,
            fontsize=8,
            color="tab:orange",
            va="top",
        )
    if fit_start_time > 0:
        ax.axvline(
            fit_start_time,
            color="tab:gray",
            linestyle="--",
            linewidth=1.1,
            label=f"начало подгонки = {fit_start_time:.0f} мин",
        )
    if startup_end_time is not None:
        ax.axvline(
            startup_end_time,
            color="tab:purple",
            linestyle=":",
            linewidth=1.1,
            label=f"конец startup washout = {startup_end_time:.2f} мин",
        )

    ax.set_title(f"{experiment_id} | {calibration_mode_label(calibration_mode)}")
    ax.set_xlabel("Время, мин")
    ax.set_ylabel("C/C0")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    from ui_presentation import localize_figure
    localize_figure(fig, "uk")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fit_all_experiments(
    data_path: str = DATA_PATH,
    output_csv: str = OUTPUT_CSV,
    figures_dir: Path | str = FIGURES_DIR,
    comparison_path: Path | str = COMPARISON_PATH,
    epsilon: float = 0.40,
    min_time_for_fit: Optional[float] = None,
    fit_n_cells: int = 320,
    plot_n_cells: Optional[int] = None,
    calibration_mode: str = "clean_bed",
) -> pd.DataFrame:
    plot_n_cells = fit_n_cells if plot_n_cells is None else plot_n_cells
    if plot_n_cells != fit_n_cells:
        raise ValueError("Fit and prediction require identical spatial grids.")
    if calibration_mode not in {"clean_bed", "diagnostic_fit_C_initial"}:
        raise ValueError(
            "calibration_mode должен быть 'clean_bed' или 'diagnostic_fit_C_initial'."
        )

    random.seed(SCIENTIFIC_RANDOM_SEED)
    np.random.seed(SCIENTIFIC_RANDOM_SEED)
    fit_initial_concentration = calibration_mode == "diagnostic_fit_C_initial"
    experiments = load_experiments_csv(data_path)
    figures_dir = Path(figures_dir)
    comparison_path = Path(comparison_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []
    plot_payloads: list[dict] = []

    for experiment_id, rows in experiments.items():
        fitted, metrics = fit_model_to_experiment(
            rows,
            epsilon=epsilon,
            min_time_for_fit=min_time_for_fit,
            n_cells=fit_n_cells,
            fit_initial_concentration=fit_initial_concentration,
            breakthrough_ratio=DEFAULT_BREAKTHROUGH_RATIO,
        )
        column = make_column_from_experiment(rows, epsilon=epsilon)
        exp_time = np.array([row["time_min"] for row in rows], dtype=float)
        exp_conc = np.array([row["C_out_mg_L"] for row in rows], dtype=float)
        exp_ratio = exp_conc / column.c0_mg_l

        original_c_break = experiment_breakthrough_concentration(rows)
        breakthrough_ratio = DEFAULT_BREAKTHROUGH_RATIO
        breakthrough_exp_result = interpret_breakthrough_curve(
            exp_time,
            exp_ratio,
            c0_mg_l=column.c0_mg_l,
            breakthrough_ratio=breakthrough_ratio,
            original_breakthrough_concentration_mg_l=original_c_break,
        )
        breakthrough_exp = breakthrough_exp_result.to_dict()
        t_break_exp = breakthrough_exp["time_min"]

        model_time = np.linspace(0.0, float(np.max(exp_time)), 260)
        model_conc = simulate_breakthrough(model_time, column, fitted, n_cells=plot_n_cells)
        model_ratio = model_conc / column.c0_mg_l
        breakthrough_model_result = interpret_breakthrough_curve(
            model_time,
            model_ratio,
            c0_mg_l=column.c0_mg_l,
            breakthrough_ratio=breakthrough_ratio,
        )
        breakthrough_model = breakthrough_model_result.to_dict()
        t_break_model = breakthrough_model["time_min"]

        t_error = (
            None
            if (
                t_break_exp is None
                or t_break_model is None
                or not breakthrough_exp["is_exact"]
                or not breakthrough_model["is_exact"]
            )
            else t_break_model - t_break_exp
        )
        vb_exp = None if t_break_exp is None else column.flow_ml_min * t_break_exp
        vb_model = None if t_break_model is None else column.flow_ml_min * t_break_model
        quality = fit_quality_label(
            metrics.get("r2"),
            metrics.get("rmse_C0"),
            breakthrough_exp["status"],
        )
        first_row = rows[0]
        temperature_values = []
        for row in rows:
            try:
                value = float(row.get("temperature_C"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                temperature_values.append(value)
        T_exp_C = float(np.median(temperature_values)) if temperature_values else None
        T_ref_C = T_exp_C if T_exp_C is not None else 20.0
        bed_loading_density = column.mass_g / (column.area_cm2 * column.length_cm)
        bed_volume_cm3 = column.area_cm2 * column.length_cm
        equivalent_diameter_cm = math.sqrt(4.0 * column.area_cm2 / math.pi)
        c_initial_c0 = fitted.c_initial_mg_l / column.c0_mg_l
        nonphysical_initial_flag = c_initial_c0 > breakthrough_ratio
        classification = classify_experiment(
            calibration_mode,
            breakthrough_exp["status"],
            breakthrough_model["status"],
            metrics.get("r2"),
            metrics.get("rmse_C0"),
        )

        output_path = figures_dir / f"{safe_filename(experiment_id)}.png"
        plot_fit(
            experiment_id,
            exp_time,
            exp_ratio,
            model_time,
            model_ratio,
            breakthrough_ratio,
            breakthrough_exp,
            breakthrough_model,
            metrics["fit_start_time_min"],
            metrics.get("startup_end_time_min"),
            calibration_mode,
            output_path,
        )

        row_out = {
            "experiment_id": experiment_id,
            "KH_L_g": fitted.KH_L_g,
            "kLDF_1_min": fitted.kLDF_1_min,
            "Dax_cm2_min": fitted.Dax_cm2_min,
            "T_exp_C": T_exp_C,
            "T_ref_C": T_ref_C,
            "temperature_reference_assumed": T_exp_C is None,
            "validation_scope": "calibration_only_not_independent_validation",
            "KH_ref_L_g": fitted.KH_L_g,
            "kLDF_ref_1_min": fitted.kLDF_1_min,
            "Dax_ref_cm2_min": fitted.Dax_cm2_min,
            "C_initial_mg_L": fitted.c_initial_mg_l,
            "C_initial_C0": c_initial_c0,
            "nonphysical_initial_flag": bool(nonphysical_initial_flag),
            "calibration_mode": calibration_mode,
            "fit_initial_concentration": bool(fit_initial_concentration),
            "RMSE_mg_L": metrics.get("rmse_mg_L"),
            "RMSE_C0": metrics.get("rmse_C0"),
            "RMSE/C0": metrics.get("rmse_C0"),
            "R2": metrics.get("r2"),
            "R_squared": metrics.get("r2"),
            "quality_label": quality,
            "use_for_calibration": classification["use_for_calibration"],
            "use_for_validation": classification["use_for_validation"],
            "exclude_from_ml_ranges": classification["exclude_from_ml_ranges"],
            "exclude_reason": classification["exclude_reason"],
            "successful_model_validation": classification["successful_model_validation"],
            "n_fit_points": metrics.get("n_fit_points"),
            "fit_start_time_min": metrics.get("fit_start_time_min"),
            "fit_start_source": metrics.get("fit_start_source"),
            "startup_transient_detected": metrics.get("startup_transient_detected"),
            "startup_end_time_min": metrics.get("startup_end_time_min"),
            "first_below_threshold_time_min": breakthrough_exp_result.first_below_threshold_time_min,
            "limited_prebreakthrough_information": metrics.get("limited_prebreakthrough_information"),
            "limited_postbreakthrough_information": metrics.get("limited_postbreakthrough_information"),
            "breakthrough_concentration_mg_L": breakthrough_exp_result.breakthrough_concentration_mg_L,
            "C_breakthrough_mg_L": breakthrough_exp_result.breakthrough_concentration_mg_L,
            "C_break_original_mg_L": original_c_break,
            "original_breakthrough_ratio": breakthrough_exp_result.original_breakthrough_ratio,
            "C_break_scientific_mg_L": breakthrough_exp_result.breakthrough_concentration_mg_L,
            "breakthrough_ratio_used": breakthrough_ratio,
            "breakthrough_ratio": breakthrough_ratio,
            "C_breakthrough_C0": breakthrough_ratio,
            "breakthrough_time_exp_min": t_break_exp,
            "breakthrough_time_model_min": t_break_model,
            "breakthrough_time_error_min": t_error,
            "breakthrough_status_exp": breakthrough_exp["status"],
            "breakthrough_status_model": breakthrough_model["status"],
            "breakthrough_bound_type_exp": breakthrough_exp_result.breakthrough_bound_type,
            "breakthrough_bound_type_model": breakthrough_model_result.breakthrough_bound_type,
            "breakthrough_time_exp_is_exact": bool(breakthrough_exp["is_exact"]),
            "breakthrough_time_model_is_exact": bool(breakthrough_model["is_exact"]),
            "breakthrough_volume_exp_ml": vb_exp,
            "breakthrough_volume_model_ml": vb_model,
            "C0_mg_L": column.c0_mg_l,
            "flow_ml_min": column.flow_ml_min,
            "flow_ml_min_assumed": column.flow_ml_min,
            "temperature_C": first_row.get("temperature_C"),
            "bed_height_cm": column.length_cm,
            "carbon_mass_g": column.mass_g,
            "column_area_cm2": column.area_cm2,
            "column_equivalent_diameter_cm": equivalent_diameter_cm,
            "bed_volume_cm3": bed_volume_cm3,
            "bed_loading_density_g_cm3": bed_loading_density,
            "n_replicates": experiment_replicate_count(rows),
            "calibration_curve_representation": (
                "mean_of_parallel_C1_C2"
                if experiment_replicate_count(rows) > 1
                else "single_run"
            ),
            "porosity": epsilon,
            "fit_n_cells": fit_n_cells,
            "plot_n_cells": plot_n_cells,
            "initial_guesses": repr(metrics.get("initial_guesses")),
            "optimizer_lower_bounds": repr(metrics.get("optimizer_lower_bounds")),
            "optimizer_upper_bounds": repr(metrics.get("optimizer_upper_bounds")),
            "optimizer_max_nfev": metrics.get("optimizer_max_nfev"),
            "optimizer_xtol": metrics.get("optimizer_xtol"),
            "optimizer_ftol": metrics.get("optimizer_ftol"),
            "optimizer_gtol": metrics.get("optimizer_gtol"),
            "optimizer_success": metrics.get("success"),
            "optimizer_status": metrics.get("optimizer_status"),
            "parameters_at_bounds": metrics.get("parameters_at_bounds"),
            "solver_method": metrics.get("solver_method"),
            "solver_rtol": metrics.get("solver_rtol"),
            "solver_atol": metrics.get("solver_atol"),
            "random_seed": SCIENTIFIC_RANDOM_SEED,
            "model_version": MODEL_VERSION,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "matplotlib_version": matplotlib.__version__,
            "figure_path": str(output_path),
        }
        rows_out.append(row_out)
        plot_payloads.append(
            {
                "experiment_id": experiment_id,
                "exp_time": exp_time,
                "exp_ratio": exp_ratio,
                "model_time": model_time,
                "model_ratio": model_ratio,
                "breakthrough_ratio": breakthrough_ratio,
                "breakthrough_exp": breakthrough_exp,
                "breakthrough_model": breakthrough_model,
                "calibration_mode": calibration_mode,
                "fit_start_time": metrics["fit_start_time_min"],
                "startup_end_time": metrics.get("startup_end_time_min"),
            }
        )

    result_df = pd.DataFrame(rows_out)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    plot_comparison(plot_payloads, comparison_path)
    return result_df


def plot_comparison(plot_payloads: list[dict], output_path: Path) -> None:
    if not plot_payloads:
        return

    n_items = len(plot_payloads)
    n_cols = 3
    n_rows = math.ceil(n_items / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 3.9 * n_rows), squeeze=False)

    for ax, payload in zip(axes.flat, plot_payloads):
        ax.plot(
            payload["exp_time"],
            payload["exp_ratio"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="экспериментальная кривая через точки",
        )
        ax.plot(
            payload["exp_time"],
            payload["exp_ratio"],
            "o",
            markersize=3,
            label="экспериментальные точки",
        )
        ax.plot(
            payload["model_time"],
            payload["model_ratio"],
            "-",
            linewidth=1.5,
            label="модельная кривая",
        )
        ax.axhline(
            payload["breakthrough_ratio"],
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"граница прорыва C/C0 = {payload['breakthrough_ratio']:.3f}",
        )
        if payload["breakthrough_exp"]["time_min"] is not None:
            ax.axvline(
                payload["breakthrough_exp"]["time_min"],
                color="tab:green",
                linestyle=":",
                linewidth=1.0,
                label=f"t прорыва, эксперимент: {payload['breakthrough_exp']['label']}",
            )
        elif payload["breakthrough_exp"]["status"] == "not_reached":
            ax.text(
                0.03,
                0.92,
                "эксп.: прорыв не достигнут",
                transform=ax.transAxes,
                fontsize=7,
                color="tab:green",
                va="top",
            )
        if payload["breakthrough_model"]["time_min"] is not None:
            ax.axvline(
                payload["breakthrough_model"]["time_min"],
                color="tab:orange",
                linestyle="-.",
                linewidth=1.0,
                label=f"t прорыва, модель: {payload['breakthrough_model']['label']}",
            )
        elif payload["breakthrough_model"]["status"] == "not_reached":
            ax.text(
                0.03,
                0.84,
                "модель: прорыв не достигнут",
                transform=ax.transAxes,
                fontsize=7,
                color="tab:orange",
                va="top",
            )
        if payload["fit_start_time"] > 0:
            ax.axvline(
                payload["fit_start_time"],
                color="tab:gray",
                linestyle="--",
                linewidth=0.9,
                label=f"начало подгонки = {payload['fit_start_time']:.0f} мин",
            )
        if payload.get("startup_end_time") is not None:
            ax.axvline(
                payload["startup_end_time"],
                color="tab:purple",
                linestyle=":",
                linewidth=0.9,
                label=f"конец startup washout = {payload['startup_end_time']:.2f} мин",
            )
        ax.set_title(
            f"{payload['experiment_id']} | {calibration_mode_label(payload['calibration_mode'])}",
            fontsize=9,
        )
        ax.set_xlabel("t, мин")
        ax.set_ylabel("C/C0")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for ax in axes.flat[n_items:]:
        ax.axis("off")

    fig.tight_layout()
    from ui_presentation import localize_figure
    localize_figure(fig, "uk")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    df = fit_all_experiments()
    print(f"Saved {OUTPUT_CSV}: {len(df)} rows")
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved comparison plot to {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
