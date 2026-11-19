"""Like-for-like diagnostics; never promotes fits into production parameters."""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd

from model import (DEFAULT_N_CELLS, SorptionParams, fit_model_to_experiment,
                   simulate_breakthrough, make_column_from_experiment,
                   interpret_breakthrough_curve)


def observation_metrics(observed, predicted):
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    if observed.shape != predicted.shape or not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require paired finite observations")
    error = predicted - observed
    total = float(np.sum((observed - observed.mean()) ** 2))
    return {"n": len(error), "RMSE_C0": float(np.sqrt(np.mean(error ** 2))),
            "MAE_C0": float(np.mean(np.abs(error))), "max_abs_error_C0": float(np.max(np.abs(error))),
            "R2": float(1 - np.sum(error ** 2) / total) if len(error) > 1 and total > 0 else None}


def fit_equilibrium_starts(rows, equilibrium_model, *, epsilon=.4,
                           fit_start=0., n_cells=DEFAULT_N_CELLS,
                           starts=None, progress=None, fit_initial_concentration=False):
    starts = starts or [((.03, .03, .01), .01), ((.01, .1, .1), .1), ((.1, .01, .001), 1.)]
    results, table = [], []
    for index, (initial, initial_b) in enumerate(starts):
        params, metrics = fit_model_to_experiment(rows, epsilon=epsilon,
            min_time_for_fit=fit_start, n_cells=n_cells,
            initial_parameters=initial, equilibrium_model=equilibrium_model,
            initial_b_L_mg=initial_b, fit_initial_concentration=fit_initial_concentration)
        row = {"equilibrium_model": equilibrium_model, "start_id": index,
               "initial_parameters": list(initial), "initial_b_L_mg": initial_b,
               **asdict(params), "RMSE_C0": metrics["rmse_C0"], "R2": metrics["r2"],
               "cost": metrics["cost"], "success": metrics["success"],
               "nfev": metrics["optimizer_nfev"], "at_bounds": metrics["parameters_at_bounds"],
               "active_bounds": metrics["optimizer_active_bounds"], "n_cells": n_cells,
               "n_fit_points": metrics["n_fit_points"]}
        results.append((params, metrics))
        table.append(row)
        if progress:
            progress(row)
    successful = [i for i, (_, m) in enumerate(results) if m["success"]]
    if not successful:
        raise RuntimeError(f"No converged {equilibrium_model} fits; inspect optimizer records")
    best = min(successful, key=lambda i: results[i][1]["cost"])
    for i, row in enumerate(table):
        row["selected_by_training_cost"] = i == best
    return results[best][0], results[best][1], pd.DataFrame(table)


def evaluate_curve(rows, params, *, epsilon=.4, n_cells=DEFAULT_N_CELLS):
    column = make_column_from_experiment(rows, epsilon=epsilon)
    times = np.array([r["time_min"] for r in rows], float)
    observed = np.array([r["C_out_mg_L"] for r in rows], float) / column.c0_mg_l
    predicted = simulate_breakthrough(times, column, params, n_cells=n_cells) / column.c0_mg_l
    dense = np.linspace(0., times.max(), 801)
    curve = simulate_breakthrough(dense, column, params, n_cells=n_cells) / column.c0_mg_l
    exp_bt = interpret_breakthrough_curve(times, observed)
    model_bt = interpret_breakthrough_curve(dense, curve)
    threshold_mask = (observed >= .025) & (observed <= .1)
    error = None
    # A late crossing after initial exceedance is not continuous protection.
    if exp_bt.is_exact and model_bt.is_exact and not exp_bt.startup_transient_detected:
        error = model_bt.breakthrough_time_min - exp_bt.breakthrough_time_min
    summary = {**observation_metrics(observed, predicted),
        "experimental_breakthrough": exp_bt.to_dict(), "model_breakthrough": model_bt.to_dict(),
        "breakthrough_error_min": error,
        "threshold_band_n": int(threshold_mask.sum()),
        "threshold_band_RMSE_C0": float(np.sqrt(np.mean((predicted[threshold_mask] - observed[threshold_mask]) ** 2))) if threshold_mask.any() else None}
    residuals = pd.DataFrame({"time_min": times, "observed_C0": observed,
        "predicted_C0": predicted, "residual_C0": predicted - observed})
    return summary, residuals, pd.DataFrame({"time_min": dense, "model_C0": curve})


def compare_equilibria(rows, hidden_indices, *, epsilon=.4, fit_start=0.,
                       n_cells=DEFAULT_N_CELLS, starts=None, progress=None,
                       fit_initial_concentration=False):
    """Hidden values never enter fit, automatic mask selection or start ranking."""
    ordered = sorted(rows, key=lambda row: row["time_min"])
    hidden = sorted(set(int(i) for i in hidden_indices))
    if not hidden or min(hidden) < 0 or max(hidden) >= len(ordered):
        raise ValueError("Select valid hidden-point indices")
    train = [r for i, r in enumerate(ordered) if i not in hidden]
    if sum(r["time_min"] >= fit_start for r in train) < 4:
        raise ValueError("At least four retained fitting points are required")
    output = {}
    for name in ("Henry", "Langmuir"):
        model_starts = starts
        if name == "Langmuir":
            reference = output["Henry"]["parameters"]
            model_starts = list(starts or [((.03,.03,.01),.01), ((.01,.1,.1),.1), ((.1,.01,.001),1.)])
            model_starts.append(((reference["KH_L_g"], reference["kLDF_1_min"], reference["Dax_cm2_min"]), 0.))
        fitted, metrics, stability = fit_equilibrium_starts(train, name,
            epsilon=epsilon, fit_start=fit_start, n_cells=n_cells, starts=model_starts, progress=progress,
            fit_initial_concentration=fit_initial_concentration)
        summary, residuals, curve = evaluate_curve(ordered, fitted, epsilon=epsilon, n_cells=n_cells)
        residuals["hidden"] = [i in hidden for i in range(len(ordered))]
        hidden_rows = residuals[residuals.hidden]
        train_rows = residuals[~residuals.hidden & residuals.time_min.ge(fit_start)]
        output[name] = {"parameters": asdict(fitted), "fit_metrics": metrics,
            "stability": stability, "residuals": residuals, "curve": curve,
            "summary": summary,
            "train_metrics": observation_metrics(train_rows.observed_C0, train_rows.predicted_C0),
            "hidden_metrics": observation_metrics(hidden_rows.observed_C0, hidden_rows.predicted_C0)}
    return output
