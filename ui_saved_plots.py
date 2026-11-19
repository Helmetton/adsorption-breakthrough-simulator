"""Re-render saved calibration curves for the UI, without fitting parameters."""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import DEFAULT_BREAKTHROUGH_RATIO, SorptionParams, breakthrough_status, load_experiments_csv, make_column_from_experiment, simulate_breakthrough
from translations import tr


def _signature(path):
    path = Path(path).resolve()
    from scientific_provenance import file_hash
    return str(path), file_hash(path)


def saved_curve(experiment_id, data_path, fit_path):
    return _saved_curve(experiment_id, _signature(data_path), _signature(fit_path))


@lru_cache(maxsize=32)
def _saved_curve(experiment_id, data_signature, fit_signature):
    rows = load_experiments_csv(data_signature[0])[experiment_id]
    fits = pd.read_csv(fit_signature[0])
    fit = fits[fits.experiment_id.eq(experiment_id)].iloc[-1]
    column = make_column_from_experiment(rows, epsilon=float(fit.porosity))
    parameters = SorptionParams(KH_L_g=float(fit.KH_L_g), kLDF_1_min=float(fit.kLDF_1_min), Dax_cm2_min=float(fit.Dax_cm2_min), c_initial_mg_l=float(fit.C_initial_mg_L))
    observed_time = np.array([row["time_min"] for row in rows], dtype=float)
    observed_ratio = np.array([row["C_out_mg_L"] for row in rows], dtype=float) / column.c0_mg_l
    # Identical forward-plot grid and spatial resolution to the saved fit report.
    time = np.linspace(0, float(observed_time.max()), 260)
    ratio = simulate_breakthrough(time, column, parameters, n_cells=int(fit.plot_n_cells)) / column.c0_mg_l
    return {"time": time, "ratio": ratio, "observed_time": observed_time, "observed_ratio": observed_ratio, "fit": fit.to_dict()}


def _draw(ax, experiment_id, curve, lang):
    ax.plot(curve["observed_time"], curve["observed_ratio"], "o-", color="0.55", label=tr(lang, "experimental_points"))
    ax.plot(curve["time"], curve["ratio"], color="tab:orange", label=tr(lang, "model_curve"))
    ax.axhline(DEFAULT_BREAKTHROUGH_RATIO, color="tab:red", linestyle="--", label=tr(lang, "breakthrough_boundary_label").format(value=f"{DEFAULT_BREAKTHROUGH_RATIO:.2f}"))
    for times, values, key, color in [(curve["observed_time"], curve["observed_ratio"], "ui_plot_exp_tb", "0.3"), (curve["time"], curve["ratio"], "ui_plot_model_tb", "tab:blue")]:
        status = breakthrough_status(times, values, DEFAULT_BREAKTHROUGH_RATIO)
        value = status.get("time_min")
        if value is not None:
            label_relation = re.search(r"t_b\s*(<=|>=|>|<|=)", status.get("label", ""))
            relation = {"<=": "≤", ">=": "≥"}.get(label_relation[1], label_relation[1]) if label_relation else "="
            ax.axvline(value, color=color, linestyle=":", label=tr(lang, key).format(relation=relation, value=f"{value:.2f}"))
    start = curve["fit"].get("fit_start_time_min")
    if start is not None and np.isfinite(start):
        ax.axvline(start, color="tab:green", linestyle="--", label=tr(lang, "fit_start_label").format(value=f"{start:.2f}"))
    ax.set_title(experiment_id, fontsize=10)
    ax.set_xlabel(tr(lang, "time_min"))
    ax.set_ylabel("C/C₀")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)


def saved_plot_png(experiment_ids, data_path, fit_path, lang):
    ids = tuple(experiment_ids)
    return _saved_plot_png(ids, _signature(data_path), _signature(fit_path), lang)


@lru_cache(maxsize=12)
def _saved_plot_png(ids, data_signature, fit_signature, lang):
    columns = min(3, len(ids))
    rows = int(np.ceil(len(ids) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9 if len(ids) == 1 else 17, 5 * rows), squeeze=False)
    for ax, experiment_id in zip(axes.flat, ids):
        _draw(ax, experiment_id, _saved_curve(experiment_id, data_signature, fit_signature), lang)
    for ax in list(axes.flat)[len(ids):]:
        ax.set_visible(False)
    fig.tight_layout()
    output = BytesIO()
    fig.savefig(output, format="png", dpi=140)
    plt.close(fig)
    return output.getvalue()
