"""Generate public demo inputs from the model, never from laboratory data."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from model import (DEFAULT_BREAKTHROUGH_RATIO, ColumnParams, SorptionParams,
                   column_area_from_diameter, simulate_breakthrough)

DATA_NAME = "cleaned_dynamic_sorption_data.csv"
SUMMARY_NAME = "dynamic_sorption_experiment_summary.csv"
DEMO_ID = "DEMO_SYNTHETIC_MODEL_OUTPUT_NOT_MEASURED"


def demo_rows():
    diameter = 3.0
    column = ColumnParams(7.5, column_area_from_diameter(diameter), 14., .40, 4., 5.)
    params = SorptionParams(.035, .025, .035)
    times = np.arange(0., 181., 5.)
    concentrations = simulate_breakthrough(times, column, params)
    return [
        {
            "experiment_id": DEMO_ID,
            "source_sheet": "SYNTHETIC DEMO (NOT MEASURED)",
            "temperature_C": 20.,
            "bed_height_cm": column.length_cm,
            "carbon_mass_g": column.mass_g,
            "column_area_cm2": column.area_cm2,
            "column_diameter_cm": diameter,
            "flow_ml_min_assumed": column.flow_ml_min,
            "flow_source": "illustrative_demo_input",
            "C0_mg_L": column.c0_mg_l,
            "C_break_mg_L": column.c0_mg_l * DEFAULT_BREAKTHROUGH_RATIO,
            "C_breakthrough_ratio": DEFAULT_BREAKTHROUGH_RATIO,
            "time_min": float(t),
            "C_out_mg_L": float(c),
            "C_out_C0": float(c / column.c0_mg_l),
            "note": "Computed illustration; no experimental measurement.",
        }
        for t, c in zip(times, concentrations)
    ]


def create_demo_inputs(directory: Path):
    from fit_all_experiments import build_experiment_summary

    data_path = directory / DATA_NAME
    summary_path = directory / SUMMARY_NAME
    for path in (data_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing input: {path.name}")
    rows = demo_rows()
    with data_path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    buffer = io.StringIO()
    build_experiment_summary(data_path=data_path, output_csv=buffer)
    with summary_path.open("x", encoding="utf-8", newline="") as destination:
        destination.write(buffer.getvalue())


if __name__ == "__main__":
    try:
        create_demo_inputs(Path(__file__).resolve().parent)
    except FileExistsError as exc:
        raise SystemExit(f"{exc}. Run this only in a fresh public checkout.") from exc
    print(f"Created {DATA_NAME} and {SUMMARY_NAME} (synthetic illustration, not measured).")
