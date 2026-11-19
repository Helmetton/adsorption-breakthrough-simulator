"""Import measured cells, never Excel formula caches, into the laboratory API."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
SMALL_SIGNAL_POLICY = (
    "Algebraic C=(A-b)/a is retained without clipping in fits and observation-time "
    "metrics. A<b is flagged, not treated as a physical negative concentration. "
    "The regression intercept is not a measured blank; no LOD/LOQ is inferred. "
    "Residuals in C correspond to absorbance residuals divided by the slope."
)


class MeasuredWorkbook:
    def __init__(self, path):
        self.path = Path(path)
        with zipfile.ZipFile(path) as archive:
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                strings = ["".join(n.itertext()) for n in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
            relationships = {n.get("Id"): n.get("Target") for n in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
            self.sheets = {}
            for sheet in ET.fromstring(archive.read("xl/workbook.xml")).find("s:sheets", NS):
                target = relationships[sheet.get(RID)]
                target = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
                cells = {}
                for cell in ET.fromstring(archive.read(target)).findall(".//s:c", NS):
                    value = cell.find("s:v", NS)
                    text = value.text if value is not None else None
                    if cell.get("t") == "s" and text is not None:
                        text = strings[int(text)]
                    formula = cell.find("s:f", NS)
                    cells[cell.get("r")] = (text, formula.text if formula is not None else None)
                self.sheets[sheet.get("name")] = cells

    def measured_number(self, sheet, address, allow_time_expression=False):
        text, formula = self.sheets[sheet][address]
        if formula:
            if not allow_time_expression:
                raise ValueError(f"Primary measurement {sheet}!{address} is a formula; cached values are forbidden.")
            # Two recorded times are sums (145+25); recompute, never use cache.
            def number(node):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return node.value
                if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                    return number(node.left) + (1 if isinstance(node.op, ast.Add) else -1) * number(node.right)
                raise ValueError(f"Unsupported recorded time expression: {formula}")
            text = number(ast.parse(formula, mode="eval").body)
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite primary measurement: {sheet}!{address}")
        return value


def import_workbook(path, output_dir="."):
    workbook = MeasuredWorkbook(path)
    concentrations = np.array([workbook.measured_number("25.03.26", f"{c}3") for c in "CDEFGH"])
    absorbances = np.array([workbook.measured_number("25.03.26", f"{c}5") for c in "CDEFGH"])
    a, b = np.linalg.lstsq(np.column_stack([concentrations, np.ones(6)]), absorbances, rcond=None)[0]
    r2 = 1 - np.sum((absorbances - (a * concentrations + b)) ** 2) / np.sum((absorbances - absorbances.mean()) ** 2)
    diameter = workbook.measured_number("01.04.26", "C3")
    area = math.pi * diameter ** 2 / 4
    # The 22.04 heights/masses are in the original registry, not new measurements.
    specs = [
        ("01.04.26", "C5", "B", ["C"], 10, 20, 2.5, 6, 3, None, "G2", "H2"),
        ("08.04.26", "C5", "B", ["C"], 10, 19, 6, 12, 3, "C3", "G2", "H2"),
        ("08.04.26", "C5", "B", ["C"], 24, 57, 11, 24, 3, "C3", "G3", "H3"),
        ("08.04.26", "C5", "B", ["C"], 63, 82, 8.5, 18, 3, "C3", "G4", "H4"),
        ("16.04.26", "C6", "B", ["C", "D"], 12, 49, 8.5, 18, 3, "C4", "H6", "I6"),
        ("16.04.26", "C6", "B", ["C"], 56, 93, 6, 12, 6, "C4", "H8", "I8"),
        ("22.04.26", "B5", "A", ["B"], 10, 28, 8.5, 18, 3, "B3", None, None),
        ("22.04.26", "B5", "A", ["B"], 38, 56, 8.5, 18, 6, "B3", None, None),
        ("22.04.26", "B5", "A", ["B"], 63, 81, 8.5, 18, 12, "B3", None, None),
    ]
    legacy_rows, samples = [], []
    for condition, (sheet, a0_cell, time_col, sample_cols, first, last, h, mass, flow, temp_cell, h_cell, mass_cell) in enumerate(specs, 1):
        if h_cell and workbook.measured_number(sheet, h_cell) != h:
            raise ValueError("Bed-height registry disagrees with measured workbook")
        if mass_cell and workbook.measured_number(sheet, mass_cell) != mass:
            raise ValueError("Mass registry disagrees with measured workbook")
        a0 = workbook.measured_number(sheet, a0_cell)
        c0 = (a0 - b) / a
        exp_id = f"{sheet}_{h:.1f}cm_{mass:.1f}g_{int(flow/3)}_{condition}"
        # Historical experiment 1 uses 2.5cm; other IDs use their existing decimals.
        base = dict(experiment_id=exp_id, source_sheet=sheet,
                    temperature_C=workbook.measured_number(sheet, temp_cell) if temp_cell else None,
                    adsorbent="Кокос", bed_height_cm=h, carbon_mass_g=mass,
                    column_area_cm2=area, column_diameter_cm=diameter,
                    legacy_column_area_cm2=8.0384, speed_label=f"{int(flow/3)} кр./сек",
                    flow_ml_min_assumed=flow, flow_source="assumed_drop_rate_conversion",
                    C0_mg_L=c0, C_break_mg_L=0.05*c0, C_breakthrough_ratio=0.05,
                    A0=a0, A0_cell=a0_cell, A_break=b+0.05*(a0-b),
                    calibration_slope=a, calibration_intercept=b, calibration_R2=r2,
                    geometry_source="workbook" if h_cell else "existing_condition_registry")
        for row in range(first, last + 1):
            time = workbook.measured_number(sheet, f"{time_col}{row}", allow_time_expression=True)
            observed = [workbook.measured_number(sheet, f"{col}{row}") for col in sample_cols]
            algebraic = [(value-b)/a for value in observed]
            for replicate, (value, concentration, col) in enumerate(zip(observed, algebraic, sample_cols), 1):
                samples.append({**base, "run_id": f"{exp_id}__rep{replicate}", "replicate": replicate,
                                "time_min": time, "time_cell": f"{time_col}{row}", "A": value,
                                "absorbance_cell": f"{col}{row}", "C_out_mg_L": concentration,
                                "C_out_C0": concentration/c0, "below_calibration_intercept": value < b})
            legacy_rows.append({**base, "time_min": time, "time_cell": f"{time_col}{row}",
                                "A1": observed[0], "A2": observed[1] if len(observed)>1 else None,
                                "A1_cell": f"{sample_cols[0]}{row}", "A2_cell": f"{sample_cols[1]}{row}" if len(observed)>1 else None,
                                "C1_mg_L": algebraic[0], "C2_mg_L": algebraic[1] if len(observed)>1 else None,
                                "C_out_mg_L": np.mean(algebraic), "C_out_C0": np.mean(algebraic)/c0,
                                "below_calibration_intercept": np.mean(observed)<b,
                                "note": "mean_curve_of_two_runs_not_mean_breakthrough_time" if len(observed)>1 else ""})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    legacy, normalized = pd.DataFrame(legacy_rows), pd.DataFrame(samples)
    assert len(legacy) == 208 and len(normalized) == 246 and normalized.run_id.nunique() == 10
    legacy.to_csv(output/"cleaned_dynamic_sorption_data.csv", index=False, encoding="utf-8-sig")
    normalized.to_csv(output/"laboratory_individual_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"C_mg_L": concentrations, "A_665nm": absorbances, "A_fitted": a*concentrations+b}).to_csv(output/"analytical_calibration_points.csv", index=False)
    manifest = {"excel_path": str(Path(path).resolve()), "excel_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "slope": float(a), "intercept": float(b), "R2": float(r2), "negative_individual_samples": int((normalized.C_out_mg_L<0).sum()),
                "number_of_conditions":9, "number_of_runs":10, "rows":208, "individual_samples":246,
                "small_signal_policy": SMALL_SIGNAL_POLICY, "formula_cache_used":False, "diameter_cm":diameter,
                "column_area_cm2":area, "flow_is_assumed":True}
    manifest["laboratory_csv_sha256"] = hashlib.sha256((output/"cleaned_dynamic_sorption_data.csv").read_bytes()).hexdigest()
    (output/"experimental_import_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return legacy, normalized, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    _, _, result = import_workbook(args.workbook, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
