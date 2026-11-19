"""Tests for the publishable source set, without private laboratory files."""
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from demo_data import DATA_NAME, DEMO_ID, create_demo_inputs
from ml_adsorber import predict_with_metadata, train_models
from model import (ColumnParams, SorptionParams, column_area_from_diameter,
                   interpret_breakthrough_curve, load_experiments_csv, simulate_breakthrough)

ROOT = Path(__file__).resolve().parents[1]


class PublicDemoTests(unittest.TestCase):
    def test_demo_does_not_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            create_demo_inputs(path)
            raw = (path / DATA_NAME).read_bytes()
            self.assertEqual(list(load_experiments_csv(path / DATA_NAME)), [DEMO_ID])
            with self.assertRaises(FileExistsError):
                create_demo_inputs(path)
            self.assertEqual(raw, (path / DATA_NAME).read_bytes())

    def test_streamlit_starts_and_runs_calibration_and_simulation(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            create_demo_inputs(path)
            try:
                os.chdir(path)
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertEqual(app.session_state["language"], "uk")
                app.button(key="fit_run").click().run()
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertEqual(app.session_state["last_calibration"]["experiment_id"], DEMO_ID)
                app.button(key="sim_run").click().run()
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertTrue(any(item.proto.url.endswith(".csv") for item in app.get("download_button")
                                    if item.proto.label and "CSV" in item.proto.label.upper()))
            finally:
                os.chdir(original)

    def test_mechanistic_scenarios_train_curve_surrogate(self):
        rows = []
        times = np.arange(0., 121., 10.)
        params = SorptionParams(.035, .025, .035)
        for case in range(12):
            flow = 3. + .15 * case
            height = 7. + .1 * case
            column = ColumnParams(height, column_area_from_diameter(3.), 14., .4, flow, 5.)
            ratios = simulate_breakthrough(times, column, params) / column.c0_mg_l
            self.assertTrue(interpret_breakthrough_curve(times, ratios).is_exact)
            rows.extend(dict(case_id=case, time_min=t, C_out_C0=c, flow=flow,
                             height=height, C0_mg_L=5.) for t, c in zip(times, ratios))
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            result = train_models(frame, "curve", output_dir=directory)
            self.assertTrue(result["success"], result.get("message"))
            meta = result["metadata"]
            self.assertEqual(meta["target_column"], "C_out_C0")
            ids = [set(group) for group in meta["split_case_ids"].values()]
            self.assertEqual(sum(map(len, ids)), len(set.union(*ids)))
            reloaded = joblib.load(result["model_path"])
            before = predict_with_metadata(result["best_model"], frame.iloc[:3], meta)
            self.assertTrue(np.isfinite(before).all())
            np.testing.assert_array_equal(predict_with_metadata(reloaded, frame.iloc[:3], meta), before)


if __name__ == "__main__":
    unittest.main()
