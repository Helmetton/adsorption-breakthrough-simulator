import sys
from pathlib import Path
import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud_app import DEFAULT_INPUT, calculate
from model import ColumnParams, SorptionParams, column_area_from_diameter, simulate_breakthrough


class CloudDemoTests(unittest.TestCase):
    def test_demo_curve_matches_existing_solver(self):
        curve, crossing, mass, _, balance = calculate(DEFAULT_INPUT)
        column = ColumnParams(7.5, column_area_from_diameter(3.0), 14.0, 0.40, 4.0, 5.0)
        expected = simulate_breakthrough(curve.time_min, column, SorptionParams(.035, .025, .035))
        np.testing.assert_allclose(curve.C_out_mg_dm3, expected, rtol=0, atol=1e-11)
        self.assertTrue(crossing.is_exact)
        self.assertAlmostEqual(mass, 14.0)
        self.assertLess(abs(balance["final_balance_error_percent"]), 1.0)

    def test_solver_rejects_out_of_range_request(self):
        for flow in (-1.0, 13.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                calculate((3.0, 7.5, flow, 5.0))

    def test_app_launches_without_setup_and_recalculates(self):
        app = AppTest.from_file(str(ROOT / "cloud_app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertEqual(len(app.get("download_button")), 1)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        first_time = app.metric[0].value
        app.number_input(key="demo_flow_ml_min").set_value(6.0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        new_time = app.metric[0].value
        self.assertNotEqual(new_time, first_time)
        app.selectbox(key="demo_language").select("en").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key="demo_flow_ml_min").value, 6.0)
        self.assertEqual(app.metric[0].label, "Breakthrough time")
        self.assertIn("min", app.metric[0].value)


if __name__ == "__main__":
    unittest.main()
