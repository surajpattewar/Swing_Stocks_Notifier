import unittest

import pandas as pd

from dashboard_metrics import compute_dashboard_metrics, parse_params


class DashboardMetricsTests(unittest.TestCase):
    def test_metrics_match_backtest_definitions(self):
        df = pd.DataFrame({
            "outcome": ["target_hit", "stop_loss_hit", "timeout", "no_data"],
            "return_pct": [10.0, -5.0, 2.0, None],
        })

        metrics = compute_dashboard_metrics(df)

        self.assertEqual(metrics["signals"], 4)
        self.assertEqual(metrics["target_hits"], 1)
        self.assertEqual(metrics["stop_hits"], 1)
        self.assertEqual(metrics["target_win_rate"], 50.0)
        self.assertAlmostEqual(metrics["profitable_rate"], 200 / 3)
        self.assertAlmostEqual(metrics["avg_return"], 7 / 3)
        self.assertEqual(metrics["profit_factor"], 2.4)

    def test_invalid_params_are_tolerated(self):
        self.assertEqual(parse_params("not-json"), {})
        self.assertEqual(parse_params(None), {})


if __name__ == "__main__":
    unittest.main()
