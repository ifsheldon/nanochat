# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Acceptance checks; run with `uv run --no-project --script FILE`."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


SCRIPT = Path(__file__).with_name("plot_training.py")
PARTIAL_RECORD = b'{"completed_updates": 5, "loss":'


class PlotTrainingCLI(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="plot-training-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_dir = self.root / "run"
        self.output_dir = self.root / "plots"
        # Create the later attempt first so filesystem order cannot decide order.
        for name, updates, frozen in (
            ("20261027T010000Z", (3, 4), False),
            ("20261026T010000Z", (1, 2), True),
        ):
            attempt = self.run_dir / "attempts" / name
            attempt.mkdir(parents=True)
            (attempt / "invocation.json").write_text(json.dumps({
                "contract": {"training_config": {"num_iterations": 4}},
            }))
            records = [{
                "completed_updates": update,
                "loss": 5.0 - update,
                "last_batch_loss": 6.0 - update,
                "cycle_losses": [6.0 - update, 5.0 - update, 4.0 - update],
                "duration_seconds": 10.0 * update,
                "input_embedding_frozen": frozen,
            } for update in updates]
            payload = "".join(json.dumps(record) + "\n" for record in records).encode()
            if updates[-1] == 4:
                payload += PARTIAL_RECORD
            (attempt / "updates.jsonl").write_bytes(payload)

    def run_cli(self, *extra_args):
        result = subprocess.run(
            ["uv", "run", "--no-project", "--script", str(SCRIPT),
             "--run-dir", str(self.run_dir), "--output-dir", str(self.output_dir),
             "--timezone", "Asia/Singapore", *map(str, extra_args)],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ("loss_curves", "time_per_update", "cycles_1_2"):
            with self.subTest(artifact=name):
                self.assertEqual((self.output_dir / f"{name}.png").read_bytes()[:8],
                                 b"\x89PNG\r\n\x1a\n")
                svg = ET.parse(self.output_dir / f"{name}.svg")
                self.assertEqual(svg.getroot().tag, "{http://www.w3.org/2000/svg}svg")
        with (self.output_dir / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        provenance = json.loads((self.output_dir / "provenance.json").read_text())
        return rows, provenance

    def assert_rlm_history(self, rows, provenance):
        rlm = [row for row in rows if row["run"] == "rlm"]
        self.assertEqual([int(row["completed_update"]) for row in rlm], [1, 2, 3, 4])
        self.assertEqual([float(row["raw_update_mean"]) for row in rlm], [4, 3, 2, 1])
        self.assertEqual([float(row["raw_last_batch"]) for row in rlm], [5, 4, 3, 2])
        self.assertEqual([float(row["cycle_3_mean"]) for row in rlm], [3, 2, 1, 0])
        self.assertEqual([float(row["duration_seconds"]) for row in rlm], [10, 20, 30, 40])
        self.assertEqual([row["input_embedding_frozen"].lower() for row in rlm],
                         ["true", "true", "false", "false"])
        self.assertTrue({"last_batch_ema", "update_mean_ema", "cycle_1_mean", "cycle_2_mean"}
                        <= rlm[0].keys())
        # Hand-derived bias-corrected EMA values span the resume at update 3.
        for field, expected in (
            ("last_batch_ema", [5, 4.473684210526316, 3.929889298892989, 3.368711834835708]),
            ("update_mean_ema", [4, 3.473684210526316, 2.929889298892989, 2.368711834835708]),
        ):
            for row, value in zip(rlm, expected):
                with self.subTest(series=field, update=row["completed_update"]):
                    self.assertAlmostEqual(float(row[field]), value)
        self.assertEqual(provenance["rlm_updates"], 4)
        self.assertEqual(provenance["resume_after_updates"], [2])
        self.assertEqual(provenance["phase_changes"], [
            {"completed_update": 3, "input_embedding_frozen": False},
        ])
        self.assertEqual(provenance["ignored_partial_trailing_bytes"], len(PARTIAL_RECORD))

    def test_rlm_only_preserves_resumes_cycles_and_phase_changes(self):
        rows, provenance = self.run_cli("--through-update", 4)
        self.assert_rlm_history(rows, provenance)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["run"] for row in rows}, {"rlm"})
        self.assertEqual(provenance["mode"], "rlm-only")
        self.assertEqual(provenance["original_updates"], 0)
        self.assertIsNone(provenance["shared_range"])

    def test_shorter_baseline_keeps_logged_ema_and_full_rlm_history(self):
        original_log = self.root / "original.log"
        original_log.write_text(
            "step 00000/00002 (0.00%) | loss: 4.000000 | lrm: 1.00 | dt: 1000.00ms\n"
            "step 00001/00002 (50.00%) | loss: 2.000000 | lrm: 1.00 | dt: 2000.00ms\n"
        )
        rows, provenance = self.run_cli("--original-log", original_log)
        self.assert_rlm_history(rows, provenance)
        self.assertEqual(len(rows), 6)
        original = [row for row in rows if row["run"] == "original_nanochat"]
        self.assertEqual([int(row["completed_update"]) for row in original], [1, 2])
        self.assertEqual([float(row["logged_last_batch_ema"]) for row in original], [4, 2])
        self.assertEqual([float(row["last_batch_ema"]) for row in original], [4, 2])
        self.assertEqual([float(row["duration_seconds"]) for row in original], [1, 2])
        self.assertTrue(all(row["raw_update_mean"] == row["raw_last_batch"] == ""
                            for row in original))
        self.assertEqual(provenance["mode"], "comparison")
        self.assertEqual(provenance["original_updates"], 2)
        self.assertEqual(provenance["shared_range"], [1, 2])

    def test_cycle_panels_keep_rlm_history_beyond_shorter_baseline(self):
        spec = importlib.util.spec_from_file_location("plot_training", SCRIPT)
        plotting = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = plotting
        spec.loader.exec_module(plotting)
        self.addCleanup(sys.modules.pop, spec.name, None)
        self.addCleanup(plotting.plt.close, "all")
        rows, _, _, _ = plotting.read_run(self.run_dir, None)
        original = plotting.np.asarray([[1, 4, 1], [2, 2, 2]])
        observed = {}
        save_figure = plotting.save_figure

        def observe_and_save(figure, output, name, *args):
            observed[name] = [{
                "lines": {line.get_label(): tuple(line.get_xdata()) for line in axis.lines},
                "legend": [text.get_text() for text in axis.get_legend().get_texts()],
            } for axis in figure.axes]
            save_figure(figure, output, name, *args)

        plotting.save_figure = observe_and_save
        self.addCleanup(setattr, plotting, "save_figure", save_figure)
        self.output_dir.mkdir()
        plotting.render(self.output_dir, rows, original, {
            "recent_updates": 2, "shared_range": [1, 2], "resume_after_updates": [2],
            "phase_changes": [{"completed_update": 3, "input_embedding_frozen": False}],
            "subtitle": "Shorter-baseline acceptance fixture",
        })
        full, recent = observed["cycles_1_2"]
        for cycle in (1, 2):
            label = f"RLM cycle {cycle} mean EMA"
            with self.subTest(panel="full", cycle=cycle):
                self.assertEqual(full["lines"][label], (1, 2, 3, 4))
            with self.subTest(panel="recent", cycle=cycle):
                self.assertEqual(recent["lines"][label], (3, 4))
        baseline = "NanoChat last-batch EMA"
        self.assertEqual(full["lines"][baseline], (1, 2))
        self.assertEqual(recent["lines"].get(baseline, ()), ())
        self.assertNotIn(baseline, recent["legend"])
        shared = observed["loss_curves"][1]["lines"]
        self.assertEqual(shared["RLM last-batch EMA"], (1, 2))
        self.assertEqual(shared[baseline], (1, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
