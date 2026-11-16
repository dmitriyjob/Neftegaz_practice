import json
import ntpath
import shlex
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import patch
from urllib.parse import urljoin, urlsplit

import matplotlib.pyplot as plt

from run_analysis import ROOT, file_link, relative_or_absolute, run


class AnalysisRunTests(unittest.TestCase):
    def test_report_links_preserve_special_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            report = parent / "results" / "report.md"
            target = parent / "Нефть #100% (данные)" / "data" / "boem" / "README.md"
            link = file_link(target, report.parent)
            self.assertEqual(urlsplit(link).fragment, "")
            self.assertNotIn(" ", link)
            self.assertEqual(urljoin(report.as_uri(), link), target.as_uri())

    def test_paths_on_different_windows_drives(self):
        project = PureWindowsPath("C:/Oil #100% Project")
        output = PureWindowsPath("D:/results")
        with patch("run_analysis.os.path.relpath", side_effect=ntpath.relpath):
            self.assertEqual(relative_or_absolute(output, project), output)
            self.assertEqual(file_link(project / "data", output), (project / "data").as_uri())

    def test_run_preserves_plot_settings_and_records_custom_output(self):
        with tempfile.TemporaryDirectory(prefix="-oil analysis ", dir=ROOT) as directory:
            output = Path(directory) / "results with spaces"
            provenance = json.loads((ROOT / "data/boem/source_manifest.json").read_text())
            with plt.rc_context({"font.size": 9, "figure.facecolor": "white"}):
                settings = dict(plt.rcParams)
                with patch("run_analysis.prepare", return_value=provenance):
                    summary = run(output)
                with self.subTest(check="plot settings"):
                    self.assertEqual(dict(plt.rcParams), settings)
            manifest = json.loads((output / "run_manifest.json").read_text())
            with self.subTest(check="recorded invocation"):
                args = shlex.split(manifest["command"])
                self.assertEqual(args, manifest["argv"])
                self.assertEqual(args[:3], ["python", "run_analysis.py", "--output"])
                self.assertFalse(args[3].startswith("-"))
                self.assertEqual((ROOT / args[3]).resolve(), output.resolve())
                self.assertEqual(summary["reproduce_command"], manifest["command"])
                self.assertIn(summary["reproduce_command"], (output / "report.md").read_text())
            self.assertEqual(summary["counts"]["selected_wells"], 1387)
            self.assertEqual(sorted(c["n_wells"] for c in summary["clusters"]), [541, 846])


if __name__ == "__main__":
    unittest.main()
