import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from cluster_wells import FEATURES, analyze, make_demo, prepare_wells, read_well_csv, report_text, save_outputs


class ClusteringTests(unittest.TestCase):
    def setUp(self):
        self.data = make_demo()

    def test_recovers_known_demo_groups_per_well(self):
        result = analyze(self.data)
        self.assertEqual(len(result.wells), 90)
        self.assertEqual(len(result.profiles), 3)
        self.assertEqual(result.profiles.n_wells.sum(), 90)
        self.assertGreater(adjusted_rand_score(np.repeat(range(3), 30), result.labels), .95)

    def test_aggregation_and_missing_entire_well_feature(self):
        self.data.loc[self.data.Well_ID.eq("DEMO_001"), "Porosity_pct"] = np.nan
        self.data.loc[4, "Oil_Rate_bpd"] = np.nan
        result = analyze(self.data)
        self.assertEqual(result.quality["missing_well_medians"]["Porosity_pct"], 1)
        self.assertTrue(np.isfinite(result.scaled).all())
        self.assertAlmostEqual(result.imputed.loc["DEMO_001", "Porosity_pct"],
                               result.wells.Porosity_pct.median())
        self.assertAlmostEqual(result.wells.loc["DEMO_002", "Oil_Rate_bpd"],
                               self.data.loc[self.data.Well_ID.eq("DEMO_002"), "Oil_Rate_bpd"].median())

    def test_future_columns_and_row_order_do_not_change_groups(self):
        original = analyze(self.data)
        self.data["Future_Oil_Rate_bpd"] = np.arange(len(self.data)) * 10000
        modified = analyze(self.data.sample(frac=1, random_state=7))
        np.testing.assert_array_equal(original.labels, modified.labels)
        with self.assertRaisesRegex(ValueError, "будущие"):
            analyze(self.data, ["Oil_Rate_bpd", "Future_Oil_Rate_bpd"])

    def test_rejects_invalid_values(self):
        cases = [("Porosity_pct", 101), ("Oil_Rate_bpd", -1),
                 ("Depth_m", np.inf), ("Depth_m", "bad")]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                data = self.data.copy()
                data[column] = data[column].astype(object)
                data.loc[0, column] = value
                with self.assertRaises(ValueError):
                    prepare_wells(data, FEATURES)

    def test_rejects_empty_features_and_missing_ids(self):
        data = self.data.copy()
        data["Porosity_pct"] = np.nan
        with self.assertRaisesRegex(ValueError, "пустые признаки"):
            analyze(data)
        data = self.data.copy()
        data.loc[0, "Well_ID"] = " "
        with self.assertRaisesRegex(ValueError, "пустым"):
            analyze(data)

    def test_rejects_duplicates_and_fully_missing_well(self):
        with self.assertRaisesRegex(ValueError, "дубли"):
            analyze(pd.concat([self.data, self.data.iloc[[0]]]))
        self.data.loc[self.data.Well_ID.eq("DEMO_001"), FEATURES] = np.nan
        with self.assertRaisesRegex(ValueError, "без единого"):
            analyze(self.data)

    def test_rejects_same_well_id_on_different_fields(self):
        self.data.loc[0, "Field_ID"] = "ANOTHER_FIELD"
        with self.assertRaisesRegex(ValueError, "несколькими Field_ID"):
            prepare_wells(self.data, FEATURES)
        self.data.loc[0, "Field_ID"] = " DEMO "
        wells, quality = prepare_wells(self.data, FEATURES)
        self.assertEqual(len(wells), 90)
        self.assertEqual(wells.loc["DEMO_001", "Field_ID"], "DEMO")
        self.assertEqual(quality["wells_with_multiple_fields"], 0)

    def test_csv_preserves_literal_ids_and_numeric_missing_values(self):
        csv = ("Well_ID;Field_ID;Oil_Rate_bpd;Water_Cut_pct\n"
               "NA;NA;10,5;90\nNULL;NULL;NA;80\n0001;00;1000;10\nN/A;A;1200;20\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wells.csv"
            path.write_text(csv, encoding="utf-8")
            data = read_well_csv(path, sep=";", decimal=",")
            self.assertEqual(data.Well_ID.tolist(), ["NA", "NULL", "0001", "N/A"])
            self.assertEqual(data.Field_ID.tolist(), ["NA", "NULL", "00", "A"])
            self.assertEqual(data.Oil_Rate_bpd.iloc[0], 10.5)
            self.assertTrue(pd.isna(data.Oil_Rate_bpd.iloc[1]))
            result = analyze(data, ["Oil_Rate_bpd", "Water_Cut_pct"])
            self.assertEqual(set(result.wells.index), {"NA", "NULL", "0001", "N/A"})
            self.assertTrue(np.isfinite(result.scaled).all())
            path.write_text(csv + ";A;1;1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "пустым"):
                analyze(read_well_csv(path, sep=";", decimal=","), ["Oil_Rate_bpd", "Water_Cut_pct"])

    def test_csv_literal_field_ids_cannot_hide_a_conflict(self):
        csv = ("Well_ID,Field_ID,Oil_Rate_bpd,Water_Cut_pct\n"
               "A,NA,10,90\nA,NULL,20,80\nB,FIELD,1000,10\nC,FIELD,1200,20\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wells.csv"
            path.write_text(csv, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "несколькими Field_ID"):
                analyze(read_well_csv(path), ["Oil_Rate_bpd", "Water_Cut_pct"])

    def test_csv_rejects_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wells.csv"
            for duplicate in ("Well_ID", "Oil_Rate_bpd"):
                with self.subTest(duplicate=duplicate):
                    path.write_text(f"Well_ID,Oil_Rate_bpd,Water_Cut_pct,{duplicate}\n"
                                    "A,10,90,999\nB,100,50,888\nC,1000,10,777\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "повторяются имена"):
                        read_well_csv(path)

    def test_csv_rejects_nonrectangular_records(self):
        rows = ["A,10,20,30\nB,20,30,40\nC,30,40,50\n", "A,10,20\nB,20\nC,30,40\n"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wells.csv"
            for records in rows:
                with self.subTest(records=records):
                    path.write_text("Well_ID,Oil_Rate_bpd,Water_Cut_pct\n" + records, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "ожидалось 3 полей"):
                        read_well_csv(path)

    def test_csv_accepts_bom_quoted_delimiters_and_multiline_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wells.csv"
            for sep in (",", ";", "\t"):
                with self.subTest(sep=sep):
                    path.write_text(f"Well_ID{sep}Field_ID{sep}Oil_Rate_bpd{sep}Water_Cut_pct\n\n"
                                    f'"00{sep}1"{sep}"Field\n""One"""{sep}{sep}90\n'
                                    f"NULL{sep}{sep}10{sep}80\n", encoding="utf-8-sig")
                    data = read_well_csv(path, sep=sep)
                    self.assertEqual(data.Well_ID.tolist(), [f"00{sep}1", "NULL"])
                    self.assertEqual(data.Field_ID.tolist(), ['Field\n"One"', ""])
                    self.assertTrue(pd.isna(data.Oil_Rate_bpd.iloc[0]))
                    self.assertEqual(data.Oil_Rate_bpd.iloc[1], 10)

    def test_csv_reports_malformed_quotes_as_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wells.csv"
            path.write_text('Well_ID,Oil_Rate_bpd,Water_Cut_pct\nA,"10,90\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Некорректный CSV"):
                read_well_csv(path)

    def test_csv_requires_single_character_separator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wells.csv"
            path.write_text("Well_ID,Oil_Rate_bpd,Water_Cut_pct\nA,10,90\n", encoding="utf-8")
            for sep in ("", "||", r"\s+"):
                with self.subTest(sep=sep):
                    with self.assertRaisesRegex(ValueError, "одним символом"):
                        read_well_csv(path, sep=sep)

    def test_rejects_identifiers_and_output_metadata_as_features(self):
        for feature in ("API_Well_Number", "n_wells", "first_observation", "last_observation"):
            with self.subTest(feature=feature):
                data = self.data.copy()
                data[feature] = np.arange(len(data))
                with self.assertRaisesRegex(ValueError, "служебные столбцы"):
                    analyze(data, ["Oil_Rate_bpd", feature])

    def test_accepts_mixed_iso_dates_and_checks_normalized_duplicates(self):
        self.data.loc[1, "Timestamp"] = "2025-01-02T03:00:00+03:00"
        wells, quality = prepare_wells(self.data, FEATURES)
        self.assertEqual(wells.loc["DEMO_001", "n_observations"], 4)
        self.assertEqual(quality["period_start"], "2025-01-01 00:00:00+00:00")
        self.data.loc[1, "Timestamp"] = "2025-01-01T03:00:00+03:00"
        with self.assertRaisesRegex(ValueError, "дубли"):
            prepare_wells(self.data, FEATURES)

    def test_custom_log_features_preserve_units_and_record_method(self):
        self.data["GOR_scf_bbl"] = self.data.Gas_Rate_Mscf_d * 1000 / self.data.Oil_Rate_bpd
        features = ["Oil_Rate_bpd", "GOR_scf_bbl", "Water_Cut_pct"]
        result = analyze(self.data, features, log_features=features[:2])
        self.assertEqual(result.quality["log_features"], features[:2])
        pd.testing.assert_frame_equal(result.imputed, result.wells[features])
        self.assertTrue(np.isfinite(result.scaled).all())
        report = report_text(result, "TEST")
        self.assertIn("log(1+x) применено к признакам: `Oil_Rate_bpd`, `GOR_scf_bbl`", report)
        self.assertNotIn("Проницаемость, если выбрана", report)
        without_log = analyze(self.data, features, log_features=[])
        self.assertEqual(without_log.quality["log_features"], [])
        self.assertFalse(np.allclose(result.scaled, without_log.scaled))
        self.assertIn("Логарифмические преобразования не применялись", report_text(without_log, "TEST"))

    def test_rejects_invalid_log_features_and_negative_gor(self):
        for log_features in [["missing"], ["Depth_m", "Depth_m"], "Depth_m", [1]]:
            with self.subTest(log_features=log_features):
                with self.assertRaisesRegex(ValueError, "log-features"):
                    analyze(self.data, log_features=log_features)
        self.data["Custom"] = self.data.Oil_Rate_bpd
        self.data.loc[0, "Custom"] = -1
        # A negative observation must not be hidden by its well's positive median.
        with self.assertRaisesRegex(ValueError, "неотрицательные"):
            analyze(self.data, ["Custom", "Water_Cut_pct"], log_features=["Custom"])
        self.data["GOR_scf_bbl"] = 100.0
        self.data.loc[0, "GOR_scf_bbl"] = -1
        with self.assertRaisesRegex(ValueError, "Отрицательные"):
            analyze(self.data, ["Oil_Rate_bpd", "GOR_scf_bbl"], log_features=[])

    def test_rejects_too_few_wells_and_identical_profiles(self):
        with self.assertRaisesRegex(ValueError, "минимум три"):
            analyze(self.data.iloc[:8])
        self.data[FEATURES] = 1.0
        with self.assertRaisesRegex(ValueError, "непостоянных"):
            analyze(self.data)

    def test_caps_k_and_writes_complete_outputs(self):
        result = analyze(self.data.iloc[:12], max_k=20)
        self.assertEqual(result.scores.k.max(), 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_outputs(result, output, "TEST DATA")
            exported = pd.read_csv(output / "well_clusters.csv")
            self.assertEqual(len(exported), 3)
            self.assertTrue(exported.Well_ID.is_unique)
            self.assertIn("TEST DATA", (output / "report.md").read_text())
            for name in ["cluster_selection.png", "clusters_pca.png", "cluster_profiles.png"]:
                self.assertGreater((output / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
