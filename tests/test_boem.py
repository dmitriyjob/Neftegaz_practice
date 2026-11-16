import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prepare_boem import RAW_COLUMNS, build_features, read_source, validate_records


def well_records(api="177194043302", completion="A001", product="O"):
    rows = []
    for month in range(1, 13):
        row = dict.fromkeys(RAW_COLUMNS, "")
        row.update({
            "Lease_Number": "00016", "Completion_Name": completion,
            "Production_Date": f"2023{month:02d}", "Days_On_Prod": "10",
            "Product_Code": product, "Oil_bbl": "100", "Gas_Mcf": "20",
            "Water_bbl": "50", "API_Well_Number": api, "Well_Status_Code": "09",
            "Area_Block": "WD 31", "Operator_Number": "03151",
            "Operator_Name": "TEST OPERATOR", "Field_Name_Code": "WD030",
            "Injection_Volume": "0", "Production_Interval_Code": "D01",
        })
        rows.append(row)
    return rows


class TestBoemPreparation(unittest.TestCase):
    def test_ratios_use_totals_and_inactive_completions_do_not_double_days(self):
        active = well_records()
        active[0].update(Oil_bbl="1000", Water_bbl="0", Gas_Mcf="10")
        inactive = well_records(completion="A001D")
        for row in inactive:
            row.update(Days_On_Prod="0", Oil_bbl="0", Gas_Mcf="0", Water_bbl="0",
                       Well_Status_Code="12", Production_Interval_Code="D02")
        features, monthly, _ = build_features(pd.DataFrame(active + inactive))
        self.assertEqual(len(features), 1)
        well = features.iloc[0]
        self.assertEqual(well["Well_ID"], "177194043302")
        self.assertEqual(well["Days_On_Prod"], 120)
        self.assertAlmostEqual(well["Oil_Rate_bpd"], 2100 / 120)
        self.assertAlmostEqual(well["Water_Cut_pct"], 100 * 550 / 2650)
        self.assertAlmostEqual(well["GOR_scf_bbl"], 1000 * 230 / 2100)
        self.assertEqual(len(monthly), 24)
        self.assertEqual(int(monthly["Is_Active"].sum()), 12)

    def test_multiple_active_completions_are_excluded(self):
        rows = well_records()
        rows += well_records(api="177194043303", completion="B001")
        rows += well_records(api="177194043303", completion="B002")
        features, _, attrition = build_features(pd.DataFrame(rows))
        self.assertEqual(features["Well_ID"].tolist(), ["177194043302"])
        removed = attrition.set_index("Stage").loc["05_single_active_completion", "Wells_Excluded"]
        self.assertEqual(removed, 1)

    def test_gas_well_with_positive_condensate_is_not_an_oil_well(self):
        rows = well_records() + well_records(api="177194043303", product="G")
        features, _, _ = build_features(pd.DataFrame(rows))
        self.assertEqual(features["Well_ID"].tolist(), ["177194043302"])

    def test_gas_status_is_rejected_even_if_product_code_is_oil(self):
        other = well_records(api="177194043303")
        other[0]["Well_Status_Code"] = "11"
        features, _, attrition = build_features(pd.DataFrame(well_records() + other))
        self.assertEqual(len(features), 1)
        self.assertEqual(attrition.set_index("Stage").loc["03_oil_status", "Wells_Excluded"], 1)

    def test_combined_production_and_injection_days_are_excluded(self):
        injecting = well_records(api="177194043303")
        injecting[0].update(Injection_Volume="100", Well_Status_Code="08")
        features, _, attrition = build_features(pd.DataFrame(well_records() + injecting))
        self.assertEqual(features["Well_ID"].tolist(), ["177194043302"])
        self.assertEqual(attrition.set_index("Stage").loc["08_no_injection", "Wells_Excluded"], 1)

    def test_missing_reporting_month_and_intermittent_well_are_excluded(self):
        incomplete = well_records(api="177194043303")[:-1]
        intermittent = well_records(api="177194043304")
        for row in intermittent[8:]:
            row.update(Days_On_Prod="0", Oil_bbl="0", Gas_Mcf="0", Water_bbl="0", Well_Status_Code="12")
        features, _, attrition = build_features(pd.DataFrame(well_records() + incomplete + intermittent))
        self.assertEqual(len(features), 1)
        counts = attrition.set_index("Stage")["Wells_Excluded"]
        self.assertEqual(counts["04_full_year_reporting"], 1)
        self.assertEqual(counts["07_at_least_nine_oil_months"], 1)

    def test_source_quality_errors_are_not_silently_fixed(self):
        mutations = [
            ("API_Well_Number", "123", "12 digits"),
            ("Production_Date", "202413", "2023"),
            ("Days_On_Prod", "32", "calendar"),
            ("Days_On_Prod", "0", "zero production days"),
            ("Oil_bbl", "-1", "nonnegative"),
        ]
        for column, value, message in mutations:
            with self.subTest(column=column, value=value):
                rows = well_records()
                rows[0][column] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_records(pd.DataFrame(rows))
        rows = well_records()
        rows[1]["Days_On_Prod"] = "29"  # February 2023 has 28 days.
        with self.assertRaisesRegex(ValueError, "calendar"):
            validate_records(pd.DataFrame(rows))

    def test_duplicate_records_and_conflicting_business_keys_fail(self):
        rows = well_records()
        with self.assertRaisesRegex(ValueError, "Exact duplicate"):
            validate_records(pd.DataFrame(rows + [rows[0].copy()]))
        conflict = rows[0].copy()
        conflict["Oil_bbl"] = "101"
        with self.assertRaisesRegex(ValueError, "business key"):
            validate_records(pd.DataFrame(rows + [conflict]))

    def test_archive_integrity_checked_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.zip"
            path.write_bytes(b"changed snapshot")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                read_source(path)

    def test_frozen_official_snapshot_preserves_the_verified_population(self):
        archive = Path(__file__).resolve().parents[1] / "data" / "boem" / "ogora2023delimit.zip"
        self.assertEqual(
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            "2fe009536793135af0d4f3122be4485ccc24ecf25c87562d97906557673edf51",
        )
        records = read_source(archive)
        self.assertEqual(len(records), 74986)
        features, monthly, _ = build_features(records)
        self.assertEqual(len(features), 1387)
        self.assertTrue(features["API_Well_Number"].is_unique)
        self.assertEqual(set(features["API_Well_Number"]), set(monthly["API_Well_Number"]))
        self.assertTrue(monthly["Injection_Volume"].eq(0).all())
        self.assertNotIn("177104096103", set(features["API_Well_Number"]))


if __name__ == "__main__":
    unittest.main()
