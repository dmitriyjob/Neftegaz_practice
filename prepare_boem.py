"""Prepare oil wellbore profiles from the pinned BOEM OGOR-A 2023 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


YEAR = 2023
ACCESSED = "2026-11-16"
SOURCE_URL = "https://www.data.boem.gov/Production/Files/ogora2023delimit.zip"
ARCHIVE_NAME = "ogora2023delimit.zip"
MEMBER_NAME = "ogora2023delimit.txt"
EXPECTED_SHA256 = "2fe009536793135af0d4f3122be4485ccc24ecf25c87562d97906557673edf51"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "boem"
RAW_COLUMNS = [
    "Lease_Number", "Completion_Name", "Production_Date", "Days_On_Prod",
    "Product_Code", "Oil_bbl", "Gas_Mcf", "Water_bbl", "API_Well_Number",
    "Well_Status_Code", "Area_Block", "Operator_Number", "Operator_Name",
    "Field_Name_Code", "Injection_Volume", "Production_Interval_Code",
    "First_Production_Date", "Unit_Agreement_Number", "Unit_Allocation_Suffix",
]
NUMERIC_COLUMNS = ["Days_On_Prod", "Oil_bbl", "Gas_Mcf", "Water_bbl", "Injection_Volume"]
COMPLETION_COLUMNS = ["Lease_Number", "Completion_Name", "Production_Interval_Code"]
BUSINESS_KEY = ["API_Well_Number", "Production_Date", *COMPLETION_COLUMNS]
OIL_PRODUCING_STATUSES = {"08", "09", "10", "19"}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_source(path: Path, expected_sha256: str = EXPECTED_SHA256) -> pd.DataFrame:
    """Verify the frozen archive before parsing the comma-delimited text."""
    content = path.read_bytes()
    actual = sha256(content)
    if actual != expected_sha256:
        raise ValueError(f"Source SHA256 mismatch: expected {expected_sha256}, got {actual}")
    with ZipFile(io.BytesIO(content)) as archive:
        if archive.namelist() != [MEMBER_NAME]:
            raise ValueError(f"Unexpected archive contents: {archive.namelist()}")
        frame = pd.read_csv(
            io.BytesIO(archive.read(MEMBER_NAME)), header=None, dtype=str,
            keep_default_na=False, encoding="utf-8-sig",
        )
    if frame.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"Expected 19 source columns, got {frame.shape[1]}")
    frame.columns = RAW_COLUMNS
    return validate_records(frame)


def validate_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the schema and preparation assumptions of the pinned 2023 case."""
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    result = frame[RAW_COLUMNS].copy()
    for column in RAW_COLUMNS:
        if result[column].isna().any():
            raise ValueError(f"Null values in {column}; source missing values must be explicit empty strings")
        result[column] = result[column].astype(str).str.strip()
    if result.empty:
        raise ValueError("Source contains no records")
    if result.duplicated().any():
        raise ValueError("Exact duplicate source records")
    if result.duplicated(BUSINESS_KEY).any():
        raise ValueError("Duplicate completion-month business key")
    if not result["API_Well_Number"].str.fullmatch(r"\d{12}").all():
        raise ValueError("API identifiers must contain exactly 12 digits")
    if not result["Production_Date"].str.fullmatch(r"2023(0[1-9]|1[0-2])").all():
        raise ValueError("Production dates must be valid YYYYMM months in 2023")
    dates = pd.to_datetime(result["Production_Date"], format="%Y%m")
    for column in NUMERIC_COLUMNS:
        if not result[column].str.fullmatch(r"\d+").all():
            raise ValueError(f"{column} must contain nonnegative integers")
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    if (result["Days_On_Prod"] > dates.dt.days_in_month).any():
        raise ValueError("Production days exceed calendar days in the reporting month")
    positive_volume = result[["Oil_bbl", "Gas_Mcf", "Water_bbl"]].sum(axis=1).gt(0)
    if (positive_volume & result["Days_On_Prod"].eq(0)).any():
        raise ValueError("Positive production volume with zero production days is unsupported by this case")
    result["Is_Active"] = result["Days_On_Prod"].gt(0) | positive_volume
    return result


def build_features(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return well features, all monthly records of selected APIs, and filter attrition.

    The oil-code restriction is applied to all records of an API, including inactive
    completions. This intentionally excludes mixed oil/gas wellbores. A positive oil
    volume alone could also identify condensate from a gas completion.
    """
    frame = validate_records(records)
    selected = set(frame["API_Well_Number"])
    attrition = []

    def retain(stage: str, eligible: set[str], reason: str) -> None:
        nonlocal selected
        before = len(selected)
        selected &= eligible
        attrition.append({
            "Stage": stage, "Wells_Before": before, "Wells_After": len(selected),
            "Wells_Excluded": before - len(selected),
            "Source_Rows_Remaining": int(frame["API_Well_Number"].isin(selected).sum()),
            "Reason": reason,
        })

    retain("00_source", selected, "All valid 12-digit API wellbores in the source")
    retain("01_positive_oil", set(frame.loc[frame["Oil_bbl"].gt(0), "API_Well_Number"]),
           "At least one month with positive oil volume")
    oil_only = frame.groupby("API_Well_Number")["Product_Code"].agg(lambda x: x.eq("O").all())
    retain("02_oil_code_only", set(oil_only.index[oil_only]),
           "Every record for the API has product code O; mixed gas/oil and unknown codes excluded")
    active = frame.loc[frame["Is_Active"]]
    valid_status = active.groupby("API_Well_Number")["Well_Status_Code"].agg(
        lambda x: x.isin(OIL_PRODUCING_STATUSES).all())
    retain("03_oil_status", set(valid_status.index[valid_status]),
           "All active records have oil-producing status 08, 09, 10 or 19")
    months = frame.groupby("API_Well_Number")["Production_Date"].nunique()
    retain("04_full_year_reporting", set(months.index[months.eq(12)]),
           "Records exist for each of the 12 calendar months in 2023")
    completions = active.drop_duplicates(["API_Well_Number", *COMPLETION_COLUMNS]).groupby(
        "API_Well_Number").size()
    retain("05_single_active_completion", set(completions.index[completions.eq(1)]),
           "One active lease/completion/interval identity per API throughout the year")
    monthly_count = active.groupby(["API_Well_Number", "Production_Date"]).size()
    max_count = monthly_count.groupby(level=0).max()
    retain("06_one_active_record_per_month", set(max_count.index[max_count.eq(1)]),
           "No overlapping active completion records within an API-month")
    oil_months = frame.loc[frame["Oil_bbl"].gt(0)].groupby("API_Well_Number")["Production_Date"].nunique()
    retain("07_at_least_nine_oil_months", set(oil_months.index[oil_months.ge(9)]),
           "Positive oil production in at least nine months; excludes intermittent/new wells")
    injecting = set(frame.loc[frame["Injection_Volume"].gt(0), "API_Well_Number"])
    retain("08_no_injection", selected - injecting,
           "No reported injection in any record; avoids mixed production/injection days")
    if not selected:
        raise ValueError("No wellbores satisfy the published selection rules")

    monthly = frame.loc[frame["API_Well_Number"].isin(selected)].sort_values(
        ["API_Well_Number", "Production_Date", *COMPLETION_COLUMNS], kind="stable").reset_index(drop=True)
    features = []
    for api, group in monthly.groupby("API_Well_Number", sort=True):
        producing = group.loc[group["Is_Active"]]
        first = producing.iloc[0]
        oil, gas, water = (int(group[column].sum()) for column in ("Oil_bbl", "Gas_Mcf", "Water_bbl"))
        days = int(producing["Days_On_Prod"].sum())
        if not (0 < days <= 365 and oil > 0):
            raise ValueError(f"Invalid annual denominator for API {api}")
        features.append({
            "Well_ID": api, "API_Well_Number": api,
            "Lease_Number": first["Lease_Number"], "Completion_Name": first["Completion_Name"],
            "Production_Interval_Code": first["Production_Interval_Code"],
            "Area_Block": first["Area_Block"], "Field_Name_Code": first["Field_Name_Code"],
            "Year": YEAR, "Reporting_Months": int(group["Production_Date"].nunique()),
            "Oil_Producing_Months": int(group.loc[group["Oil_bbl"].gt(0), "Production_Date"].nunique()),
            "Days_On_Prod": days, "Annual_Oil_bbl": oil, "Annual_Gas_Mcf": gas,
            "Annual_Water_bbl": water, "Oil_Rate_bpd": oil / days,
            "Water_Cut_pct": 100.0 * water / (oil + water),
            "GOR_scf_bbl": 1000.0 * gas / oil,
        })
    return pd.DataFrame(features), monthly, pd.DataFrame(attrition)


def download_snapshot(destination: Path) -> None:
    """Download only the verified revision; do not silently accept newer data."""
    with urlopen(SOURCE_URL, timeout=120) as response:
        content = response.read()
    actual = sha256(content)
    if actual != EXPECTED_SHA256:
        raise ValueError(f"The remote revision differs from the pinned snapshot: {actual}")
    destination.write_bytes(content)


def prepare(data_dir: Path = DEFAULT_DATA_DIR, download: bool = False) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / ARCHIVE_NAME
    if download:
        download_snapshot(archive)
    records = read_source(archive)
    features, monthly, attrition = build_features(records)
    generated = {
        "well_features.csv": features,
        "monthly_selected.csv": monthly,
        "attrition.csv": attrition,
    }
    for name, frame in generated.items():
        frame.to_csv(data_dir / name, index=False, float_format="%.10g", lineterminator="\n")
    manifest = {
        "dataset": "BOEM Oil and Gas Operations Reports, Part A (OGOR-A), 2023",
        "publisher": "Bureau of Ocean Energy Management (BOEM), U.S. Department of the Interior",
        "landing_page": "https://www.data.boem.gov/main/ogor-a.aspx",
        "download_url": SOURCE_URL,
        "data_dictionary": "https://www.data.boem.gov/Main/HtmlPage.aspx?page=ogorA",
        "code_dictionary": "https://www.data.boem.gov/Main/HtmlPage.aspx?page=ogorAFields",
        "unit_documentation": "https://www.data.boem.gov/Main/HtmlPage.aspx?page=ogorAPDF",
        "operating_status_documentation": "https://onrr.gov/document/MPRH-Appendix-H.pdf",
        "reuse_terms": "https://www.boem.gov/about-boem/disclaimer-liability-and-endorsement",
        "data_quality_notice": "https://www.data.boem.gov/Main/HtmlPage.aspx?page=Disclaimer",
        "accessed_on": ACCESSED,
        "period": {"start": "2023-01", "end": "2023-12"},
        "source_revision": "ZIP member timestamp 2026-11-01 05:39:52; byte snapshot identified by SHA256",
        "archive_name": ARCHIVE_NAME, "archive_sha256": EXPECTED_SHA256,
        "archive_bytes": archive.stat().st_size,
        "member_name": MEMBER_NAME,
        "source_rows": len(records), "source_api_wellbores": int(records["API_Well_Number"].nunique()),
        "selected_api_wellbores": len(features), "selected_monthly_rows": len(monthly),
        "observation_unit": "One 12-digit API wellbore, including sidetrack suffix, aggregated over 2023",
        "source_observation_unit": "Completion-month",
        "model_features": ["Oil_Rate_bpd", "Water_Cut_pct", "GOR_scf_bbl"],
        "formulas": {
            "Oil_Rate_bpd": "sum(Oil_bbl) / sum(Days_On_Prod of the single active completion)",
            "Water_Cut_pct": "100 * sum(Water_bbl) / (sum(Oil_bbl) + sum(Water_bbl))",
            "GOR_scf_bbl": "1000 * sum(Gas_Mcf) / sum(Oil_bbl)",
        },
        "validation": {
            "exact_duplicates": 0, "duplicate_completion_month_keys": 0,
            "invalid_api_identifiers": 0, "invalid_calendar_days": 0,
            "negative_or_missing_numeric_values": 0, "positive_volume_with_zero_days": 0,
        },
        "generated_files_sha256": {name: sha256((data_dir / name).read_bytes()) for name in generated},
    }
    (data_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true", help="Download and verify the pinned source revision")
    args = parser.parse_args()
    manifest = prepare(args.data_dir, args.download)
    print(f"Prepared {manifest['selected_api_wellbores']} oil wellbores from {manifest['source_rows']} source records")


if __name__ == "__main__":
    main()
