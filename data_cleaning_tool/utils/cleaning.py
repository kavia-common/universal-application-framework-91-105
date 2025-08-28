from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _standardize_column_names(columns):
    return (
        columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def _iqr_bounds(s: pd.Series) -> Tuple[float, float]:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


# PUBLIC_INTERFACE
def detect_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect missing values, duplicates, wrong data types, and outliers.

    Returns:
        issues dict with keys: missing, duplicates, types, outliers
    """
    issues: Dict[str, Any] = {}

    # Missing
    missing_by_column = df.isna().sum().rename("missing").to_frame()
    issues["missing"] = {
        "total_missing": int(df.isna().sum().sum()),
        "missing_by_column": missing_by_column,
    }

    # Duplicates
    dup_count = int(df.duplicated().sum())
    issues["duplicates"] = {"duplicate_count": dup_count}

    # Types
    non_numeric_counts = {}
    maybe_datetimes = {}
    for col in df.columns:
        s = df[col]
        # Try numeric coercion
        coerced_num = pd.to_numeric(s, errors="coerce")
        non_numeric_counts[col] = int(s.notna().sum() - coerced_num.notna().sum())
        # Try date parsing
        coerced_date = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        # Count entries that parsed successfully to datetime (as hint)
        maybe_datetimes[col] = int(coerced_date.notna().sum())

    issues["types"] = {
        "non_numeric_counts": pd.Series(non_numeric_counts, name="non_numeric_values")
        .to_frame()
        .sort_index(),
        "maybe_datetimes": pd.Series(maybe_datetimes, name="parseable_as_datetime")
        .to_frame()
        .sort_index(),
    }

    # Outliers (IQR per numeric col)
    outlier_counts = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            outlier_counts[col] = 0
            continue
        lower, upper = _iqr_bounds(s)
        mask = (s < lower) | (s > upper)
        outlier_counts[col] = int(mask.sum())

    issues["outliers"] = {
        "outlier_counts": pd.Series(outlier_counts, name="outliers_iqr")
        .to_frame()
        .sort_index()
    }

    return issues


# PUBLIC_INTERFACE
def apply_cleaning_pipeline(
    df: pd.DataFrame,
    remove_duplicates: bool,
    fill_missing_strategy: str = "mean",
    constant_value: Optional[str] = None,
    standardize_columns: bool = True,
    convert_types: bool = True,
    outlier_action: str = "cap",  # "none" | "remove" | "cap"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply selected cleaning operations and return cleaned df and details.

    Returns:
        (cleaned_df, details)
    """
    details: Dict[str, Any] = {}
    work = df.copy()

    # Standardize column names
    if standardize_columns:
        before_cols = list(work.columns)
        work.columns = _standardize_column_names(work.columns)
        details["standardize_columns"] = {
            "before": before_cols,
            "after": list(work.columns),
        }

    # Remove duplicates
    if remove_duplicates:
        before = len(work)
        work = work.drop_duplicates()
        details["remove_duplicates"] = {"before": before, "after": len(work)}

    # Convert types (best effort)
    if convert_types:
        converted = {}
        for col in work.columns:
            s = work[col]
            s_num = pd.to_numeric(s, errors="coerce")
            num_gain = int(s_num.notna().sum() - s.notna().sum())
            # If many values become numeric, adopt numeric where notna
            if s_num.notna().sum() >= max(3, int(0.5 * s.notna().sum())):
                work[col] = s_num
                converted[col] = "numeric"
                continue
            # Try datetime
            s_dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            if s_dt.notna().sum() >= max(3, int(0.5 * s.notna().sum())):
                work[col] = s_dt
                converted[col] = "datetime"
        details["convert_types"] = converted

    # Fill missing
    strategy = (fill_missing_strategy or "none").lower()
    fill_info = {}
    if strategy != "none":
        for col in work.columns:
            s = work[col]
            if s.isna().sum() == 0:
                continue
            to_fill = None
            if pd.api.types.is_numeric_dtype(s):
                if strategy == "mean":
                    to_fill = s.mean()
                elif strategy == "median":
                    to_fill = s.median()
                elif strategy == "mode":
                    m = s.mode(dropna=True)
                    to_fill = m.iloc[0] if not m.empty else 0
                elif strategy == "constant":
                    # try numeric cast
                    try:
                        to_fill = float(constant_value) if constant_value is not None else 0.0
                    except Exception:
                        to_fill = 0.0
            else:
                if strategy in {"mean", "median"}:
                    # mean/median don't apply to non-numeric; fall back to mode
                    strategy_eff = "mode"
                else:
                    strategy_eff = strategy
                if strategy_eff == "mode":
                    m = s.mode(dropna=True)
                    to_fill = m.iloc[0] if not m.empty else ""
                elif strategy_eff == "constant":
                    to_fill = "" if constant_value is None else constant_value
            if to_fill is not None:
                work[col] = s.fillna(to_fill)
                fill_info[col] = {"filled": int(s.isna().sum()), "value": to_fill}
    details["fill_missing"] = {"strategy": strategy, "columns": fill_info}

    # Outlier handling using IQR per numeric column
    outlier_result = {}
    if outlier_action in {"remove", "cap"}:
        numeric_cols = work.select_dtypes(include=[np.number]).columns
        if outlier_action == "remove":
            mask = pd.Series(False, index=work.index)
            for col in numeric_cols:
                s = work[col]
                if s.dropna().empty:
                    continue
                lower, upper = _iqr_bounds(s.dropna())
                mask = mask | (s < lower) | (s > upper)
            before = len(work)
            work = work[~mask]
            outlier_result["remove"] = {"removed_rows": int(mask.sum()), "before": before, "after": len(work)}
        elif outlier_action == "cap":
            caps = {}
            for col in numeric_cols:
                s = work[col]
                if s.dropna().empty:
                    continue
                lower, upper = _iqr_bounds(s.dropna())
                s_capped = s.clip(lower=lower, upper=upper)
                changes = int((s != s_capped).sum(skipna=True))
                work[col] = s_capped
                caps[col] = {"lower": float(lower), "upper": float(upper), "capped_values": changes}
            outlier_result["cap"] = caps
    details["outliers"] = {"action": outlier_action, "details": outlier_result}

    return work, details


# PUBLIC_INTERFACE
def generate_summary_report_markdown(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    issues: Dict[str, Any],
    cleaning_details: Dict[str, Any],
    file_meta: Dict[str, Any],
) -> str:
    """Build a Markdown report summarizing dataset state, issues, and performed cleaning."""
    lines = []
    lines.append(f"# Data Cleaning Summary Report")
    lines.append("")
    lines.append("## File")
    lines.append(f"- Name: {file_meta.get('name')}")
    lines.append(f"- Type: {file_meta.get('type')}")
    lines.append(f"- Size: {file_meta.get('size')} bytes")
    lines.append("")
    lines.append("## Shapes")
    lines.append(f"- Original: {original_df.shape}")
    lines.append(f"- Cleaned: {cleaned_df.shape}")
    lines.append("")
    lines.append("## Detected Issues")
    lines.append("### Missing Values")
    lines.append(f"- Total missing: {issues['missing']['total_missing']}")
    lines.append("")
    lines.append(issues["missing"]["missing_by_column"].to_markdown())

    lines.append("")
    lines.append("### Duplicates")
    lines.append(f"- Duplicate rows: {issues['duplicates']['duplicate_count']}")

    lines.append("")
    lines.append("### Type Checks")
    lines.append(issues["types"]["non_numeric_counts"].to_markdown())
    lines.append("")
    lines.append(issues["types"]["maybe_datetimes"].to_markdown())

    lines.append("")
    lines.append("### Outliers (IQR)")
    lines.append(issues["outliers"]["outlier_counts"].to_markdown())

    lines.append("")
    lines.append("## Cleaning Applied")
    for key, val in cleaning_details.items():
        lines.append(f"### {key}")
        if isinstance(val, dict):
            try:
                df_like = pd.DataFrame(val)
                lines.append(df_like.to_markdown())
            except Exception:
                lines.append(f"- {val}")
        else:
            lines.append(f"- {val}")

        lines.append("")

    return "\n".join(lines)
