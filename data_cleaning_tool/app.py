import io
import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.io_helpers import read_uploaded_file, to_csv_bytes
from utils.cleaning import (
    detect_issues,
    apply_cleaning_pipeline,
    generate_summary_report_markdown,
)


# PUBLIC_INTERFACE
def app() -> None:
    """Streamlit entrypoint for the Data Cleaning Tool.

    Purpose:
        Provides a UI to upload a dataset (CSV/Excel), inspect data quality issues,
        configure cleaning operations, preview results, export cleaned data, and
        download a summary report.

    Parameters:
        None (Streamlit UI).

    Returns:
        None. Renders the Streamlit UI and provides file downloads.
    """
    st.set_page_config(
        page_title="Data Cleaning Tool",
        page_icon="🧹",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🧹 Data Cleaning Tool")
    st.caption("Upload a file, detect issues, clean, and export your dataset.")

    with st.sidebar:
        st.header("Upload")
        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
            help="Supported: .csv, .xlsx, .xls",
        )

        st.markdown("---")
        st.header("Cleaning Options")
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        fill_missing_strategy = st.selectbox(
            "Fill missing values strategy",
            options=["none", "mean", "median", "mode", "constant"],
            index=1,
            help="Strategy for numeric columns; non-numeric uses 'mode' or 'constant'.",
        )
        constant_value = None
        if fill_missing_strategy == "constant":
            constant_value = st.text_input(
                "Constant value (applied as string; will attempt numeric cast if possible)",
                value="",
            )

        standardize_columns = st.checkbox(
            "Standardize column names (lowercase, underscore, trim)", value=True
        )
        convert_types = st.checkbox(
            "Convert types (numeric/date parsing where possible)", value=True
        )

        st.markdown("---")
        st.subheader("Outlier Handling")
        outlier_action = st.selectbox(
            "Outliers in numeric columns (IQR rule)",
            options=["none", "remove", "cap"],
            index=2,
            help="Remove rows with outliers or cap them to IQR bounds.",
        )

        st.markdown("---")
        st.header("Export Options")
        export_filename = st.text_input(
            "Export filename (CSV)",
            value=f"cleaned_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

    if uploaded is None:
        st.info("Upload a CSV or Excel file to get started.")
        return

    # Read file
    try:
        df, meta = read_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    st.success(f"Loaded dataset with shape {df.shape}.")
    st.write("Preview:")
    st.dataframe(df.head(50), use_container_width=True)

    # Detect issues
    with st.expander("Detected Data Issues", expanded=True):
        issues = detect_issues(df)
        # Missing values
        st.subheader("Missing Values")
        st.write(f"Total missing values: {issues['missing']['total_missing']}")
        st.dataframe(issues["missing"]["missing_by_column"], use_container_width=True)

        # Duplicates
        st.subheader("Duplicate Rows")
        st.write(f"Total duplicate rows: {issues['duplicates']['duplicate_count']}")

        # Wrong types
        st.subheader("Type Checks")
        st.write(
            "Numeric check = values that could not be coerced to numeric. "
            "Datetime check = values that could not be parsed to datetime (if attempted)."
        )
        st.dataframe(issues["types"]["non_numeric_counts"], use_container_width=True)
        st.dataframe(issues["types"]["maybe_datetimes"], use_container_width=True)

        # Outliers
        st.subheader("Outliers (IQR)")
        st.write("Outlier counts per numeric column:")
        st.dataframe(issues["outliers"]["outlier_counts"], use_container_width=True)

    # Cleaning pipeline
    with st.spinner("Applying cleaning pipeline..."):
        cleaned_df, cleaning_details = apply_cleaning_pipeline(
            df=df,
            remove_duplicates=remove_duplicates,
            fill_missing_strategy=fill_missing_strategy,
            constant_value=constant_value,
            standardize_columns=standardize_columns,
            convert_types=convert_types,
            outlier_action=outlier_action,
        )

    st.markdown("### Cleaned Data Preview")
    st.dataframe(cleaned_df.head(50), use_container_width=True)
    st.caption(f"Cleaned shape: {cleaned_df.shape}")

    # Downloads
    csv_bytes = to_csv_bytes(cleaned_df)
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=csv_bytes,
        file_name=export_filename or "cleaned.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Summary report
    report_md = generate_summary_report_markdown(
        original_df=df,
        cleaned_df=cleaned_df,
        issues=issues,
        cleaning_details=cleaning_details,
        file_meta=meta,
    )

    st.markdown("### Summary Report")
    st.markdown(report_md)

    st.download_button(
        label="⬇️ Download Summary Report (Markdown)",
        data=report_md.encode("utf-8"),
        file_name="cleaning_summary.md",
        mime="text/markdown",
        use_container_width=True,
    )


if __name__ == "__main__":
    app()
