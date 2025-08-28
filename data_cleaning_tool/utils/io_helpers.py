from __future__ import annotations

import io
from typing import Tuple, Dict, Any

import pandas as pd


# PUBLIC_INTERFACE
def read_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read a Streamlit uploaded file (CSV or Excel) into a DataFrame.

    Returns:
        (df, meta) where meta includes {"name": str, "type": str, "size": int, "ext": str}
    """
    filename = getattr(uploaded_file, "name", "uploaded_file")
    filetype = getattr(uploaded_file, "type", "application/octet-stream")
    size = getattr(uploaded_file, "size", None)
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    if ext == "csv":
        df = pd.read_csv(uploaded_file)
    elif ext in {"xlsx", "xls"}:
        # Read first sheet by default
        df = pd.read_excel(uploaded_file, sheet_name=0)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    meta = {"name": filename, "type": filetype, "size": size, "ext": ext}
    return df, meta


# PUBLIC_INTERFACE
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return a UTF-8 CSV bytes representation of the DataFrame."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
