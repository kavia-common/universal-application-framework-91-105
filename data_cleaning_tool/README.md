# Data Cleaning Tool (Streamlit)

A modern, minimal Streamlit application for uploading datasets (CSV/Excel), detecting common data issues, applying basic cleaning operations, exporting the cleaned dataset, and generating a summary report.

## Features

- Upload CSV or Excel files (single sheet)
- Detect:
  - Missing values
  - Duplicate rows
  - Wrong data types (non-numeric in numeric columns, parseable dates)
  - Outliers (IQR method for numeric columns)
- Cleaning options:
  - Remove duplicates
  - Fill missing values (mean/median/mode/constant)
  - Standardize column names (lowercase, spaces to underscore, trim)
  - Convert data types (numeric/date parsing where possible)
  - Outlier handling (remove or cap using IQR)
- Export cleaned file (CSV) and summary report (Markdown download)

## Tech

- Python 3.9+
- Streamlit
- Pandas, NumPy
- Scikit-learn (for optional scaling utilities)
- PyYAML (optional future config)

## Quickstart (Local)

1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. Open the displayed local URL in your browser.

## Deploy

- Streamlit Community Cloud:
  - Point to this folder as the app root and `app.py` as the entrypoint.
  - Ensure `requirements.txt` is present.

- Heroku:
  - Add a `Procfile` with `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
  - Push to Heroku and set buildpack for Python.

## Project Structure

```
data_cleaning_tool/
├─ app.py
├─ requirements.txt
├─ Procfile
├─ .streamlit/
│  └─ config.toml
├─ utils/
│  ├─ __init__.py
│  ├─ io_helpers.py
│  └─ cleaning.py
└─ README.md
```

## Environment

- No environment variables are required by default. See `.env.example` if you plan to add configuration.

## Notes

- The app reads only the first sheet for Excel files.
- Large files may take time to process—progress is indicated where possible.
- This tool does not persist uploaded data; data lives in session during a single run.

