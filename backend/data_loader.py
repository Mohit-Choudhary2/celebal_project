"""Data Loader module for CSV, Excel, and JSON files.
Supports automated format detection, sheet selection, encoding handling,
and column name sanitation.
"""

import io
import json
from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd


class DataLoader:
    """Robust multi-format data ingestion for data analysis."""

    @staticmethod
    def load_file(
        file_input: Union[str, bytes, io.BytesIO],
        file_name: str,
        sheet_name: Optional[Union[str, int]] = 0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads a file from path or bytes into a pandas DataFrame.

        Returns (DataFrame, metadata_dict).
        """
        lower_name = file_name.lower()
        metadata: Dict[str, Any] = {
            "file_name": file_name,
            "format": "unknown",
            "sheets": [],
            "selected_sheet": None,
            "row_count": 0,
            "col_count": 0,
            "columns": [],
            "memory_usage_mb": 0.0
        }

        # Handle bytes or file-like objects
        if isinstance(file_input, bytes):
            buffer = io.BytesIO(file_input)
        elif isinstance(file_input, str):
            buffer = file_input
        else:
            buffer = file_input

        df = None

        if lower_name.endswith(".csv") or lower_name.endswith(".txt"):
            metadata["format"] = "CSV"
            # Try comma, then semicolon, then tab with fallback encodings
            for encoding in ["utf-8", "latin1", "cp1252"]:
                try:
                    if isinstance(buffer, io.BytesIO):
                        buffer.seek(0)
                    df = pd.read_csv(buffer, sep=None, engine="python", encoding=encoding)
                    break
                except Exception:
                    continue
            if df is None:
                if isinstance(buffer, io.BytesIO):
                    buffer.seek(0)
                df = pd.read_csv(buffer, encoding="utf-8", on_bad_lines="skip")

        elif lower_name.endswith((".xlsx", ".xls")):
            metadata["format"] = "Excel"
            xl = pd.ExcelFile(buffer)
            metadata["sheets"] = xl.sheet_names
            target_sheet = sheet_name if sheet_name in xl.sheet_names else xl.sheet_names[0]
            metadata["selected_sheet"] = target_sheet
            df = xl.parse(target_sheet)

        elif lower_name.endswith(".json"):
            metadata["format"] = "JSON"
            if isinstance(buffer, io.BytesIO):
                buffer.seek(0)
                content = buffer.read().decode("utf-8", errors="replace")
                data = json.loads(content)
            elif isinstance(buffer, str):
                with open(buffer, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
            else:
                data = json.load(buffer)

            # Normalize nested JSON or standard record list
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                # Check for common envelope keys like "data", "records", "items", "results"
                found = False
                for key in ["data", "records", "items", "results", "rows"]:
                    if key in data and isinstance(data[key], list):
                        df = pd.json_normalize(data[key])
                        found = True
                        break
                if not found:
                    try:
                        df = pd.DataFrame(data)
                    except Exception:
                        df = pd.json_normalize(data)
            else:
                raise ValueError(f"Unsupported JSON structure: {type(data)}")
        else:
            raise ValueError(f"Unsupported file format for: {file_name}. Supported: CSV, Excel (.xlsx, .xls), JSON")

        if df is None or df.empty:
            raise ValueError(f"The loaded file '{file_name}' contains no readable data rows.")

        # Sanitize column names: remove whitespace, normalize string headers
        df.columns = [str(c).strip() for c in df.columns]

        # Calculate metadata
        metadata["row_count"] = int(len(df))
        metadata["col_count"] = int(len(df.columns))
        metadata["columns"] = list(df.columns)
        metadata["memory_usage_mb"] = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        return df, metadata
