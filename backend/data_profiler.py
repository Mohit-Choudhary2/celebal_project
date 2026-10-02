"""Data Profiler and Pattern Discovery module.
Performs automated exploratory data analysis, data quality scoring,
type inference, correlation detection, and anomaly flagging.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class DataProfiler:
    """Performs deep automated profiling, data health analysis, and pattern discovery."""

    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Generates a complete statistical and structural profile of the dataset."""
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols if total_rows > 0 else 1

        # 1. Missing values and duplicates
        missing_per_col = df.isnull().sum().to_dict()
        missing_total = int(sum(missing_per_col.values()))
        missing_pct = round((missing_total / total_cells) * 100, 2)
        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0

        # Health score (100 max, deducted for missingness and duplicates)
        health_deductions = (missing_pct * 1.5) + (duplicate_pct * 2.0)
        health_score = max(0, min(100, round(100 - health_deductions, 1)))

        # 2. Column Classification
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        id_cols = []
        text_cols = []

        col_details: Dict[str, Any] = {}

        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)
            n_unique = series.nunique()
            null_count = int(series.isnull().sum())
            null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0

            # Check if likely datetime
            is_datetime = False
            if pd.api.types.is_datetime64_any_dtype(series):
                is_datetime = True
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                # Try sample parsing for datetime
                sample = series.dropna().head(20)
                if len(sample) >= 5:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            parsed = pd.to_datetime(sample, errors="coerce")
                            if parsed.notnull().mean() > 0.8:
                                is_datetime = True
                        except Exception:
                            pass

            if is_datetime:
                datetime_cols.append(col)
                col_type = "datetime"
            elif pd.api.types.is_numeric_dtype(series):
                # Check if it's an ID column (e.g. sequence 1..N or strictly unique integers)
                if n_unique == total_rows and total_rows > 10 and ("id" in col.lower() or "key" in col.lower()):
                    id_cols.append(col)
                    col_type = "id"
                else:
                    numeric_cols.append(col)
                    col_type = "numeric"
            else:
                # String / Object
                if n_unique == total_rows and total_rows > 10 and ("id" in col.lower() or "key" in col.lower()):
                    id_cols.append(col)
                    col_type = "id"
                elif n_unique <= min(50, total_rows * 0.2):
                    categorical_cols.append(col)
                    col_type = "categorical"
                else:
                    text_cols.append(col)
                    col_type = "text"

            # Column stats
            col_info: Dict[str, Any] = {
                "name": col,
                "type": col_type,
                "dtype": dtype_str,
                "unique_count": n_unique,
                "null_count": null_count,
                "null_pct": null_pct,
            }

            if col_type == "numeric":
                clean_s = series.dropna()
                if len(clean_s) > 0:
                    q1 = float(clean_s.quantile(0.25))
                    q3 = float(clean_s.quantile(0.75))
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers_count = int(((clean_s < lower_bound) | (clean_s > upper_bound)).sum())

                    col_info.update({
                        "mean": round(float(clean_s.mean()), 2),
                        "std": round(float(clean_s.std()), 2) if len(clean_s) > 1 else 0.0,
                        "median": round(float(clean_s.median()), 2),
                        "min": round(float(clean_s.min()), 2),
                        "max": round(float(clean_s.max()), 2),
                        "outliers_count": outliers_count,
                        "skewness": round(float(clean_s.skew()), 2) if len(clean_s) > 2 else 0.0
                    })
            elif col_type == "categorical":
                val_counts = series.value_counts(dropna=False).head(5).to_dict()
                col_info["top_categories"] = {str(k): int(v) for k, v in val_counts.items()}

            col_details[col] = col_info

        # 3. Correlation Analysis (Numeric columns)
        correlations: List[Dict[str, Any]] = []
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        c1 = numeric_cols[i]
                        c2 = numeric_cols[j]
                        val = corr_matrix.loc[c1, c2]
                        if not np.isnan(val):
                            strength = "strong" if abs(val) >= 0.7 else ("moderate" if abs(val) >= 0.4 else "weak")
                            correlations.append({
                                "col1": c1,
                                "col2": c2,
                                "correlation": round(float(val), 3),
                                "abs_correlation": round(abs(float(val)), 3),
                                "strength": strength,
                                "direction": "positive" if val > 0 else "negative"
                            })
                correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
            except Exception:
                pass

        # 4. Pattern & Key Insights Generation
        key_patterns: List[str] = []
        key_patterns.append(f"Dataset contains {total_rows:,} rows across {total_cols} columns with overall health score of {health_score}/100.")

        if duplicate_rows > 0:
            key_patterns.append(f"Identified {duplicate_rows:,} duplicate rows ({duplicate_pct}% of dataset) requiring de-duplication.")

        cols_with_missing = [c for c, count in missing_per_col.items() if count > 0]
        if cols_with_missing:
            top_missing = sorted(cols_with_missing, key=lambda c: missing_per_col[c], reverse=True)[:3]
            key_patterns.append(f"Missing values detected in {len(cols_with_missing)} columns. Highest missing in: " +
                                ", ".join([f"{c} ({round(missing_per_col[c]/total_rows*100, 1)}%)" for c in top_missing]))
        else:
            key_patterns.append("Data completeness is 100% with zero null or missing entries.")

        if correlations:
            top_corr = correlations[0]
            key_patterns.append(f"Highest correlation found between '{top_corr['col1']}' and '{top_corr['col2']}' "
                                f"({top_corr['direction']} r = {top_corr['correlation']}, {top_corr['strength']}).")

        # Potential Target Column detection
        candidate_targets = []
        for col in df.columns:
            lcol = col.lower()
            if any(term in lcol for term in ["churn", "target", "label", "sales", "revenue", "profit", "price", "status", "outcome", "converted", "default"]):
                candidate_targets.append(col)

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "health_score": health_score,
            "missing_total": missing_total,
            "missing_pct": missing_pct,
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": duplicate_pct,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "id_cols": id_cols,
            "text_cols": text_cols,
            "col_details": col_details,
            "correlations": correlations[:10],
            "candidate_targets": candidate_targets,
            "key_patterns": key_patterns
        }
