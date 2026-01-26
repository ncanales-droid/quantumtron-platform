"""Data utility functions for file handling and data processing."""

from typing import Any
import pandas as pd
import numpy as np


def load_dataset(file_path: str, file_type: str) -> pd.DataFrame:
    """
    Load a dataset from file based on file type.

    Args:
        file_path: Path to the dataset file
        file_type: Type of file (csv, json, parquet, excel)

    Returns:
        pd.DataFrame: Loaded DataFrame

    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file does not exist
    """
    file_type_lower = file_type.lower()

    if file_type_lower == "csv":
        return pd.read_csv(file_path)
    elif file_type_lower == "json":
        return pd.read_json(file_path)
    elif file_type_lower == "parquet":
        return pd.read_parquet(file_path)
    elif file_type_lower in ["xlsx", "xls", "excel"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate a DataFrame and return validation results.

    Args:
        df: DataFrame to validate

    Returns:
        dict: Validation results including shape, missing values, etc.
    """
    validation_results: dict[str, Any] = {
        "is_valid": True,
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "missing_values": {},
        "dtypes": {},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Check for missing values
    missing = df.isnull().sum()
    for col in df.columns:
        missing_count = int(missing[col])
        validation_results["missing_values"][col] = {
            "count": missing_count,
            "percentage": float(missing_count / len(df) * 100) if len(df) > 0 else 0,
        }
        validation_results["dtypes"][col] = str(df[col].dtype)

    # Check if DataFrame is empty
    if len(df) == 0:
        validation_results["is_valid"] = False
        validation_results["error"] = "DataFrame is empty"

    return validation_results


def get_dataframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Get a summary of the DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        dict: Summary information
    """
    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "memory_usage": int(df.memory_usage(deep=True).sum()),
        "column_names": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=[np.number]).columns.tolist(),
    }
