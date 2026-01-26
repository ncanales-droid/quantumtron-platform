"""Statistical utility functions for diagnostic analysis."""

from typing import Dict, Any, List, Optional
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def calculate_vif(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate Variance Inflation Factor (VIF) for multicollinearity detection.
    
    Args:
        df: DataFrame with numeric features
        target_column: Optional target column to exclude from VIF calculation
        
    Returns:
        Dictionary with VIF results
    """
    warnings.filterwarnings('ignore')
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if len(numeric_cols) < 2:
        return {"vif_scores": {}, "interpretation": "Insufficient numeric columns for VIF calculation"}
    
    vif_data = {}
    for i, col in enumerate(numeric_cols):
        # Use other columns as predictors
        other_cols = [c for c in numeric_cols if c != col]
        
        if not other_cols:
            vif_data[col] = float('inf')
            continue
            
        X = df[other_cols].values
        y = df[col].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 2 or len(y_clean) < 2:
            vif_data[col] = float('inf')
            continue
            
        try:
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            r_squared = model.score(X_clean, y_clean)
            vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
            vif_data[col] = vif
        except:
            vif_data[col] = float('inf')
    
    # Interpretation
    high_vif_cols = [col for col, vif in vif_data.items() if vif > 10]
    moderate_vif_cols = [col for col, vif in vif_data.items() if 5 < vif <= 10]
    
    return {
        "vif_scores": vif_data,
        "high_vif_columns": high_vif_cols,
        "moderate_vif_columns": moderate_vif_cols,
        "interpretation": f"Found {len(high_vif_cols)} columns with high VIF (>10), {len(moderate_vif_cols)} with moderate VIF (5-10)"
    }


def calculate_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df: DataFrame with data
        
    Returns:
        Dictionary with correlation results
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return {"correlation_matrix": {}, "strong_correlations": []}
    
    corr_matrix = df[numeric_cols].corr()
    
    # Find strong correlations (absolute value > 0.7)
    strong_correlations = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                strong_correlations.append({
                    "column1": numeric_cols[i],
                    "column2": numeric_cols[j],
                    "correlation": float(corr_value)
                })
    
    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "strong_correlations": strong_correlations,
        "numeric_columns_count": len(numeric_cols)
    }


def calculate_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for numeric columns.
    
    Args:
        df: DataFrame with data
        
    Returns:
        Dictionary with descriptive statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return {"descriptive_stats": {}, "summary": "No numeric columns found"}
    
    stats_dict = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            stats_dict[col] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "25%": None,
                "50%": None,
                "75%": None,
                "max": None,
                "skewness": None,
                "kurtosis": None
            }
            continue
        
        stats_dict[col] = {
            "count": int(len(col_data)),
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "25%": float(col_data.quantile(0.25)),
            "50%": float(col_data.quantile(0.5)),
            "75%": float(col_data.quantile(0.75)),
            "max": float(col_data.max()),
            "skewness": float(col_data.skew()),
            "kurtosis": float(col_data.kurtosis())
        }
    
    return {
        "descriptive_stats": stats_dict,
        "numeric_columns_analyzed": numeric_cols
    }


def detect_outliers_iqr(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame with data
        
    Returns:
        Dictionary with outlier information
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return {"outliers": {}, "summary": "No numeric columns found"}
    
    outliers_dict = {}
    total_outliers = 0
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        
        if len(col_data) < 4:  # Need at least 4 points for IQR
            outliers_dict[col] = {
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "outlier_indices": []
            }
            continue
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            outliers_dict[col] = {
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "outlier_indices": []
            }
            continue
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        outlier_indices = col_data[outlier_mask].index.tolist()
        outlier_count = len(outlier_indices)
        
        outliers_dict[col] = {
            "outlier_count": outlier_count,
            "outlier_percentage": (outlier_count / len(col_data)) * 100,
            "outlier_indices": outlier_indices,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }
        
        total_outliers += outlier_count
    
    return {
        "outliers": outliers_dict,
        "total_outliers": total_outliers,
        "columns_with_outliers": [col for col, data in outliers_dict.items() if data["outlier_count"] > 0]
    }


def test_normality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test normality of numeric columns using Shapiro-Wilk test.
    
    Args:
        df: DataFrame with data
        
    Returns:
        Dictionary with normality test results
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return {"normality_tests": {}, "summary": "No numeric columns found"}
    
    normality_dict = {}
    normal_columns = []
    non_normal_columns = []
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        
        if len(col_data) < 3 or len(col_data) > 5000:
            # Shapiro-Wilk test requires 3-5000 samples
            normality_dict[col] = {
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "sample_size": len(col_data),
                "message": "Sample size outside valid range (3-5000) for Shapiro-Wilk test"
            }
            continue
        
        try:
            statistic, p_value = stats.shapiro(col_data)
            is_normal = p_value > 0.05  # Using 0.05 significance level
            
            normality_dict[col] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": is_normal,
                "sample_size": len(col_data)
            }
            
            if is_normal:
                normal_columns.append(col)
            else:
                non_normal_columns.append(col)
                
        except Exception as e:
            normality_dict[col] = {
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "sample_size": len(col_data),
                "message": f"Error in normality test: {str(e)}"
            }
    
    return {
        "normality_tests": normality_dict,
        "normal_columns": normal_columns,
        "non_normal_columns": non_normal_columns,
        "summary": f"{len(normal_columns)} columns appear normal, {len(non_normal_columns)} do not"
    }


def calculate_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate missing values statistics.
    
    Args:
        df: DataFrame with data
        
    Returns:
        Dictionary with missing value information
    """
    missing_dict = {}
    total_missing = 0
    total_cells = df.shape[0] * df.shape[1]
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / df.shape[0]) * 100
        
        missing_dict[col] = {
            "missing_count": int(missing_count),
            "missing_percentage": float(missing_percentage),
            "total_rows": df.shape[0]
        }
        
        total_missing += missing_count
    
    columns_with_missing = [col for col, data in missing_dict.items() if data["missing_count"] > 0]
    
    return {
        "missing_by_column": missing_dict,
        "total_missing_cells": total_missing,
        "total_missing_percentage": (total_missing / total_cells) * 100,
        "columns_with_missing": columns_with_missing,
        "summary": f"Found missing values in {len(columns_with_missing)} out of {len(df.columns)} columns"
    }