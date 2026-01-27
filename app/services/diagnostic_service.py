"""Diagnostic service for statistical analysis of datasets."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import asyncio
import logging

from app.models.sql_models import DiagnosticResult
from app.utils.statistical_utils import (
    calculate_descriptive_stats,
    calculate_correlation_matrix,
    detect_outliers_iqr,
    test_normality,
    calculate_missing_values,
    calculate_vif
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class StatisticalDiagnosticEngine:
    """Engine for performing statistical diagnostics on datasets."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def perform_analysis(self, dataset_id: int, analysis_type: str, 
                             parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform statistical analysis on a dataset.
        
        Args:
            dataset_id: ID of the dataset to analyze
            analysis_type: Type of analysis to perform
            parameters: Optional parameters for the analysis
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # TODO: Load dataset from file_path based on dataset_id
            # For now, create a dummy dataset for testing
            df = await self._load_dataset(dataset_id)
            
            if df is None or df.empty:
                return {
                    "status": "failed",
                    "error_message": f"Could not load dataset {dataset_id} or dataset is empty"
                }
            
            # Perform the requested analysis
            if analysis_type == "descriptive":
                result_data = calculate_descriptive_stats(df)
            elif analysis_type == "correlation":
                result_data = calculate_correlation_matrix(df)
            elif analysis_type == "outlier_detection":
                result_data = detect_outliers_iqr(df)
            elif analysis_type == "normality":
                result_data = test_normality(df)
            elif analysis_type == "missing_values":
                result_data = calculate_missing_values(df)
            elif analysis_type == "multicollinearity":
                target_col = parameters.get("target_column") if parameters else None
                result_data = calculate_vif(df, target_col)
            else:
                return {
                    "status": "failed",
                    "error_message": f"Unsupported analysis type: {analysis_type}"
                }
            
            # Save results to database
            diagnostic_result = await self._save_results(
                dataset_id=dataset_id,
                analysis_type=analysis_type,
                result_data=result_data,
                parameters=parameters,
                status="completed"
            )
            
            return {
                "status": "completed",
                "result_id": diagnostic_result.id,
                "result_data": result_data
            }
            
        except Exception as e:
            logger.error(f"Error performing {analysis_type} analysis on dataset {dataset_id}: {str(e)}")
            
            # Save error to database
            try:
                await self._save_results(
                    dataset_id=dataset_id,
                    analysis_type=analysis_type,
                    result_data={},
                    parameters=parameters,
                    status="failed",
                    error_message=str(e)
                )
            except Exception as save_error:
                logger.error(f"Failed to save error result: {str(save_error)}")
            
            return {
                "status": "failed",
                "error_message": str(e)
            }
    
    async def _load_dataset(self, dataset_id: int) -> Optional[pd.DataFrame]:
        """
        Load dataset from storage.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            DataFrame with the dataset, or None if failed
        """
        try:
            # TODO: Implement actual dataset loading from file_path
            # For now, create sample data for testing
            np.random.seed(42)
            
            # Create a sample dataframe with various data types
            data = {
                'numeric_1': np.random.normal(100, 15, 100),
                'numeric_2': np.random.exponential(scale=2.0, size=100),
                'numeric_3': np.random.uniform(0, 1, 100),
                'categorical': np.random.choice(['A', 'B', 'C'], size=100),
                'date': pd.date_range('2024-01-01', periods=100, freq='D')
            }
            
            # Add some missing values
            for col in ['numeric_1', 'numeric_2']:
                missing_idx = np.random.choice(100, size=10, replace=False)
                data[col][missing_idx] = np.nan
            
            df = pd.DataFrame(data)
            
            # Add some outliers
            df.loc[95:99, 'numeric_1'] = [500, 600, 700, 800, 900]
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
            return None
    
    async def _save_results(self, dataset_id: int, analysis_type: str, 
                          result_data: Dict[str, Any], parameters: Optional[Dict[str, Any]],
                          status: str = "completed", error_message: Optional[str] = None) -> DiagnosticResult:
        """
        Save analysis results to database.
        
        Args:
            dataset_id: ID of the analyzed dataset
            analysis_type: Type of analysis performed
            result_data: Analysis results
            parameters: Analysis parameters
            status: Analysis status
            error_message: Error message if failed
            
        Returns:
            Saved DiagnosticResult object
        """
        diagnostic_result = DiagnosticResult(
            dataset_id=dataset_id,
            analysis_type=analysis_type,
            result_data=result_data,
            parameters=parameters or {},
            status=status,
            error_message=error_message,
            created_by="system"  # TODO: Replace with actual user
        )
        
        self.db_session.add(diagnostic_result)
        await self.db_session.commit()
        await self.db_session.refresh(diagnostic_result)
        
        return diagnostic_result
    
    async def get_analysis_history(self, dataset_id: int, 
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get analysis history for a dataset.
        
        Args:
            dataset_id: Dataset ID
            limit: Maximum number of results to return
            
        Returns:
            List of analysis results
        """
        result = await self.db_session.execute(
            select(DiagnosticResult)
            .where(DiagnosticResult.dataset_id == dataset_id)
            .order_by(DiagnosticResult.created_at.desc())
            .limit(limit)
        )
        
        diagnostic_results = result.scalars().all()
        
        return [
            {
                "id": dr.id,
                "analysis_type": dr.analysis_type,
                "status": dr.status,
                "created_at": dr.created_at.isoformat() if dr.created_at else None,
                "parameters": dr.parameters,
                "error_message": dr.error_message
            }
            for dr in diagnostic_results
        ]


async def perform_quick_diagnostic(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a quick diagnostic on a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with quick diagnostic results
    """
    try:
        results = {}
        
        # Basic info
        results["basic_info"] = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Missing values
        missing_results = calculate_missing_values(df)
        results["missing_values"] = missing_results
        
        # Descriptive stats for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            desc_results = calculate_descriptive_stats(df[numeric_cols])
            results["descriptive_stats"] = desc_results
        
        return results
        
    except Exception as e:
        logger.error(f"Error in quick diagnostic: {str(e)}")
        return {"error": str(e)}


async def perform_diagnosis(
    df: pd.DataFrame, 
    db: AsyncSession, 
    current_user: Dict = None
) -> Dict[str, Any]:
    """
    Perform automated diagnosis on a DataFrame for the upload endpoint.
    
    Args:
        df: Pandas DataFrame to analyze
        db: Database session
        current_user: Current user information (optional)
        
    Returns:
        Dict with diagnosis results
    """
    try:
        logger.info(f"Starting diagnosis for DataFrame shape: {df.shape}")
        
        # Use the quick diagnostic function for basic analysis
        quick_results = await perform_quick_diagnostic(df)
        
        # Format results for the diagnostics endpoint
        results = {
            "summary": f"Analyzed {len(df)} rows with {len(df.columns)} columns",
            "issues_found": 0,
            "recommendations": [],
            "warnings": [],
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "basic_statistics": quick_results.get("basic_info", {}),
            "missing_values": quick_results.get("missing_values", {}),
            "descriptive_stats": quick_results.get("descriptive_stats", {})
        }
        
        # Check for issues based on the analysis
        missing_data = quick_results.get("missing_values", {})
        if isinstance(missing_data, pd.DataFrame) and not missing_data.empty:
            total_missing = missing_data['missing_count'].sum() if 'missing_count' in missing_data.columns else 0
            if total_missing > 0:
                results["issues_found"] += 1
                results["warnings"].append(f"Found {total_missing} missing values")
                results["recommendations"].append("Consider imputation strategies for missing values")
        
        # Check for potential data quality issues
        if len(df) == 0:
            results["issues_found"] += 1
            results["warnings"].append("Dataset is empty")
            results["recommendations"].append("Upload a non-empty CSV file")
        
        if len(df.columns) == 0:
            results["issues_found"] += 1
            results["warnings"].append("Dataset has no columns")
            results["recommendations"].append("Check the CSV file format")
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            results["issues_found"] += 1
            results["warnings"].append(f"Found {duplicate_rows} duplicate rows")
            results["recommendations"].append("Consider removing duplicate rows")
            results["duplicate_rows"] = duplicate_rows
        
        # Check column names for issues
        problematic_cols = []
        for col in df.columns:
            if pd.isna(col) or str(col).strip() == "":
                problematic_cols.append(str(col))
            elif any(char in str(col) for char in ['\n', '\r', '\t']):
                problematic_cols.append(str(col))
        
        if problematic_cols:
            results["issues_found"] += 1
            results["warnings"].append(f"Found {len(problematic_cols)} columns with problematic names")
            results["recommendations"].append("Clean column names (remove empty names, newlines, tabs)")
            results["problematic_columns"] = problematic_cols
        
        # Try to detect outliers in numeric columns
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Simple outlier detection using IQR
                for col in numeric_cols[:5]:  # Limit to first 5 columns for performance
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                        
                        if len(outliers) > 0:
                            results["warnings"].append(f"Column '{col}' has {len(outliers)} potential outliers")
        except Exception as outlier_error:
            logger.warning(f"Outlier detection failed: {str(outlier_error)}")
        
        # Save results to database if user is authenticated
        if current_user:
            try:
                from datetime import datetime
                
                # Use the existing DiagnosticResult model
                result_record = DiagnosticResult(
                    dataset_id=0,  # Use 0 for uploaded files (not from database)
                    analysis_type="upload_diagnosis",
                    result_data=results,
                    parameters={"filename": "uploaded_file", "user_id": current_user.get("id")},
                    status="completed",
                    created_by=current_user.get("username", "anonymous")
                )
                
                db.add(result_record)
                await db.commit()
                await db.refresh(result_record)
                
                results["database_record_id"] = result_record.id
                
            except Exception as db_error:
                logger.error(f"Failed to save diagnostic result to database: {str(db_error)}")
                # Continue even if database save fails
        
        logger.info(f"Diagnosis completed. Issues found: {results['issues_found']}")
        return results
        
    except Exception as e:
        logger.error(f"Error in perform_diagnosis: {str(e)}", exc_info=True)
        # Return minimal results in case of error
        return {
            "summary": f"Error during analysis: {str(e)[:100]}",
            "issues_found": 1,
            "recommendations": ["Please check your data format and try again"],
            "error": str(e),
            "basic_info": {
                "row_count": len(df) if 'df' in locals() else 0,
                "column_count": len(df.columns) if 'df' in locals() else 0
            }
        }