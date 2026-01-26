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