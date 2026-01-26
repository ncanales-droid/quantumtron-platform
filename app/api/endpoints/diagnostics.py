"""Diagnostic analysis endpoints."""

from typing import List, Optional, Dict, Any
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from app.api.dependencies import get_database_session, get_optional_user
from app.core.exceptions import NotFoundError, StatisticalAnalysisError
from app.models.sql_models import Dataset, DiagnosticResult
from app.models.pydantic_models import (
    DiagnosticRequest,
    DiagnosticResultResponse,
    UploadDiagnosticRequest,
)
from app.services.diagnostic_service import StatisticalDiagnosticEngine

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@router.post("/upload", response_model=DiagnosticResultResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_diagnose(
    file: UploadFile = File(...),
    target_column: Optional[str] = None,
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
) -> DiagnosticResultResponse:
    """
    Upload a file and perform complete statistical diagnosis.

    Supports CSV and Parquet files. Runs comprehensive diagnostics including:
    - Data quality assessment (0-100 score)
    - Statistical assumption validation
    - Power analysis and sample size evaluation
    - Missingness pattern detection
    - Multicollinearity detection

    Args:
        file: CSV or Parquet file to analyze
        target_column: Optional target column for regression diagnostics
        db: Database session
        current_user: Current authenticated user

    Returns:
        DiagnosticResultResponse: Complete diagnostic results

    Raises:
        HTTPException: If file type is unsupported or analysis fails
    """
    # Validate file type
    if not file.filename.lower().endswith(('.csv', '.parquet')):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Parquet files are supported"
        )
    
    file_type = file.filename.split('.')[-1].lower()
    
    # Create diagnostic engine
    diagnostic_engine = StatisticalDiagnosticEngine(random_seed=42)
    
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load and validate dataset using engine
            data = await diagnostic_engine.load_and_validate_dataset(
                tmp_file_path, 
                file_type
            )
            
            # Run complete diagnosis
            complete_report = await diagnostic_engine.run_complete_diagnosis(
                data=data,
                target_column=target_column,
            )
            
            # Add file metadata to report
            complete_report["file_metadata"] = {
                "filename": file.filename,
                "file_size_bytes": len(content),
                "file_type": file_type,
                "rows": len(data),
                "columns": len(data.columns),
                "upload_timestamp": pd.Timestamp.now().isoformat(),
            }
            
            # Create dataset record
            dataset = Dataset(
                name=file.filename,
                file_path=tmp_file_path,
                file_type=file_type,
                file_size_bytes=len(content),
                columns=list(data.columns.tolist()),
                created_by=current_user.get("user_id") if current_user else None,
            )
            
            db.add(dataset)
            await db.commit()
            await db.refresh(dataset)
            
            # Save diagnostic results
            diagnostic_result = DiagnosticResult(
                dataset_id=dataset.id,
                analysis_type="complete_upload_diagnosis",
                result_data=complete_report,
                parameters={
                    "target_column": target_column,
                    "random_seed": 42,
                    "original_filename": file.filename
                } if target_column else {
                    "random_seed": 42,
                    "original_filename": file.filename
                },
                status="completed",
                created_by=current_user.get("user_id") if current_user else None,
            )
            
            db.add(diagnostic_result)
            await db.commit()
            await db.refresh(diagnostic_result)
            
            return DiagnosticResultResponse.model_validate(diagnostic_result)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Diagnostic analysis failed: {str(e)}"
        )


@router.post("", response_model=DiagnosticResultResponse, status_code=status.HTTP_201_CREATED)
async def perform_diagnostic(
    request: DiagnosticRequest,
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
) -> DiagnosticResultResponse:
    """
    Perform statistical diagnostic analysis on a dataset.

    Args:
        request: Diagnostic analysis request
        db: Database session
        current_user: Current authenticated user (optional)

    Returns:
        DiagnosticResultResponse: Diagnostic analysis results

    Raises:
        NotFoundError: If dataset is not found
        StatisticalAnalysisError: If analysis fails
    """
    # Verify dataset exists
    result = await db.execute(
        select(Dataset).where(
            Dataset.id == request.dataset_id, Dataset.is_active == True
        )
    )
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise NotFoundError(resource="Dataset", resource_id=str(request.dataset_id))

    # Perform analysis
    diagnostic_engine = StatisticalDiagnosticEngine(random_seed=42)
    try:
        analysis_results = await diagnostic_engine.perform_custom_analysis(
            file_path=dataset.file_path,
            file_type=dataset.file_type,
            analysis_type=request.analysis_type,
            parameters=request.parameters,
        )

        # Save results to database
        diagnostic_result = DiagnosticResult(
            dataset_id=request.dataset_id,
            analysis_type=request.analysis_type,
            result_data=analysis_results,
            parameters=request.parameters,
            status="completed",
            created_by=current_user.get("user_id") if current_user else None,
        )

        db.add(diagnostic_result)
        await db.commit()
        await db.refresh(diagnostic_result)

        return DiagnosticResultResponse.model_validate(diagnostic_result)

    except Exception as e:
        # Save failed result
        diagnostic_result = DiagnosticResult(
            dataset_id=request.dataset_id,
            analysis_type=request.analysis_type,
            result_data={},
            parameters=request.parameters,
            status="failed",
            error_message=str(e),
            created_by=current_user.get("user_id") if current_user else None,
        )

        db.add(diagnostic_result)
        await db.commit()

        raise StatisticalAnalysisError(detail=f"Analysis failed: {str(e)}")


@router.get("", response_model=List[DiagnosticResultResponse])
async def list_diagnostics(
    dataset_id: Optional[int] = None,
    analysis_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_database_session),
) -> List[DiagnosticResultResponse]:
    """
    List diagnostic results with optional filtering.

    Args:
        dataset_id: Optional dataset ID to filter by
        analysis_type: Optional analysis type to filter by
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List[DiagnosticResultResponse]: List of diagnostic results
    """
    query = select(DiagnosticResult)

    if dataset_id:
        query = query.where(DiagnosticResult.dataset_id == dataset_id)
    
    if analysis_type:
        query = query.where(DiagnosticResult.analysis_type == analysis_type)

    result = await db.execute(
        query.offset(skip).limit(limit).order_by(DiagnosticResult.created_at.desc())
    )
    diagnostics = result.scalars().all()

    return [
        DiagnosticResultResponse.model_validate(diagnostic)
        for diagnostic in diagnostics
    ]


@router.get("/{diagnostic_id}", response_model=DiagnosticResultResponse)
async def get_diagnostic(
    diagnostic_id: int,
    db: AsyncSession = Depends(get_database_session),
) -> DiagnosticResultResponse:
    """
    Get a specific diagnostic result by ID.

    Args:
        diagnostic_id: Diagnostic result ID
        db: Database session

    Returns:
        DiagnosticResultResponse: Diagnostic result information

    Raises:
        NotFoundError: If diagnostic result is not found
    """
    result = await db.execute(
        select(DiagnosticResult).where(DiagnosticResult.id == diagnostic_id)
    )
    diagnostic = result.scalar_one_or_none()

    if not diagnostic:
        raise NotFoundError(
            resource="Diagnostic result", resource_id=str(diagnostic_id)
        )

    return DiagnosticResultResponse.model_validate(diagnostic)


@router.post("/complete", response_model=DiagnosticResultResponse, status_code=status.HTTP_201_CREATED)
async def perform_complete_diagnosis(
    dataset_id: int,
    target_column: Optional[str] = None,
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
) -> DiagnosticResultResponse:
    """
    Perform complete statistical diagnosis on a dataset.

    Runs comprehensive diagnostic analysis including:
    - Data quality assessment (score 0-100)
    - Statistical assumption validation (normality, homoscedasticity, etc.)
    - Power analysis and sample size evaluation
    - Missingness pattern detection (MCAR, MAR, MNAR)
    - Multicollinearity detection (VIF)

    Args:
        dataset_id: Dataset ID to analyze
        target_column: Optional target column for regression-based diagnostics
        db: Database session
        current_user: Current authenticated user (optional)

    Returns:
        DiagnosticResultResponse: Complete diagnostic analysis results

    Raises:
        NotFoundError: If dataset is not found
        StatisticalAnalysisError: If analysis fails
    """
    # Verify dataset exists
    result = await db.execute(
        select(Dataset).where(
            Dataset.id == dataset_id, Dataset.is_active == True
        )
    )
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise NotFoundError(resource="Dataset", resource_id=str(dataset_id))

    # Perform complete diagnosis
    diagnostic_engine = StatisticalDiagnosticEngine(random_seed=42)
    try:
        # Load dataset using the service method
        data = await diagnostic_engine.load_and_validate_dataset(
            dataset.file_path, 
            dataset.file_type
        )

        # Run complete diagnosis using our new method
        complete_report = await diagnostic_engine.run_complete_diagnosis(
            data=data,
            target_column=target_column,
        )

        # Enhance report with dataset metadata
        complete_report["dataset_metadata"] = {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "rows": len(data),
            "columns": len(data.columns),
            "file_type": dataset.file_type,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "statistical_engine_version": "1.0.0"
        }

        # Add statistical interpretation summary
        quality_score = complete_report.get("data_quality", {}).get("score", 0)
        
        if quality_score >= 80:
            interpretation = "Excellent data quality - suitable for advanced modeling"
            confidence = "high"
        elif quality_score >= 60:
            interpretation = "Good data quality - may require some preprocessing"
            confidence = "medium"
        else:
            interpretation = "Poor data quality - requires significant preprocessing"
            confidence = "low"
        
        complete_report["interpretation_summary"] = {
            "overall_quality": quality_score,
            "quality_tier": confidence,
            "interpretation": interpretation,
            "recommended_next_steps": [
                "Review data quality issues in the report",
                "Validate statistical assumptions before modeling",
                "Consider appropriate models based on diagnostics",
                "Check sample size adequacy for intended analysis"
            ],
            "modeling_recommendations": self._generate_modeling_recommendations(complete_report)
        }

        # Save results to database
        diagnostic_result = DiagnosticResult(
            dataset_id=dataset_id,
            analysis_type="complete_diagnosis",
            result_data=complete_report,
            parameters={
                "target_column": target_column,
                "random_seed": 42,
                "analysis_version": "1.0.0"
            } if target_column else {
                "random_seed": 42, 
                "analysis_version": "1.0.0"
            },
            status="completed",
            created_by=current_user.get("user_id") if current_user else None,
        )

        db.add(diagnostic_result)
        await db.commit()
        await db.refresh(diagnostic_result)

        return DiagnosticResultResponse.model_validate(diagnostic_result)

    except Exception as e:
        # Save failed result
        diagnostic_result = DiagnosticResult(
            dataset_id=dataset_id,
            analysis_type="complete_diagnosis",
            result_data={},
            parameters={"target_column": target_column} if target_column else {},
            status="failed",
            error_message=str(e),
            created_by=current_user.get("user_id") if current_user else None,
        )

        db.add(diagnostic_result)
        await db.commit()

        raise StatisticalAnalysisError(detail=f"Complete diagnosis failed: {str(e)}")


@router.post("/direct-json", response_model=Dict[str, Any])
async def diagnose_json_data(
    data: Dict[str, Any],
    target_column: Optional[str] = None,
    current_user: dict = Depends(get_optional_user),
) -> Dict[str, Any]:
    """
    Perform diagnostic analysis on JSON data directly.

    Args:
        data: JSON data in dictionary format
        target_column: Optional target column for analysis

    Returns:
        dict: Diagnostic analysis results
    """
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame(data)
        
        # Initialize engine
        engine = StatisticalDiagnosticEngine(random_seed=42)
        
        # Run complete diagnosis
        report = await engine.run_complete_diagnosis(df, target_column=target_column)
        
        return {
            "success": True,
            "report": report,
            "metadata": {
                "rows": len(df),
                "columns": len(df.columns),
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"JSON analysis failed: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def diagnostic_health_check() -> Dict[str, Any]:
    """
    Health check for the diagnostic service.
    
    Returns:
        dict: Health status and capabilities
    """
    try:
        # Test with sample data
        df = pd.DataFrame({
            "test_numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
            "test_categorical": ["A", "B", "A", "B", "A"]
        })
        
        engine = StatisticalDiagnosticEngine()
        quality_score = engine.calculate_data_quality_score(df)
        
        return {
            "status": "healthy",
            "service": "Statistical Diagnostic Engine",
            "version": "1.0.0",
            "test_score": quality_score["score"],
            "capabilities": [
                "data_quality_scoring",
                "statistical_assumption_validation", 
                "sample_size_evaluation",
                "missingness_pattern_analysis",
                "power_analysis",
                "multicollinearity_detection",
                "complete_diagnostic_reporting"
            ],
            "supported_file_types": ["csv", "parquet"],
            "random_seed_support": True,
            "reproducibility": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "Statistical Diagnostic Engine"
        }


def _generate_modeling_recommendations(report: Dict[str, Any]) -> List[str]:
    """
    Generate modeling recommendations based on diagnostic results.
    
    Args:
        report: Complete diagnostic report
        
    Returns:
        List[str]: Recommended modeling approaches
    """
    recommendations = []
    
    # Get key metrics
    quality_score = report.get("data_quality", {}).get("score", 0)
    normality_passed = report.get("assumption_tests", {}).get("normality", {}).get("passed", True)
    has_multicollinearity = not report.get("assumption_tests", {}).get("multicollinearity", {}).get("passed", True)
    sample_adequate = report.get("sample_size", {}).get("adequate", False)
    
    # Generate recommendations
    if quality_score < 60:
        recommendations.append("Consider extensive data preprocessing before modeling")
    
    if not normality_passed:
        recommendations.append("Use non-parametric models or transform data for normality")
    else:
        recommendations.append("Parametric models (linear regression, ANOVA) are appropriate")
    
    if has_multicollinearity:
        recommendations.append("Use regularization (Ridge/Lasso) or feature selection")
    
    if not sample_adequate:
        recommendations.append("Consider simpler models or collect more data")
    else:
        recommendations.append("Complex models (XGBoost, Neural Networks) are feasible")
    
    # Add general recommendation
    recommendations.append("Start with baseline model and iterate based on validation")
    
    return recommendations


@router.delete("/{diagnostic_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_diagnostic(
    diagnostic_id: int,
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
) -> None:
    """
    Delete a diagnostic result (soft delete).
    
    Args:
        diagnostic_id: Diagnostic result ID to delete
        db: Database session
        current_user: Current authenticated user
        
    Raises:
        NotFoundError: If diagnostic result is not found
    """
    result = await db.execute(
        select(DiagnosticResult).where(
            DiagnosticResult.id == diagnostic_id,
            DiagnosticResult.status != "deleted"
        )
    )
    diagnostic = result.scalar_one_or_none()
    
    if not diagnostic:
        raise NotFoundError(
            resource="Diagnostic result", resource_id=str(diagnostic_id)
        )
    
    # Soft delete
    diagnostic.status = "deleted"
    diagnostic.deleted_by = current_user.get("user_id") if current_user else None
    diagnostic.deleted_at = pd.Timestamp.now()
    
    await db.commit()