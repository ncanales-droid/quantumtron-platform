from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import pandas as pd
from io import StringIO
from typing import Dict, Any

from app.core.database import get_database_session
from app.core.security import get_optional_user
from app.services.diagnostic_service import perform_diagnosis

# Crear el router
router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])

logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_and_diagnose(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
):
    """Upload a file and perform automated diagnosis."""
    logger.info(f"Upload endpoint called with file: {file.filename}")
    
    try:
        # Validar tipo de archivo
        if file.content_type not in ['text/csv', 'application/vnd.ms-excel']:
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only CSV files are supported"
            )
        
        # Validar extensión del archivo
        if not file.filename.endswith('.csv'):
            logger.error(f"Invalid file extension: {file.filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must have .csv extension"
            )
        
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        # Verificar que el archivo no esté vacío
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        
        try:
            # Leer el CSV
            content_str = contents.decode('utf-8')
            df = pd.read_csv(StringIO(content_str))
            
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")
            
            # Validar que el CSV tenga datos
            if df.empty:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV file contains no data"
                )
            
            # Realizar diagnóstico
            diagnosis_result = await perform_diagnosis(df, db, current_user)
            
            # Registrar el diagnóstico en la base de datos si hay usuario
            if current_user:
                from app.models.diagnostic import DiagnosticRecord
                from datetime import datetime
                
                record = DiagnosticRecord(
                    user_id=current_user.get("id"),
                    filename=file.filename,
                    file_size=len(contents),
                    row_count=len(df),
                    column_count=len(df.columns),
                    diagnosis_type="automated",
                    result_summary=str(diagnosis_result.get("summary", "N/A")),
                    created_at=datetime.utcnow()
                )
                
                db.add(record)
                await db.commit()
                await db.refresh(record)
                
                diagnosis_result["record_id"] = record.id
            
            return {
                "status": "success",
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns),
                "diagnosis": diagnosis_result,
                "user_logged_in": bool(current_user)
            }
            
        except pd.errors.EmptyDataError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV file appears to be empty or malformed"
            )
        except pd.errors.ParserError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to parse CSV file. Please check the format."
            )
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File encoding error. Please use UTF-8 encoded CSV files."
            )
        
    except HTTPException:
        # Re-lanzar excepciones HTTP que ya manejamos
        raise
    except Exception as e:
        logger.error(f"Error in upload_and_diagnose: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/history")
async def get_diagnostic_history(
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
    limit: int = 10,
    offset: int = 0
):
    """Get diagnostic history for the current user."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        from app.models.diagnostic import DiagnosticRecord
        from sqlalchemy import select
        
        # Obtener historial del usuario
        stmt = (
            select(DiagnosticRecord)
            .where(DiagnosticRecord.user_id == current_user.get("id"))
            .order_by(DiagnosticRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await db.execute(stmt)
        records = result.scalars().all()
        
        return {
            "status": "success",
            "count": len(records),
            "history": [
                {
                    "id": record.id,
                    "filename": record.filename,
                    "file_size": record.file_size,
                    "row_count": record.row_count,
                    "column_count": record.column_count,
                    "diagnosis_type": record.diagnosis_type,
                    "result_summary": record.result_summary,
                    "created_at": record.created_at.isoformat() if record.created_at else None
                }
                for record in records
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in get_diagnostic_history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for diagnostics."""
    return {
        "status": "healthy",
        "service": "diagnostics",
        "timestamp": pd.Timestamp.now().isoformat()
    }