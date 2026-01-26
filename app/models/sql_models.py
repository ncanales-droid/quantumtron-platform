"""SQLAlchemy models for the database."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func

from app.core.database import Base


class Dataset(Base):
    """Dataset model for storing dataset metadata."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # csv, json, parquet, etc.
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    dataset_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    created_by = Column(String(255), nullable=True)

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name='{self.name}')>"


class DiagnosticResult(Base):
    """Diagnostic result model for storing statistical analysis results."""

    __tablename__ = "diagnostic_results"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, nullable=False, index=True)
    analysis_type = Column(String(100), nullable=False)  # descriptive, correlation, etc.
    result_data = Column(JSON, nullable=False)  # Statistical results as JSON
    parameters = Column(JSON, nullable=True)  # Analysis parameters
    status = Column(String(50), default="completed", nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_by = Column(String(255), nullable=True)

    def __repr__(self) -> str:
        return f"<DiagnosticResult(id={self.id}, dataset_id={self.dataset_id}, analysis_type='{self.analysis_type}')>"