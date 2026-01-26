"""Dataset CRUD endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from app.api.dependencies import get_database_session, get_optional_user
from app.core.exceptions import NotFoundError, ValidationError
from app.models.sql_models import Dataset
from app.models.pydantic_models import (
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset_data: DatasetCreate,
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
) -> DatasetResponse:
    """
    Create a new dataset.

    Args:
        dataset_data: Dataset creation data
        db: Database session
        current_user: Current authenticated user (optional)

    Returns:
        DatasetResponse: Created dataset
    """
    # Check if dataset with same name already exists
    result = await db.execute(
        select(Dataset).where(Dataset.name == dataset_data.name)
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise ValidationError(detail=f"Dataset with name '{dataset_data.name}' already exists")

    # Create new dataset
    new_dataset = Dataset(
        name=dataset_data.name,
        description=dataset_data.description,
        file_path=dataset_data.file_path,
        file_type=dataset_data.file_type,
        dataset_metadata=dataset_data.dataset_metadata,  # Cambiado de metadata a dataset_metadata
        created_by=current_user.get("user_id") if current_user else None,
    )

    db.add(new_dataset)
    await db.commit()
    await db.refresh(new_dataset)

    return DatasetResponse.model_validate(new_dataset)


@router.get("", response_model=List[DatasetResponse])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    is_active: bool = True,
    db: AsyncSession = Depends(get_database_session),
) -> List[DatasetResponse]:
    """
    List all datasets with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        is_active: Filter by active status
        db: Database session

    Returns:
        List[DatasetResponse]: List of datasets
    """
    result = await db.execute(
        select(Dataset)
        .where(Dataset.is_active == is_active)
        .offset(skip)
        .limit(limit)
        .order_by(Dataset.created_at.desc())
    )
    datasets = result.scalars().all()

    return [DatasetResponse.model_validate(dataset) for dataset in datasets]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_database_session),
) -> DatasetResponse:
    """
    Get a specific dataset by ID.

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        DatasetResponse: Dataset information

    Raises:
        NotFoundError: If dataset is not found
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise NotFoundError(resource="Dataset", resource_id=str(dataset_id))

    return DatasetResponse.model_validate(dataset)


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: int,
    dataset_data: DatasetUpdate,
    db: AsyncSession = Depends(get_database_session),
) -> DatasetResponse:
    """
    Update a dataset.

    Args:
        dataset_id: Dataset ID
        dataset_data: Dataset update data
        db: Database session

    Returns:
        DatasetResponse: Updated dataset

    Raises:
        NotFoundError: If dataset is not found
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise NotFoundError(resource="Dataset", resource_id=str(dataset_id))

    # Update fields
    update_data = dataset_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(dataset, field, value)

    await db.commit()
    await db.refresh(dataset)

    return DatasetResponse.model_validate(dataset)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_database_session),
) -> None:
    """
    Delete a dataset (soft delete by setting is_active=False).

    Args:
        dataset_id: Dataset ID
        db: Database session

    Raises:
        NotFoundError: If dataset is not found
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise NotFoundError(resource="Dataset", resource_id=str(dataset_id))

    # Soft delete
    dataset.is_active = False
    await db.commit()