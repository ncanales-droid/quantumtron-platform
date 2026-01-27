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
            raise HTTPException(...)
        
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        # ... resto del código ...
        
    except Exception as e:
        logger.error(f"Error in upload_and_diagnose: {str(e)}", exc_info=True)
        raise