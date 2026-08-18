from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from api.auth import require_api_key
from api.middleware.audit import record_request_parts
from api.schemas import (
    PassportBatchData,
    PassportBatchItem,
    PassportBatchResponse,
    PassportResponse,
)
from services.passport.service import PassportService, get_passport_service
from shared.config import Settings, get_settings
from shared.image_io import save_upload


router = APIRouter(
    prefix="/v1",
    tags=["passport"],
    dependencies=[Depends(require_api_key)],
)


@router.post(
    "/passport",
    response_model=PassportResponse,
    summary="Extract MRZ from one passport image",
    description="Upload the data page of a passport. Returns raw MRZ lines when detected.",
    responses={
        400: {"description": "Empty upload"},
        401: {"description": "Missing or invalid API key"},
        413: {"description": "Upload too large"},
        415: {"description": "Unsupported image type"},
        422: {"description": "Invalid image content"},
        500: {"description": "OCR engine failed"},
    },
)
async def detect_passport(
    request: Request,
    service: Annotated[PassportService, Depends(get_passport_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    mrz: Annotated[UploadFile, File(description="Passport data page image")],
) -> PassportResponse:
    record_request_parts(request, "mrz", [mrz])
    with TemporaryDirectory(prefix="ocr-passport-") as directory:
        temporary_directory = Path(directory)
        image_path = await save_upload(
            mrz,
            temporary_directory,
            "mrz",
            settings,
        )
        data = await service.process(image_path=image_path)

    return PassportResponse(service="passport", data=data)


@router.post(
    "/passport/batch",
    response_model=PassportBatchResponse,
    summary="Extract MRZ from multiple passport images",
    description="Upload one or more passport data page images.",
    responses={
        400: {"description": "An uploaded image is empty"},
        401: {"description": "Missing or invalid API key"},
        413: {"description": "Too many files or upload too large"},
        415: {"description": "Unsupported image type"},
        422: {"description": "Invalid image content"},
        500: {"description": "OCR engine failed to process an image"},
    },
)
async def detect_passport_batch(
    request: Request,
    service: Annotated[PassportService, Depends(get_passport_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    mrz: Annotated[
        list[UploadFile],
        File(description="Passport data page images"),
    ],
) -> PassportBatchResponse:
    record_request_parts(request, "mrz", list(mrz))
    if len(mrz) > settings.max_batch_files:
        for upload in mrz:
            await upload.close()
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"A batch may contain at most {settings.max_batch_files} images.",
        )

    with TemporaryDirectory(prefix="ocr-passport-batch-") as directory:
        temporary_directory = Path(directory)
        image_paths: list[Path] = []
        batch_bytes = 0
        for index, upload in enumerate(mrz):
            image_path = await save_upload(
                upload,
                temporary_directory,
                f"mrz_{index}",
                settings,
            )
            batch_bytes += image_path.stat().st_size
            if batch_bytes > settings.max_batch_bytes:
                for remaining_upload in mrz[index + 1 :]:
                    await remaining_upload.close()
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Batch exceeds the configured total size limit.",
                )
            image_paths.append(image_path)
        detected_results = await service.process_many(image_paths=image_paths)

    results = [
        PassportBatchItem(
            index=index,
            file_name=Path(upload.filename or f"mrz_{index}").name,
            **detected_result,
        )
        for index, (upload, detected_result) in enumerate(
            zip(mrz, detected_results, strict=True)
        )
    ]
    return PassportBatchResponse(
        service="passport",
        data=PassportBatchData(count=len(results), results=results),
    )
