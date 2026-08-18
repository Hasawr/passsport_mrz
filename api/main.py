import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from api.middleware.audit import AuditMiddleware
from api.routes import health, passport
from api.schemas import ErrorDetail, PassportResponse
from services.passport.detector import PassportOCRProcessingError
from services.passport.service import get_passport_service
from shared.audit import get_audit_store
from shared.config import get_settings, validate_security_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings_provider = app.dependency_overrides.get(
        get_settings,
        get_settings,
    )
    settings = settings_provider()
    validate_security_settings(settings)
    app.state.settings = settings
    app.state.audit_store = get_audit_store(
        settings.audit_db_path,
        payload_dir=settings.audit_payload_dir,
    )
    app.state.audit_store.prune(
        retention_days=settings.audit_retention_days,
        max_payload_bytes=settings.audit_max_payload_bytes,
    )
    try:
        yield
    finally:
        if get_passport_service.cache_info().currsize:
            passport_service = get_passport_service()
            await asyncio.to_thread(passport_service.close)
            get_passport_service.cache_clear()


app = FastAPI(
    title="Passport MRZ Extraction Service API",
    description=(
        "Production-quality OCR API service for extracting Machine Readable Zone (MRZ) "
        "data from TD3 passport data page images.\n\n"
        "**Authentication:** Send a configured key in the `X-API-Key` header."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "passport",
            "description": "Passport MRZ OCR extraction.",
        },
        {
            "name": "health",
            "description": "Service health checks.",
        },
    ],
)
app.add_middleware(AuditMiddleware)


@app.exception_handler(PassportOCRProcessingError)
async def handle_passport_ocr_error(
    request: Request,
    exc: PassportOCRProcessingError,
) -> JSONResponse:
    del request, exc
    response = PassportResponse(
        service="passport",
        error=ErrorDetail(
            code="ocr_processing_failed",
            message="The OCR engine could not process the image.",
        ),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.model_dump(),
    )


app.include_router(health.router)
app.include_router(passport.router)
