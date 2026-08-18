from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import asyncio
import logging
from pathlib import Path
from threading import Event, Lock

from services.passport import PassportDetectionOutput
from services.passport.detector import PassportDetector
from shared.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ServiceClosedError(RuntimeError):
    """Raised when OCR work is submitted after service shutdown."""


def serialize_passport_detection(
    result: PassportDetectionOutput,
) -> dict[str, object]:
    """Convert PassportDetectionOutput dataclass to response payload dictionary."""
    data = asdict(result)
    data["confidence"] = round(float(result.confidence), 4)
    data["mrz_details"] = data.pop("mrz_result")
    if data["mrz_details"] is not None:
        for internal_field in (
            "line_confidences",
            "quality_score",
            "is_canonical",
        ):
            data["mrz_details"].pop(internal_field, None)
    return data


class PassportService:
    """Async adapter around one or more passport PaddleOCR engines."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        use_gpu: bool | None = None,
    ) -> None:
        settings = settings or get_settings()
        detector_uses_gpu = settings.use_gpu if use_gpu is None else use_gpu
        worker_count = max(1, getattr(settings, "ocr_worker_count", 1))
        concurrent_attempts = max(
            1, getattr(settings, "ocr_concurrent_attempts", 1)
        )
        det_limit_side_len = max(
            320, getattr(settings, "ocr_det_limit_side_len", 736)
        )
        self._detectors = [
            PassportDetector(
                use_gpu=detector_uses_gpu,
                save_debug_images=settings.save_ocr_debug_images,
                concurrent_attempts=concurrent_attempts,
                det_limit_side_len=det_limit_side_len,
            )
            for _ in range(worker_count)
        ]
        self._next_detector_index = 0
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="passport-ocr",
        )
        self._lifecycle_lock = Lock()
        self._is_closed = False
        self._close_complete = Event()
        logger.info("Passport service ready with %d OCR engine(s)", worker_count)

    def _acquire_detector(self) -> PassportDetector:
        detector = self._detectors[self._next_detector_index]
        self._next_detector_index = (
            self._next_detector_index + 1
        ) % len(self._detectors)
        return detector

    async def process(
        self,
        *,
        image_path: Path,
    ) -> dict[str, object]:
        result = await self.process_detection(image_path=image_path)
        return serialize_passport_detection(result)

    async def process_detection(
        self,
        *,
        image_path: Path,
    ) -> PassportDetectionOutput:
        with self._lifecycle_lock:
            if self._is_closed:
                raise ServiceClosedError("Passport service is closed.")
            detector = self._acquire_detector()
            future = self._executor.submit(
                detector.detect_from_passport,
                image_path,
            )
        return await asyncio.wrap_future(future)

    async def process_many(
        self,
        *,
        image_paths: list[Path],
    ) -> list[dict[str, object]]:
        with self._lifecycle_lock:
            if self._is_closed:
                raise ServiceClosedError("Passport service is closed.")
        if not image_paths:
            return []

        if len(self._detectors) == 1:
            results = []
            for image_path in image_paths:
                result = await self.process_detection(image_path=image_path)
                results.append(serialize_passport_detection(result))
            return results

        detections = await asyncio.gather(
            *(
                self.process_detection(image_path=image_path)
                for image_path in image_paths
            )
        )
        return [serialize_passport_detection(result) for result in detections]

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._is_closed:
                is_first_closer = False
            else:
                self._is_closed = True
                is_first_closer = True
        if not is_first_closer:
            self._close_complete.wait()
            return
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
            for detector in self._detectors:
                detector.close()
        finally:
            self._close_complete.set()
        logger.info("Passport service shut down after draining OCR work")


@lru_cache(maxsize=1)
def get_passport_service() -> PassportService:
    return PassportService()
