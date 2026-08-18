import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module
import logging
from pathlib import Path
import time
from typing import Callable

import numpy as np

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "bool": [np.bool_],
    }

from . import PassportDetectionOutput, PassportMRZResult
from .image_utils import draw_mrz_debug, save_debug_image
from .mrz_extractor import PassportMRZExtractor
from .preprocessor import PassportPreprocessor

logger = logging.getLogger(__name__)
_DLL_DIRECTORY_HANDLES: list[object] = []
PRODUCTION_OCR_VERSION = "PP-OCRv4"


class PassportOCRProcessingError(RuntimeError):
    """Raised when the passport OCR engine cannot complete an image."""


@dataclass(frozen=True)
class OCRAttempt:
    name: str
    image_factory: Callable[[], object | None]


def configure_nvidia_dll_directories() -> None:
    """Expose pip-installed NVIDIA DLLs to Paddle on Windows."""
    if os.name != "nt" or _DLL_DIRECTORY_HANDLES:
        return

    bin_directories: list[str] = []
    for module_name in ("nvidia.cudnn", "nvidia.cublas", "nvidia.cuda_nvrtc"):
        try:
            module = import_module(module_name)
            bin_directory = Path(next(iter(module.__path__))) / "bin"
            bin_directories.append(str(bin_directory))
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(bin_directory)))
        except (ImportError, StopIteration, AttributeError):
            continue

    if bin_directories:
        os.environ["PATH"] = os.pathsep.join(
            [*bin_directories, os.environ.get("PATH", "")]
        )


PADDLE_ALLOCATOR_DEFAULTS = {
    "FLAGS_allocator_strategy": "auto_growth",
    "FLAGS_initial_gpu_memory_in_mb": "1024",
    "FLAGS_reallocate_gpu_memory_in_mb": "1024",
}


def configure_paddle_allocator() -> None:
    """Apply GPU allocator defaults, leaving any operator override intact."""
    for flag, value in PADDLE_ALLOCATOR_DEFAULTS.items():
        os.environ.setdefault(flag, value)


class PassportDetector:
    """Multi-attempt cascade detector for TD3 passport MRZs."""

    HIGH_CONFIDENCE_ACCEPT = 0.90
    RECTIFIED_FAIL_FAST_ATTEMPTS = 3
    ORIGINAL_IMAGE_RECOVERY_ATTEMPTS = frozenset(
        {"mrz_strip", "mrz_roi", "full_image"}
    )
    RECOVERY_ATTEMPTS = frozenset(
        {
            "deskewed_mrz_roi",
            "deskewed_mrz_strip",
            "deskewed_mrz_strip_binarized",
            "full_image",
            "deskewed_full_image",
            "line_recognition",
            "original_mrz_strip",
            "original_mrz_roi",
            "original_full_image",
        }
    )

    def __init__(
        self,
        use_gpu: bool = True,
        save_debug_images: bool = False,
        max_ocr_side: int = PassportPreprocessor.DEFAULT_MAX_OCR_SIDE,
        det_limit_side_len: int = 736,
        concurrent_attempts: int = 1,
    ):
        if use_gpu:
            configure_nvidia_dll_directories()
            configure_paddle_allocator()

        import paddle
        import paddle.inference as paddle_infer
        from paddleocr import PaddleOCR

        if not hasattr(paddle_infer, "_mkldnn_patched"):
            _orig_cp = paddle_infer.create_predictor
            def _safe_cp(config):
                try:
                    config.disable_mkldnn()
                except Exception:
                    pass
                return _orig_cp(config)
            paddle_infer.create_predictor = _safe_cp
            paddle_infer._mkldnn_patched = True

        if use_gpu and not paddle.device.is_compiled_with_cuda():
            raise RuntimeError(
                "GPU mode requires the paddlepaddle-gpu package. "
                "Reinstall dependencies from requirements.txt."
            )

        logger.info(
            "Initializing PassportDetector (use_gpu=%s, save_debug_images=%s, concurrent_attempts=%d)",
            use_gpu,
            save_debug_images,
            concurrent_attempts,
        )

        def build_engine() -> PaddleOCR:
            if not use_gpu:
                try:
                    paddle.set_flags({"FLAGS_use_mkldnn": False})
                except Exception:
                    pass
            return PaddleOCR(
                use_angle_cls=False,
                lang="en",
                show_log=False,
                use_gpu=use_gpu,
                max_text_length=50,
                det_limit_side_len=det_limit_side_len,
                det_limit_type="max",
                ocr_version=PRODUCTION_OCR_VERSION,
                enable_mkldnn=False,
            )

        self.preprocessor = PassportPreprocessor()
        self.mrz_extractor = PassportMRZExtractor(build_engine())
        extra_engine_count = max(0, concurrent_attempts - 1)
        self._extractor_pool = [self.mrz_extractor] + [
            PassportMRZExtractor(build_engine()) for _ in range(extra_engine_count)
        ]
        self._attempt_executor = (
            ThreadPoolExecutor(
                max_workers=len(self._extractor_pool),
                thread_name_prefix="passport-attempt",
            )
            if extra_engine_count
            else None
        )
        self.save_debug_images = save_debug_images
        self.max_ocr_side = max_ocr_side

    def detect_from_passport(self, image_path: str | Path) -> PassportDetectionOutput:
        image_path = Path(image_path)
        notes: list[str] = []
        try:
            image = self.preprocessor.load(image_path)
            rectified = self.preprocessor.detect_card_roi(image)
            notes.append(
                "Passport ROI boundary not detected; using original image."
                if rectified is image
                else "Passport ROI successfully detected and rectified."
            )
            attempts = self._build_attempts(rectified)
            attempted_results = []
            valid_results = []
            roi_applied = rectified is not image

            self._run_attempt_cascade(
                attempts,
                attempted_results=attempted_results,
                valid_results=valid_results,
                notes=notes,
                image_path=image_path,
                roi_applied=roi_applied,
            )

            if not valid_results and roi_applied:
                for attempt in self._build_attempts(image):
                    if (
                        attempt.name
                        not in self.ORIGINAL_IMAGE_RECOVERY_ATTEMPTS
                    ):
                        continue
                    attempt_image = attempt.image_factory()
                    if attempt_image is None:
                        continue
                    attempt_name = f"original_{attempt.name}"
                    attempt_started = time.perf_counter()
                    attempted_result = self.mrz_extractor.extract(
                        attempt_image,
                        attempt=attempt_name,
                    )
                    attempt_seconds = time.perf_counter() - attempt_started
                    notes.append(
                        f"OCR attempt {attempt_name}: {attempt_seconds:.4f}s."
                    )
                    attempted_results.append(attempted_result)
                    if not self.mrz_extractor.is_structurally_valid(
                        attempted_result
                    ):
                        continue
                    valid_results.append(attempted_result)
                    if self._should_stop_after_valid_result(
                        attempted_result,
                        valid_results,
                        notes,
                        attempt_name=attempt_name,
                    ):
                        break

            if not valid_results:
                candidate_sources = [rectified, image] if roi_applied else [rectified]
                for src in candidate_sources:
                    for hr in (None, self.preprocessor.MRZ_HEIGHT_RATIO_WIDE):
                        line_images = self.preprocessor.extract_mrz_line_crops(
                            src,
                            line_count=2,
                            height_ratio=hr,
                        )
                        if len(line_images) == 2:
                            attempt_name = "line_recognition" if hr is None else "line_recognition_wide"
                            attempt_started = time.perf_counter()
                            attempted_result = (
                                self.mrz_extractor.extract_recognition_lines(
                                    line_images,
                                    attempt=attempt_name,
                                )
                            )
                            attempt_seconds = time.perf_counter() - attempt_started
                            notes.append(
                                f"OCR attempt {attempt_name}: {attempt_seconds:.4f}s."
                            )
                            attempted_results.append(attempted_result)
                            if self.mrz_extractor.is_structurally_valid(
                                attempted_result
                            ):
                                valid_results.append(attempted_result)
                                break
                    if valid_results:
                        break

            return self._finalize_detection(
                rectified=rectified,
                attempted_results=attempted_results,
                valid_results=valid_results,
                notes=notes,
                image_path=image_path,
            )
        except Exception as exc:
            logger.exception("Error processing passport image %s", image_path.name)
            raise PassportOCRProcessingError(
                f"Failed to process passport image {image_path.name}."
            ) from exc

    def _run_attempt_cascade(
        self,
        attempts: list[OCRAttempt],
        *,
        attempted_results: list[PassportMRZResult],
        valid_results: list[PassportMRZResult],
        notes: list[str],
        image_path: Path,
        roi_applied: bool,
    ) -> None:
        extractor_pool = getattr(self, "_extractor_pool", None) or [self.mrz_extractor]
        executor = getattr(self, "_attempt_executor", None)
        batch_size = len(extractor_pool) if executor is not None else 1

        failed_rectified_attempts = 0
        index = 0
        total = len(attempts)

        while index < total:
            mrz_counts = Counter(
                (r.line1, r.line2) for r in valid_results if r.line1 and r.line2
            )
            has_conflict = len(mrz_counts) > 1
            strongest_consensus = max(mrz_counts.values(), default=0)

            batch: list[OCRAttempt] = []
            cascade_done = False
            while len(batch) < batch_size and index < total:
                attempt = attempts[index]
                if (
                    attempt.name in self.RECOVERY_ATTEMPTS
                    and strongest_consensus >= 2
                    and not has_conflict
                ):
                    cascade_done = True
                    break
                index += 1
                batch.append(attempt)

            prepared = [
                (attempt, attempt.image_factory()) for attempt in batch
            ]
            prepared = [
                (attempt, img) for attempt, img in prepared if img is not None
            ]

            if not prepared:
                if cascade_done:
                    return
                continue

            if len(prepared) == 1 or executor is None:
                timed_results = [
                    self._time_attempt(extractor_pool[0], attempt, img)
                    for attempt, img in prepared
                ]
            else:
                futures = [
                    executor.submit(
                        self._time_attempt,
                        extractor_pool[pos % len(extractor_pool)],
                        attempt,
                        img,
                    )
                    for pos, (attempt, img) in enumerate(prepared)
                ]
                timed_results = [f.result() for f in futures]

            for attempt, attempted_result, attempt_seconds in timed_results:
                notes.append(f"OCR attempt {attempt.name}: {attempt_seconds:.4f}s.")
                logger.info(
                    "OCR attempt %s for %s completed in %.4fs",
                    attempt.name,
                    image_path.name,
                    attempt_seconds,
                )
                attempted_results.append(attempted_result)
                if self.mrz_extractor.is_structurally_valid(attempted_result):
                    valid_results.append(attempted_result)
                    if self._should_stop_after_valid_result(
                        attempted_result,
                        valid_results,
                        notes,
                        attempt_name=attempt.name,
                    ):
                        return
                elif roi_applied and not valid_results:
                    failed_rectified_attempts += 1
                    if failed_rectified_attempts >= self.RECTIFIED_FAIL_FAST_ATTEMPTS:
                        notes.append(
                            f"Rectified crop produced no MRZ after {self.RECTIFIED_FAIL_FAST_ATTEMPTS} attempts; trying original image."
                        )
                        return

            if cascade_done:
                return

    @staticmethod
    def _time_attempt(
        extractor: PassportMRZExtractor,
        attempt: OCRAttempt,
        attempt_image: object,
    ) -> tuple[OCRAttempt, PassportMRZResult, float]:
        attempt_started = time.perf_counter()
        result = extractor.extract(attempt_image, attempt=attempt.name)
        return attempt, result, time.perf_counter() - attempt_started

    def _should_stop_after_valid_result(
        self,
        result: PassportMRZResult,
        valid_results: list[PassportMRZResult],
        notes: list[str],
        *,
        attempt_name: str,
    ) -> bool:
        distinct_mrzs = {
            (r.line1, r.line2) for r in valid_results if r.line1 and r.line2
        }
        if len(distinct_mrzs) > 1:
            return False

        if (
            result.is_canonical
            and (result.checksum_valid or result.composite_checksum_valid)
            and (
                result.confidence >= self.HIGH_CONFIDENCE_ACCEPT
                or (result.checksum_valid and result.composite_checksum_valid)
            )
        ):
            notes.append(
                f"High-confidence/valid-checksum passport MRZ accepted after {attempt_name}."
            )
            return True

        matching_mrz = [
            r for r in valid_results if r.line1 == result.line1 and r.line2 == result.line2
        ]
        if len(matching_mrz) >= 2:
            notes.append(f"Strong passport MRZ consensus reached after {attempt_name}.")
            return True

        return False

    def _finalize_detection(
        self,
        *,
        rectified: object,
        attempted_results: list[PassportMRZResult],
        valid_results: list[PassportMRZResult],
        notes: list[str],
        image_path: Path,
    ) -> PassportDetectionOutput:
        if valid_results:
            mrz_counts = Counter(
                (r.line1, r.line2) for r in valid_results if r.line1 and r.line2
            )
            best = max(
                valid_results,
                key=lambda r: self._result_rank(r, mrz_counts),
            )
            notes.append(
                f"Passport MRZ selected from {best.method} with "
                f"{mrz_counts[(best.line1, best.line2)]} agreeing attempt(s)."
            )
            if self.save_debug_images and hasattr(rectified, "copy"):
                debug_img = draw_mrz_debug(rectified, best.line1, best.line2)
                save_debug_image(debug_img, f"passport_debug_{image_path.stem}.jpg")

            return PassportDetectionOutput(
                mrz_line1=best.line1,
                mrz_line2=best.line2,
                confidence=best.confidence,
                mrz_result=best,
                notes=notes,
            )

        notes.append(
            "Passport MRZ region was not reliably detected, or OCR output did not contain a valid TD3 structure."
        )
        best_attempt = (
            max(attempted_results, key=lambda r: r.confidence)
            if attempted_results
            else None
        )
        return PassportDetectionOutput(
            mrz_line1=None,
            mrz_line2=None,
            confidence=0.0,
            mrz_result=best_attempt,
            notes=notes,
        )

    @staticmethod
    def _result_rank(
        result: PassportMRZResult,
        mrz_counts: Counter,
    ) -> tuple[int, int, int, float, float]:
        consensus_count = mrz_counts.get((result.line1, result.line2), 0)
        checks_passed = (1 if result.checksum_valid else 0) + (
            1 if result.composite_checksum_valid else 0
        )
        canonical = 1 if result.is_canonical else 0
        return (consensus_count, checks_passed, canonical, result.quality_score, result.confidence)

    def _build_attempts(self, image: np.ndarray) -> list[OCRAttempt]:
        deskewed_cache: dict[str, object] = {}
        mrz_roi_cache: dict[int, object] = {}
        deskewed_strip_cache: dict[float, object] = {}

        prep = self.preprocessor
        max_side = self.max_ocr_side

        def prepare_crop(
            img,
            *,
            binarize: bool = False,
            adaptive_upscale: bool = False,
        ):
            if img is None:
                return None
            return prep.bound_ocr_input(
                prep.prepare_mrz_for_ocr(
                    img,
                    binarize=binarize,
                    adaptive_upscale=adaptive_upscale,
                ),
                max_side=max_side,
            )

        def get_mrz_roi(source, *, binarize: bool = False):
            source_key = id(source)
            if source_key not in mrz_roi_cache:
                mrz_roi_cache[source_key] = prep.detect_mrz_roi(source)
            mrz_roi = mrz_roi_cache[source_key]
            if mrz_roi is None:
                return None
            return prepare_crop(mrz_roi, binarize=binarize)

        def get_deskewed():
            if "image" not in deskewed_cache:
                deskewed_cache["image"] = prep.deskew(image)
            deskewed = deskewed_cache["image"]
            return None if (deskewed is image or np.array_equal(deskewed, image)) else deskewed

        def get_strip(ratio: float, *, binarize: bool = False):
            strip = prep.crop_mrz_strip(image, height_ratio=ratio)
            return prepare_crop(strip, binarize=binarize)

        def get_deskewed_strip(ratio: float, *, binarize: bool = False):
            if ratio not in deskewed_strip_cache:
                strip = prep.crop_mrz_strip(image, height_ratio=ratio)
                deskewed = prep.deskew_mrz_strip(strip)
                deskewed_strip_cache[ratio] = (
                    None if deskewed is strip else deskewed
                )
            deskewed = deskewed_strip_cache[ratio]
            if deskewed is None:
                return None
            return prepare_crop(
                deskewed,
                binarize=binarize,
                adaptive_upscale=True,
            )

        def get_deskewed_mrz_roi(*, binarize: bool = False):
            deskewed = get_deskewed()
            return None if deskewed is None else get_mrz_roi(deskewed, binarize=binarize)

        def get_deskewed_full():
            deskewed = get_deskewed()
            if deskewed is None:
                return None
            return prep.bound_ocr_input(deskewed, max_side=max_side)

        return [
            OCRAttempt("mrz_strip", lambda: get_strip(prep.MRZ_HEIGHT_RATIO)),
            OCRAttempt("mrz_strip_wide", lambda: get_strip(prep.MRZ_HEIGHT_RATIO_WIDE)),
            OCRAttempt("full_image", lambda: prep.bound_ocr_input(image, max_side=max_side)),
            OCRAttempt("mrz_strip_tight", lambda: get_strip(prep.MRZ_HEIGHT_RATIO_TIGHT)),
            OCRAttempt("mrz_roi", lambda: get_mrz_roi(image)),
            OCRAttempt("mrz_strip_binarized", lambda: get_strip(prep.MRZ_HEIGHT_RATIO, binarize=True)),
            OCRAttempt("mrz_roi_binarized", lambda: get_mrz_roi(image, binarize=True)),
            OCRAttempt("deskewed_mrz_strip", lambda: get_deskewed_strip(prep.MRZ_HEIGHT_RATIO)),
            OCRAttempt("deskewed_mrz_strip_binarized", lambda: get_deskewed_strip(prep.MRZ_HEIGHT_RATIO, binarize=True)),
            OCRAttempt("deskewed_mrz_roi", lambda: get_deskewed_mrz_roi()),
            OCRAttempt("deskewed_full_image", get_deskewed_full),
        ]

    def close(self) -> None:
        if self._attempt_executor is not None:
            self._attempt_executor.shutdown(wait=False)
