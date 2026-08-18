import numpy as np

# NumPy 2.0 polyfill for legacy dependencies (imgaug / paddleocr)
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }

from dataclasses import dataclass, field


@dataclass
class PassportMRZResult:
    """Low-level MRZ extraction result for a TD3 passport."""

    line1: str
    line2: str
    confidence: float
    checksum_valid: bool
    composite_checksum_valid: bool
    method: str
    passport_number: str | None
    nationality: str | None
    date_of_birth: str | None
    sex: str | None
    expiry_date: str | None
    personal_number: str | None
    surname: str | None
    given_names: str | None
    line_confidences: tuple[float, ...] = ()
    quality_score: float = 0.0
    is_canonical: bool = False


@dataclass
class PassportDetectionOutput:
    """High-level output from the passport detector."""

    mrz_line1: str | None
    mrz_line2: str | None
    confidence: float
    mrz_result: PassportMRZResult | None = None
    notes: list[str] = field(default_factory=list)
