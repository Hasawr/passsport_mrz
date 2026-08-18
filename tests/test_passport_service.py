from dataclasses import replace
from unittest.mock import MagicMock
import pytest

from services.passport import PassportDetectionOutput, PassportMRZResult
from services.passport.service import PassportService, serialize_passport_detection
from shared.config import get_settings


def test_serialize_passport_detection():
    mrz_res = PassportMRZResult(
        line1="P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        line2="C123456780AZE9001011M3001011<<<<<<<<<<<<<<04",
        confidence=0.95,
        checksum_valid=True,
        composite_checksum_valid=True,
        method="mrz_strip",
        passport_number="C12345678",
        nationality="AZE",
        date_of_birth="900101",
        sex="M",
        expiry_date="300101",
        personal_number=None,
        surname="DOE",
        given_names="JOHN",
    )
    output = PassportDetectionOutput(
        mrz_line1=mrz_res.line1,
        mrz_line2=mrz_res.line2,
        confidence=0.95,
        mrz_result=mrz_res,
        notes=["Test note"],
    )

    serialized = serialize_passport_detection(output)
    assert serialized["mrz_line1"] == "P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    assert serialized["mrz_details"]["passport_number"] == "C12345678"
    assert "line_confidences" not in serialized["mrz_details"]


@pytest.mark.asyncio
async def test_passport_service_process_mock(tmp_path):
    img_file = tmp_path / "passport.png"
    img_file.write_bytes(b"fake image data")

    service = PassportService.__new__(PassportService)
    service._is_closed = False
    service._lifecycle_lock = MagicMock()
    mock_detector = MagicMock()

    mrz_res = PassportMRZResult(
        line1="P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        line2="C123456780AZE9001011M3001011<<<<<<<<<<<<<<04",
        confidence=0.95,
        checksum_valid=True,
        composite_checksum_valid=True,
        method="mrz_strip",
        passport_number="C12345678",
        nationality="AZE",
        date_of_birth="900101",
        sex="M",
        expiry_date="300101",
        personal_number=None,
        surname="DOE",
        given_names="JOHN",
    )
    mock_detector.detect_from_passport.return_value = PassportDetectionOutput(
        mrz_line1=mrz_res.line1,
        mrz_line2=mrz_res.line2,
        confidence=0.95,
        mrz_result=mrz_res,
    )

    from concurrent.futures import ThreadPoolExecutor
    service._executor = ThreadPoolExecutor(max_workers=1)
    service._detectors = [mock_detector]
    service._next_detector_index = 0

    try:
        res = await service.process(image_path=img_file)
        assert res["mrz_line1"].startswith("P<AZE")
    finally:
        service._executor.shutdown(wait=False)
