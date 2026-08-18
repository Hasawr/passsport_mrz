from unittest.mock import MagicMock
import numpy as np
from services.passport.mrz_extractor import PassportMRZExtractor
from services.passport.detector import PassportDetector, OCRAttempt
from services.passport.preprocessor import PassportPreprocessor
from services.passport import PassportMRZResult, PassportDetectionOutput


def test_passport_mrz_extractor_scoring():
    line1 = "P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    score1 = PassportMRZExtractor._mrz_line_score(line1)
    score_random = PassportMRZExtractor._mrz_line_score("REPUBLIC OF AZERBAIJAN PASSPORT")
    score_header = PassportMRZExtractor._mrz_line_score("PASSPORTTYPECOUNTRY<CODEAAH714669<<<<<<<<<<<")
    assert score1 > score_random
    assert score1 > score_header


def test_passport_mrz_extractor_parse_valid():
    extractor = PassportMRZExtractor(ocr_engine=MagicMock())
    lines = (
        ("P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", 0.95),
        ("C123456788AZE9001011M3001011<<<<<<<<<<<<<<04", 0.95),
    )
    res = extractor._parse_td3_lines(lines, attempt="test")
    assert res.passport_number == "C12345678"
    assert res.surname == "DOE"
    assert res.given_names == "JOHN"
    assert res.checksum_valid is True


def test_passport_mrz_extractor_structurally_valid():
    res = PassportMRZResult(
        line1="P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        line2="C123456780AZE9001011M3001011<<<<<<<<<<<<<<04",
        confidence=0.9,
        checksum_valid=True,
        composite_checksum_valid=True,
        method="test",
        passport_number="C12345678",
        nationality="AZE",
        date_of_birth="900101",
        sex="M",
        expiry_date="300101",
        personal_number=None,
        surname="DOE",
        given_names="JOHN",
    )
    assert PassportMRZExtractor.is_structurally_valid(res) is True


def test_passport_mrz_extractor_rejects_header_noise():
    res = PassportMRZResult(
        line1="PASSPORTTYPECOUNTRY<CODEAAH714669<<<<<<<<<<<",
        line2="SEXO<SEXLUGAR<DE<NACIMIENTO<PLACE<OF<BIRTH<<",
        confidence=0.98,
        checksum_valid=False,
        composite_checksum_valid=False,
        method="full_image",
        passport_number="SEXOSEXL",
        nationality="GAR",
        date_of_birth=None,
        sex=None,
        expiry_date=None,
        personal_number=None,
        surname=None,
        given_names=None,
    )
    assert PassportMRZExtractor.is_structurally_valid(res) is False


def test_extract_recognition_lines_nested_paddle_format():
    class FakePaddleOCR:
        def ocr(self, images, det=False, rec=True, cls=False):
            # PaddleOCR multi-image batch return format
            return [
                [("P<ARGDOBERTI<<JULIANA<<<<<<<<<<<<<<<<<<<<<<", 0.92)],
                [("AAH7146692ARG1210233F270606352768315<<<<96", 0.96)],
            ]

    extractor = PassportMRZExtractor(FakePaddleOCR())
    res = extractor.extract_recognition_lines(
        [np.zeros((30, 200, 3)), np.zeros((30, 200, 3))],
        attempt="line_recognition",
    )
    assert res.passport_number == "AAH714669"
    assert res.surname == "DOBERTI"
    assert res.given_names == "JULIANA"
    assert res.checksum_valid is True
    assert PassportMRZExtractor.is_structurally_valid(res) is True


def test_non_adjacent_candidates_matched():
    merged_lines = [
        ("P<USADOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", 0.95),
        ("RANDOM MIDDLE PROSE LINE HERE 12345678", 0.88),
        ("C123456788USA9001011M3001011<<<<<<<<<<<<<<04", 0.95),
    ]
    candidates = PassportMRZExtractor._find_td3_candidates(merged_lines)
    assert len(candidates) >= 1
    best_candidate = candidates[0]
    assert best_candidate.lines[0][0].startswith("P<USA")
    assert best_candidate.lines[1][0].startswith("C123456788")


def test_detector_caching_and_noop_deskew():
    detector = PassportDetector.__new__(PassportDetector)
    detector.preprocessor = PassportPreprocessor()
    detector.max_ocr_side = 1600
    flat_image = np.zeros((100, 300, 3), dtype=np.uint8)

    attempts = detector._build_attempts(flat_image)
    deskewed_full_factory = next(a.image_factory for a in attempts if a.name == "deskewed_full_image")
    # For a flat image with no angle, deskew returns the original image, so factory returns None (no-op skip)
    assert deskewed_full_factory() is None


def test_detector_finalize_returns_none_when_all_invalid():
    detector = PassportDetector.__new__(PassportDetector)
    detector.save_debug_images = False

    invalid_res = PassportMRZResult(
        line1="PASSPORTTYPECOUNTRY<CODEAAH714669<<<<<<<<<<<",
        line2="SEXO<SEXLUGAR<DE<NACIMIENTO<PLACE<OF<BIRTH<<",
        confidence=0.98,
        checksum_valid=False,
        composite_checksum_valid=False,
        method="full_image",
        passport_number="SEXOSEXL",
        nationality=None,
        date_of_birth=None,
        sex=None,
        expiry_date=None,
        personal_number=None,
        surname=None,
        given_names=None,
    )

    out = detector._finalize_detection(
        rectified=np.zeros((10, 10, 3)),
        attempted_results=[invalid_res],
        valid_results=[],
        notes=[],
        image_path=MagicMock(),
    )
    assert out.mrz_line1 is None
    assert out.mrz_line2 is None
    assert out.confidence == 0.0

