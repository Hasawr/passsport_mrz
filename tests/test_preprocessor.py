import cv2
import numpy as np
import pytest

from services.passport.preprocessor import PassportPreprocessor


def test_deskew_mrz_strip_preserves_shape():
    strip = np.full((120, 500, 3), 255, dtype=np.uint8)
    for y in (30, 60, 90):
        cv2.line(strip, (20, y), (480, y + 35), (0, 0, 0), 3)

    corrected = PassportPreprocessor.deskew_mrz_strip(strip)
    assert corrected.shape == strip.shape


def test_crop_mrz_strip_returns_bottom_portion():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    strip = PassportPreprocessor.crop_mrz_strip(img, 0.25)
    assert strip.shape == (100, 600, 3)


def test_bound_ocr_input_resizes_large_images():
    img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    bounded = PassportPreprocessor.bound_ocr_input(img, max_side=1600)
    assert max(bounded.shape[:2]) == 1600
