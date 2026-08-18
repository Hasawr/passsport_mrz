from shared.config import Settings, validate_security_settings
import pytest


def test_validate_security_settings():
    valid = Settings(
        api_keys=("a" * 32,),
        use_gpu=False,
        save_ocr_debug_images=False,
        max_upload_bytes=10000,
        max_image_pixels=1000000,
        max_batch_files=10,
        audit_db_path=None,
        audit_payload_dir=None,
    )
    validate_security_settings(valid)

    with pytest.raises(ValueError):
        invalid = Settings(
            api_keys=(),
            use_gpu=False,
            save_ocr_debug_images=False,
            max_upload_bytes=10000,
            max_image_pixels=1000000,
            max_batch_files=10,
            audit_db_path=None,
            audit_payload_dir=None,
        )
        validate_security_settings(invalid)
