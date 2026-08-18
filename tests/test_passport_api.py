from dataclasses import replace
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from api.main import app
from services.passport import PassportDetectionOutput, PassportMRZResult
from services.passport.service import PassportService, get_passport_service
from shared.audit import get_audit_store
from shared.config import get_settings

TEST_API_KEY = "passport-test-key-32-characters-min-len"


def make_test_image() -> bytes:
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 300), "P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", fill=(0, 0, 0))
    draw.text((20, 340), "C123456780AZE9001011M3001011<<<<<<<<<<<<<<04", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockPassportService:
    async def process(self, image_path):
        mrz_res = PassportMRZResult(
            line1="P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            line2="C123456780AZE9001011M3001011<<<<<<<<<<<<<<04",
            confidence=0.98,
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
            confidence=0.98,
            mrz_result=mrz_res,
            notes=[],
        )
        from services.passport.service import serialize_passport_detection
        return serialize_passport_detection(output)

    async def process_many(self, image_paths):
        return [await self.process(p) for p in image_paths]


@pytest.fixture
def client(tmp_path):
    audit_db = tmp_path / "audit.db"
    payload_dir = tmp_path / "payloads"
    test_settings = replace(
        get_settings(),
        api_keys=(TEST_API_KEY,),
        audit_db_path=audit_db,
        audit_payload_dir=payload_dir,
    )
    store = get_audit_store(audit_db, payload_dir=payload_dir)
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_passport_service] = lambda: MockPassportService()
    app.state.settings = test_settings
    app.state.audit_store = store
    with TestClient(app) as test_client:
        yield test_client, store
    app.dependency_overrides.clear()


def test_passport_single_endpoint_success(client):
    test_client, _ = client
    response = test_client.post(
        "/v1/passport",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("passport.png", make_test_image(), "image/png")},
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["service"] == "passport"
    assert json_data["data"]["mrz_line1"].startswith("P<AZE")
    assert json_data["data"]["mrz_details"]["passport_number"] == "C12345678"


def test_passport_unauthenticated_rejected(client):
    test_client, _ = client
    response = test_client.post(
        "/v1/passport",
        headers={"X-API-Key": "invalid-key"},
        files={"mrz": ("passport.png", make_test_image(), "image/png")},
    )
    assert response.status_code == 401


def test_passport_batch_endpoint_success(client):
    test_client, _ = client
    img_bytes = make_test_image()
    response = test_client.post(
        "/v1/passport/batch",
        headers={"X-API-Key": TEST_API_KEY},
        files=[
            ("mrz", ("pass1.png", img_bytes, "image/png")),
            ("mrz", ("pass2.png", img_bytes, "image/png")),
        ],
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["service"] == "passport"
    assert json_data["data"]["count"] == 2
    assert len(json_data["data"]["results"]) == 2


def test_passport_audit_redaction(client):
    test_client, store = client
    response = test_client.post(
        "/v1/passport",
        headers={"X-API-Key": TEST_API_KEY},
        files={"mrz": ("passport.png", make_test_image(), "image/png")},
    )
    assert response.status_code == 200
    records = store.recent_events(hours=None)
    assert len(records) > 0
    payload_str = str(records[0].response_body)
    assert "[REDACTED]" in payload_str
    assert "C12345678" not in payload_str
    assert "DOE" not in payload_str
