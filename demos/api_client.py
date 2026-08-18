from collections.abc import Iterable
from dataclasses import dataclass
from mimetypes import guess_type
from pathlib import Path
from typing import Any

import httpx

from shared.config import Settings, get_settings


@dataclass(frozen=True)
class APIJob:
    client_ref: str
    image_path: Path


def prepare_file_tuple(image_path: Path) -> tuple[str, tuple[str, bytes, str]]:
    media_type = guess_type(image_path.name)[0] or "application/octet-stream"
    return ("mrz", (image_path.name, image_path.read_bytes(), media_type))


def request_passport(
    client: httpx.Client,
    image_path: Path,
    api_key: str,
) -> dict[str, Any]:
    file_field, file_tuple = prepare_file_tuple(image_path)
    response = client.post(
        "/v1/passport",
        headers={"X-API-Key": api_key},
        files=[(file_field, file_tuple)],
    )
    response.raise_for_status()
    return response.json()


def process_passport_jobs(
    jobs: Iterable[APIJob],
    *,
    settings: Settings | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    key = api_key or (settings.api_keys[0] if settings.api_keys else "")
    client = httpx.Client(
        base_url=settings.ocr_api_base_url,
        timeout=120.0,
    )
    results = []
    try:
        for job in jobs:
            payload = request_passport(client, job.image_path, key)
            results.append(
                {
                    "client_ref": job.client_ref,
                    "response": payload,
                }
            )
    finally:
        client.close()
    return results
