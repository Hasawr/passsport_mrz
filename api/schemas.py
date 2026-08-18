from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class ServiceResponse(BaseModel):
    service: str
    version: str = "v1"
    data: dict[str, Any] | None = None
    error: ErrorDetail | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    services: list[str] = Field(default_factory=list)


class PassportMRZDetails(BaseModel):
    line1: str = Field("", description="Reconstructed MRZ line 1 (44 chars).")
    line2: str = Field("", description="Reconstructed MRZ line 2 (44 chars).")
    confidence: float = Field(0.0, description="MRZ parse confidence from 0 to 1.")
    checksum_valid: bool = Field(False, description="Whether the passport number checksum validated.")
    composite_checksum_valid: bool = Field(False, description="Whether the composite check digit validated.")
    method: str = Field("", description="Detection method used.")
    passport_number: str | None = Field(None, description="Passport number from MRZ line 2.")
    nationality: str | None = Field(None, description="Nationality code.")
    date_of_birth: str | None = Field(None, description="Date of birth (YYMMDD).")
    sex: str | None = Field(None, description="Sex (M/F).")
    expiry_date: str | None = Field(None, description="Expiry date (YYMMDD).")
    personal_number: str | None = Field(None, description="Personal number / optional data.")
    surname: str | None = Field(None, description="Surname parsed from MRZ.")
    given_names: str | None = Field(None, description="Given names parsed from MRZ.")


class PassportData(BaseModel):
    mrz_line1: str | None = Field(None, description="Raw MRZ line 1 (44 chars), or null when not found.")
    mrz_line2: str | None = Field(None, description="Raw MRZ line 2 (44 chars), or null when not found.")
    confidence: float = Field(..., description="Overall detection confidence from 0 to 1.")
    mrz_details: PassportMRZDetails | None = Field(None, description="Parsed MRZ details.")
    notes: list[str] = Field(default_factory=list, description="Processing notes.")


class PassportResponse(BaseModel):
    service: str = Field("passport", examples=["passport"])
    version: str = "v1"
    data: PassportData | None = None
    error: ErrorDetail | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "service": "passport",
                    "version": "v1",
                    "data": {
                        "mrz_line1": "P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
                        "mrz_line2": "C123456780AZE9001011M3001011<<<<<<<<<<<<<<04",
                        "confidence": 0.95,
                        "mrz_details": {
                            "line1": "P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
                            "line2": "C123456780AZE9001011M3001011<<<<<<<<<<<<<<04",
                            "confidence": 0.95,
                            "checksum_valid": True,
                            "composite_checksum_valid": True,
                            "method": "mrz_strip",
                            "passport_number": "C12345678",
                            "nationality": "AZE",
                            "date_of_birth": "900101",
                            "sex": "M",
                            "expiry_date": "300101",
                            "personal_number": None,
                            "surname": "DOE",
                            "given_names": "JOHN",
                        },
                        "notes": [],
                    },
                    "error": None,
                }
            ]
        }
    }


class PassportBatchItem(PassportData):
    index: int = Field(..., description="Zero-based index in the batch.", examples=[0])
    file_name: str = Field(..., description="Original upload file name.", examples=["passport.jpg"])


class PassportBatchData(BaseModel):
    count: int = Field(..., description="Number of results.", examples=[1])
    results: list[PassportBatchItem]


class PassportBatchResponse(BaseModel):
    service: str = Field("passport", examples=["passport"])
    version: str = "v1"
    data: PassportBatchData | None = None
    error: ErrorDetail | None = None
