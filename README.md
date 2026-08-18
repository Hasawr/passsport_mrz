# Passport MRZ Extraction Service API

Production-quality OCR API service for extracting Machine Readable Zone (MRZ) data from **TD3 format** (2 lines × 44 characters) passport data page images.

Built with **FastAPI**, **PaddleOCR (PP-OCRv3)**, OpenCV, SQLite Audit Store, and Streamlit.

---

## Features

- **ICAO Doc 9303 TD3 Standard**: Full support for 2-line × 44-character passport MRZs.
- **Robust Field Extraction**: Passport number, issuing country, nationality, date of birth, sex, expiry date, personal number, surname, and given names.
- **Check Digit Verification**: Standard 7-3-1 weighted check digit algorithms and composite check digit validation.
- **OCR Shift & Confusion Recovery**: Automatic correction of digit/letter confusions and column shift recovery.
- **11-Attempt Cascade Engine**: Adaptive ROI detection, image deskewing, CLAHE enhancement, and candidate ranking.
- **SQLite Audit Store**: Redacts sensitive PII (`[REDACTED]`) while logging latency and operational metrics.
- **API Security**: `X-API-Key` authentication and payload validation.

---

## API Endpoints

### 1. Extract Single Passport MRZ
`POST /v1/passport`

**Headers:**
`X-API-Key: passport-mrz-secret-key-32-chars-minimum-length`

**Body (`multipart/form-data`):**
- `mrz`: Passport data page image file (PNG, JPG, BMP).

**Sample Response (`200 OK`):**
```json
{
  "service": "passport",
  "version": "v1",
  "data": {
    "mrz_line1": "P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
    "mrz_line2": "C123456780AZE9001011M3001011<<<<<<<<<<<<<<04",
    "confidence": 0.98,
    "mrz_details": {
      "line1": "P<AZEDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
      "line2": "C123456780AZE9001011M3001011<<<<<<<<<<<<<<04",
      "confidence": 0.98,
      "checksum_valid": true,
      "composite_checksum_valid": true,
      "method": "mrz_strip",
      "passport_number": "C12345678",
      "nationality": "AZE",
      "date_of_birth": "900101",
      "sex": "M",
      "expiry_date": "300101",
      "personal_number": null,
      "surname": "DOE",
      "given_names": "JOHN"
    },
    "notes": []
  },
  "error": null
}
```

### 2. Extract Batch Passport MRZs
`POST /v1/passport/batch`

**Body (`multipart/form-data`):**
- Multiple `mrz` file parts.

### 3. Health Check
`GET /health`

---

## Running the Application

### Start API Server
```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Run Streamlit Web Dashboard
```powershell
streamlit run demos/streamlit_app.py
```

### Run CLI Tool
```powershell
python demos/cli.py passport_image.jpg --save-debug
```

### Run Unit & Integration Tests
```powershell
python -m pytest
```
