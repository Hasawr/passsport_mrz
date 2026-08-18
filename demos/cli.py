import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.passport.detector import PassportDetector
from shared.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Passport MRZ OCR CLI detector",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to passport data page image files to evaluate",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug visualization overlay images",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (ignore USE_GPU setting)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    detector = PassportDetector(
        use_gpu=not args.cpu and settings.use_gpu,
        save_debug_images=args.save_debug or settings.save_ocr_debug_images,
    )
    try:
        results = []
        for image_path_str in args.images:
            image_path = Path(image_path_str)
            output = detector.detect_from_passport(image_path)
            res_dict = {
                "file": image_path.name,
                "mrz_line1": output.mrz_line1,
                "mrz_line2": output.mrz_line2,
                "confidence": round(output.confidence, 4),
                "notes": output.notes,
            }
            if output.mrz_result:
                res_dict["mrz_details"] = {
                    "passport_number": output.mrz_result.passport_number,
                    "surname": output.mrz_result.surname,
                    "given_names": output.mrz_result.given_names,
                    "nationality": output.mrz_result.nationality,
                    "date_of_birth": output.mrz_result.date_of_birth,
                    "sex": output.mrz_result.sex,
                    "expiry_date": output.mrz_result.expiry_date,
                    "checksum_valid": output.mrz_result.checksum_valid,
                    "composite_checksum_valid": output.mrz_result.composite_checksum_valid,
                    "method": output.mrz_result.method,
                }
            results.append(res_dict)

        print(json.dumps(results, indent=2))
    finally:
        detector.close()


if __name__ == "__main__":
    main()
