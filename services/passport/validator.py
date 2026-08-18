import re

PASSPORT_NUMBER_PATTERN = re.compile(r"^[A-Z0-9]{1,9}$")

# Common OCR-B confusions for digit-only MRZ fields.
OCR_DIGIT_CONFUSIONS = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }
)


def correct_ocr_digits(value: str) -> str:
    """Map letter shapes that OCR often confuses with digits."""
    if not value:
        return ""
    return value.upper().translate(OCR_DIGIT_CONFUSIONS)


def clean_mrz_line(text: str) -> str:
    """Normalize OCR text into standard upper-case MRZ character set."""
    if not text:
        return ""
    text = text.strip().upper()
    for char in ["(", ")", "{", "}", "[", "]", ",", ".", ";", ":", "`", '"', "'"]:
        text = text.replace(char, "<")
    text = text.replace(" ", "<")
    return re.sub(r"[^A-Z0-9<]", "<", text)


def compute_mrz_check_digit(data: str) -> int:
    """Standard ICAO Doc 9303 weighted check digit algorithm (weights 7, 3, 1)."""
    weights = [7, 3, 1]
    total = 0
    for index, char in enumerate(data):
        if "0" <= char <= "9":
            value = int(char)
        elif "A" <= char <= "Z":
            value = ord(char) - ord("A") + 10
        else:
            value = 0
        total += value * weights[index % 3]
    return total % 10


def is_valid_passport_number(number: str) -> bool:
    """Validate that the passport number is a plausible alphanumeric string."""
    if not number:
        return False
    cleaned = number.replace("<", "").strip()
    return bool(cleaned and PASSPORT_NUMBER_PATTERN.match(cleaned))


def is_valid_date_field(date_str: str) -> bool:
    """Check YYMMDD format: 6 digits, valid month 01-12, valid day 01-31."""
    if not date_str or len(date_str) != 6:
        return False
    digits = correct_ocr_digits(date_str)
    if not digits.isdigit():
        return False
    month = int(digits[2:4])
    day = int(digits[4:6])
    return 1 <= month <= 12 and 1 <= day <= 31


def parse_td3_name(name_field: str) -> tuple[str | None, str | None]:
    """Split TD3 name field at '<<' into (surname, given_names)."""
    if not name_field:
        return None, None
    parts = name_field.split("<<", 1)
    surname_raw = parts[0].replace("<", " ").strip()
    surname = surname_raw if surname_raw else None

    given_names = None
    if len(parts) > 1:
        given_raw = parts[1].replace("<", " ").strip()
        given_names = given_raw if given_raw else None

    return surname, given_names


def compute_td3_composite_check(line2: str) -> int:
    """Composite check digit over passport number + check + DOB + check + expiry + check + personal number + check."""
    if len(line2) < 43:
        return -1
    composite_data = line2[0:10] + line2[13:20] + line2[21:43]
    return compute_mrz_check_digit(composite_data)
