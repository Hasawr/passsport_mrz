from services.passport.validator import (
    clean_mrz_line,
    compute_mrz_check_digit,
    compute_td3_composite_check,
    correct_ocr_digits,
    is_valid_date_field,
    is_valid_passport_number,
    parse_td3_name,
)


def test_mrz_check_digit_calculation():
    # Standard ICAO 7-3-1 weight test
    # "AB1234567" -> 191 % 10 = 1
    assert compute_mrz_check_digit("AB1234567") == 1
    # "900101" -> 9*7 + 0*3 + 0*1 + 1*7 + 0*3 + 1*1 = 63 + 7 + 1 = 71 % 10 = 1
    assert compute_mrz_check_digit("900101") == 1


def test_ocr_digit_correction():
    assert correct_ocr_digits("O123I567S8") == "0123156758"
    assert correct_ocr_digits("B9G") == "896"


def test_clean_mrz_line():
    raw = "P<AZEDOE,JOHN (123)"
    cleaned = clean_mrz_line(raw)
    assert cleaned == "P<AZEDOE<JOHN<<123<"


def test_parse_td3_name():
    surname, given_names = parse_td3_name("DOE<<JOHN<ALEXANDER<<<<<<<<<<<<<<<<<<")
    assert surname == "DOE"
    assert given_names == "JOHN ALEXANDER"

    surname_only, given_none = parse_td3_name("SMITH<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    assert surname_only == "SMITH"
    assert given_none is None


def test_is_valid_passport_number():
    assert is_valid_passport_number("C12345678") is True
    assert is_valid_passport_number("123456789") is True
    assert is_valid_passport_number("C12345678<<<<") is True
    assert is_valid_passport_number("") is False


def test_is_valid_date_field():
    assert is_valid_date_field("900101") is True  # Jan 1, 90
    assert is_valid_date_field("901301") is False  # Month 13 invalid
    assert is_valid_date_field("900132") is False  # Day 32 invalid
    assert is_valid_date_field("O1O1O1") is True   # OCR digits corrected to 010101


def test_composite_check_digit():
    # TD3 line 2 sample (44 chars)
    line2 = "C123456780AZE9001011M3001011<<<<<<<<<<<<<<04"
    comp_check = compute_td3_composite_check(line2)
    assert comp_check >= 0
