from dataclasses import dataclass
import logging
import re

from . import PassportMRZResult
from .layout import (
    TD3_COMPOSITE_CHECK,
    TD3_DOC_SUBTYPE,
    TD3_DOC_TYPE,
    TD3_DOB,
    TD3_DOB_CHECK,
    TD3_EXPIRY,
    TD3_EXPIRY_CHECK,
    TD3_ISSUING_COUNTRY,
    TD3_LINE_LENGTH,
    TD3_NAME,
    TD3_NATIONALITY,
    TD3_PASSPORT_CHECK,
    TD3_PASSPORT_NUMBER,
    TD3_PERSONAL_CHECK,
    TD3_PERSONAL_NUMBER,
    TD3_SEX,
)
from .validator import (
    clean_mrz_line,
    compute_mrz_check_digit,
    compute_td3_composite_check,
    correct_ocr_digits,
    is_valid_date_field,
    is_valid_passport_number,
    parse_td3_name,
)

logger = logging.getLogger(__name__)
MRZ_LINE_MIN_LENGTH = 10

# Common passport page header/label words that should NOT appear in MRZ lines.
# Used to penalise OCR lines that contain natural-language text from the data page.
PASSPORT_NOISE_WORDS = frozenset({
    "PASSPORT", "PASSAPORTE", "PASSAPORT", "PASSEPORT", "REISEPASS", "PASAPORTE",
    "REPUBLICA", "REPUBLIC", "REPUBLIQUE", "REPUBLIKA", "MERCOSUR",
    "SURNAME", "NATIONALITY", "NACIONALIDADE", "NACIMIENTO",
    "FUNCIONARIO", "PUBLICO", "AUTHORITY", "AUTORITE", "AUTORIDAD",
    "NOME", "APELIDO", "APELLIDO", "PELLIDO", "PROFESSION", "PROFISSAO",
    "DOCUMENT", "DOCUMENTO", "TRAVEL", "SEXO", "LUGAR", "TIPO", "CODIGO", "NUMERO",
    "FECHA", "EMISION", "VENCIMIENTO", "EXPIRY", "BIRTH", "HUELLA", "FIRMA", "SIGNATURE", "FINGER",
})


@dataclass(frozen=True)
class PassportMRZCandidate:
    lines: tuple[tuple[str, float], ...]
    score: float


class PassportMRZExtractor:
    """Core TD3 passport MRZ extraction engine."""

    def __init__(self, ocr_engine):
        self.ocr = ocr_engine

    def extract(
        self,
        image,
        *,
        attempt: str,
    ) -> PassportMRZResult:
        try:
            ocr_results = self.ocr.ocr(image, det=True, rec=True, cls=False)
        except Exception:
            logger.exception("PaddleOCR invocation failed in PassportMRZExtractor")
            return self._not_found()

        if not ocr_results or not ocr_results[0]:
            return self._not_found()

        blocks = []
        for line in ocr_results[0]:
            bbox = line[0]
            text, confidence = line[1]
            cleaned = clean_mrz_line(text)
            if cleaned:
                blocks.append(
                    {
                        "text": cleaned,
                        "bbox": bbox,
                        "confidence": confidence,
                        "cy": sum(point[1] for point in bbox) / 4.0,
                        "xmin": min(point[0] for point in bbox),
                    }
                )
        if not blocks:
            return self._not_found()

        merged_lines = self._merge_blocks(blocks)
        merged_lines = self._prefer_mrz_like_lines(merged_lines)
        return self._select_best_result(merged_lines, attempt)

    def extract_recognition_lines(
        self,
        line_images: list,
        *,
        attempt: str,
    ) -> PassportMRZResult:
        """Recognize pre-cropped MRZ lines when text detection found nothing."""
        if len(line_images) != 2:
            return self._not_found()
        try:
            raw_results = self.ocr.ocr(
                line_images,
                det=False,
                rec=True,
                cls=False,
            )
        except Exception:
            logger.exception("PaddleOCR line recognition recovery failed")
            return self._not_found()

        if not raw_results:
            return self._not_found()

        lines: list[tuple[str, float]] = []

        def _flatten_rec(res):
            items = []
            if isinstance(res, list):
                for elem in res:
                    if isinstance(elem, list):
                        items.extend(_flatten_rec(elem))
                    elif isinstance(elem, tuple) and len(elem) == 2:
                        items.append(elem)
            return items

        for text, confidence in _flatten_rec(raw_results):
            cleaned = clean_mrz_line(str(text))
            if cleaned:
                lines.append((cleaned, float(confidence)))

        if len(lines) != 2:
            return self._diagnostic_result(lines)
        return self._select_best_result(lines, attempt)

    @classmethod
    def _strip_leading_noise(cls, text: str) -> str:
        if not text:
            return ""
        if cls._has_td3_line1_header(text):
            return text
        for offset in (1, 2, 3):
            if offset < len(text):
                sub = text[offset:]
                if cls._has_td3_line1_header(sub):
                    return sub
        return text

    @classmethod
    def _has_td3_line1_header(cls, text: str) -> bool:
        if not text or len(text) < 4:
            return False
        for noise in ("PASSPORT", "PASAPORTE", "PASSAPORTE", "PASSEPORT", "PASSAPORT", "REISEPASS"):
            if text.startswith(noise):
                return False
        return bool(re.match(r"^P[<A-Z0-9][A-Z0-9<]{2,3}", text))

    @classmethod
    def _mrz_line_score(cls, text: str) -> float:
        """Score how MRZ-like a cleaned OCR line is vs normal page prose."""
        if not text:
            return -10.0
        normalized = cls._strip_leading_noise(text)
        filler_ratio = normalized.count("<") / max(len(normalized), 1)
        has_line1_header = cls._has_td3_line1_header(normalized)
        has_name_sep = bool(re.search(r'[A-Z]<<[A-Z]', normalized))
        digit_ratio = sum(char.isdigit() for char in normalized) / max(len(normalized), 1)
        length_score = 10 if (has_line1_header and len(normalized) >= 15) else max(0, 10 - abs(len(normalized) - TD3_LINE_LENGTH))
        
        # Penalize lines containing passport header / data-page label words
        noise_count = sum(1 for w in PASSPORT_NOISE_WORDS if w in text)
        noise_penalty = noise_count * -15.0
        prose_penalty = (
            -15.0
            if len(text) > 35 and filler_ratio < 0.08 and digit_ratio < 0.3
            and not has_line1_header
            else 0.0
        )
        return (
            filler_ratio * 12
            + (12 if has_line1_header else 0)
            + (4 if has_name_sep else 0)
            + digit_ratio * 4
            + length_score
            + prose_penalty
            + noise_penalty
        )

    @classmethod
    def _prefer_mrz_like_lines(
        cls,
        merged_lines: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        if len(merged_lines) <= 2:
            return merged_lines
        scored = [
            (index, line, cls._mrz_line_score(line[0]))
            for index, line in enumerate(merged_lines)
        ]
        best_score = max(score for _, _, score in scored)
        threshold = max(best_score * 0.45, 6.0)
        selected = [
            (index, line)
            for index, line, score in scored
            if score >= threshold
        ]
        if len(selected) < 2:
            selected = [
                (index, line)
                for index, line, _ in sorted(
                    scored,
                    key=lambda item: item[2],
                    reverse=True,
                )[:2]
            ]
        selected.sort(key=lambda item: item[0])
        return [line for _, line in selected]

    @staticmethod
    def _merge_blocks(blocks: list[dict]) -> list[tuple[str, float]]:
        blocks.sort(key=lambda block: block["cy"])
        average_height = sum(
            max(point[1] for point in block["bbox"])
            - min(point[1] for point in block["bbox"])
            for block in blocks
        ) / len(blocks)
        grouped_lines: list[list[dict]] = []
        current_line: list[dict] = []
        for block in blocks:
            current_center = (
                sum(item["cy"] for item in current_line) / len(current_line)
                if current_line
                else block["cy"]
            )
            if (
                current_line
                and abs(block["cy"] - current_center)
                >= average_height * 0.55
            ):
                grouped_lines.append(current_line)
                current_line = []
            current_line.append(block)
        if current_line:
            grouped_lines.append(current_line)

        merged_lines = []
        for group in grouped_lines:
            group.sort(key=lambda block: block["xmin"])
            text = "".join(block["text"] for block in group)
            confidence = sum(block["confidence"] for block in group) / len(group)
            if len(text) >= MRZ_LINE_MIN_LENGTH:
                merged_lines.append((text, confidence))
        return merged_lines

    def _select_best_result(
        self,
        merged_lines: list[tuple[str, float]],
        attempt: str,
    ) -> PassportMRZResult:
        candidates = self._find_td3_candidates(merged_lines)
        if not candidates:
            return self._diagnostic_result(merged_lines)

        for candidate in candidates:
            result = self._parse_td3_lines(candidate.lines, attempt)
            if self.is_structurally_valid(result):
                return result
        return self._diagnostic_result(merged_lines)

    @classmethod
    def _find_td3_candidates(
        cls,
        merged_lines: list[tuple[str, float]],
    ) -> list[PassportMRZCandidate]:
        candidates: list[PassportMRZCandidate] = []
        if len(merged_lines) < 2:
            return candidates

        seen: set[tuple[str, str]] = set()
        for i in range(len(merged_lines) - 1):
            line1_raw, line1_conf = merged_lines[i]
            line1_text = cls._strip_leading_noise(line1_raw)
            if not cls._has_td3_line1_header(line1_text) and not (line1_text.startswith("P") and "<" in line1_text):
                continue
            for j in range(i + 1, len(merged_lines)):
                line2_text, line2_conf = merged_lines[j]
                key = (line1_text, line2_text)
                if key in seen:
                    continue
                seen.add(key)

                if not (15 <= len(line1_text) <= 55 and 20 <= len(line2_text) <= 55):
                    continue

                score = cls._score_td3_candidate(
                    ((line1_text, line1_conf), (line2_text, line2_conf)),
                    is_adjacent=(j == i + 1),
                )
                candidates.append(
                    PassportMRZCandidate(
                        lines=((line1_text, line1_conf), (line2_text, line2_conf)),
                        score=score,
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    @classmethod
    def _score_td3_candidate(
        cls,
        pair: tuple[tuple[str, float], tuple[str, float]],
        *,
        is_adjacent: bool = True,
    ) -> float:
        (line1_text, line1_conf), (line2_text, line2_conf) = pair
        score = 0.0

        if cls._has_td3_line1_header(line1_text):
            score += 25.0
        elif line1_text.startswith("P") or line1_text.startswith("P<"):
            score += 10.0

        score += max(0.0, 10.0 - abs(len(line1_text) - TD3_LINE_LENGTH) * 2)
        score += max(0.0, 10.0 - abs(len(line2_text) - TD3_LINE_LENGTH) * 2)

        fillers1 = line1_text.count("<")
        fillers2 = line2_text.count("<")
        score += min(fillers1, 20) * 0.5 + min(fillers2, 20) * 0.5

        norm_line2 = line2_text.ljust(TD3_LINE_LENGTH, "<")[:TD3_LINE_LENGTH]
        pass_num = norm_line2[0:9]
        pass_check = correct_ocr_digits(norm_line2[9:10])
        if pass_check.isdigit() and compute_mrz_check_digit(pass_num) == int(pass_check):
            score += 30.0

        comp_check = correct_ocr_digits(norm_line2[43:44])
        if comp_check.isdigit() and compute_td3_composite_check(norm_line2) == int(comp_check):
            score += 20.0

        dob = norm_line2[13:19]
        expiry = norm_line2[21:27]
        if is_valid_date_field(dob):
            score += 5.0
        if is_valid_date_field(expiry):
            score += 5.0

        avg_conf = (line1_conf + line2_conf) / 2.0
        score += avg_conf * 15.0

        if is_adjacent:
            score += 5.0

        # Penalize if Line 1 or Line 2 contains passport header/label words
        line1_noise = sum(1 for w in PASSPORT_NOISE_WORDS if w in line1_text)
        line2_noise = sum(1 for w in PASSPORT_NOISE_WORDS if w in line2_text)
        score -= (line1_noise + line2_noise) * 15.0

        return score

    def _parse_td3_lines(
        self,
        lines: tuple[tuple[str, float], ...],
        attempt: str,
    ) -> PassportMRZResult:
        line1_raw, line1_conf = lines[0]
        line2_raw, line2_conf = lines[1]

        line1 = line1_raw.ljust(TD3_LINE_LENGTH, "<")[:TD3_LINE_LENGTH]
        line2 = line2_raw.ljust(TD3_LINE_LENGTH, "<")[:TD3_LINE_LENGTH]

        name_raw = line1[5:44]
        surname, given_names = parse_td3_name(name_raw)

        passport_number_raw = line2[0:9]
        passport_check_raw = correct_ocr_digits(line2[9:10])
        nationality = line2[10:13]
        dob_raw = correct_ocr_digits(line2[13:19])
        dob_check_raw = correct_ocr_digits(line2[19:20])
        sex_raw = line2[20:21]
        expiry_raw = correct_ocr_digits(line2[21:27])
        expiry_check_raw = correct_ocr_digits(line2[27:28])
        personal_num_raw = line2[28:42]
        personal_check_raw = correct_ocr_digits(line2[42:43])
        composite_check_raw = correct_ocr_digits(line2[43:44])

        computed_pn_check = compute_mrz_check_digit(passport_number_raw)
        pn_checksum_valid = (
            passport_check_raw.isdigit()
            and computed_pn_check == int(passport_check_raw)
        )

        computed_comp_check = compute_td3_composite_check(line2)
        comp_checksum_valid = (
            composite_check_raw.isdigit()
            and computed_comp_check == int(composite_check_raw)
        )

        is_canonical = True
        passport_number = passport_number_raw.replace("<", "").strip() or None

        if not pn_checksum_valid and not is_valid_passport_number(passport_number):
            aze_idx = line2_raw.find("AZE")
            if aze_idx in (9, 11, 12):
                recovered_pn = line2_raw[:aze_idx - 1]
                recovered_check = line2_raw[aze_idx - 1 : aze_idx]
                check_digit = correct_ocr_digits(recovered_check)
                if check_digit.isdigit() and compute_mrz_check_digit(recovered_pn) == int(check_digit):
                    passport_number = recovered_pn.replace("<", "").strip()
                    pn_checksum_valid = True
                    is_canonical = False

        dob = dob_raw if is_valid_date_field(dob_raw) else None
        expiry = expiry_raw if is_valid_date_field(expiry_raw) else None
        sex = sex_raw if sex_raw in ("M", "F", "<") else None
        personal_number = personal_num_raw.replace("<", "").strip() or None

        avg_confidence = round((line1_conf + line2_conf) / 2.0, 4)
        quality_score = (
            (1.0 if pn_checksum_valid else 0.0)
            + (1.0 if comp_checksum_valid else 0.0)
            + (1.0 if is_canonical else 0.5)
            + avg_confidence
        )

        return PassportMRZResult(
            line1=line1,
            line2=line2,
            confidence=avg_confidence,
            checksum_valid=pn_checksum_valid,
            composite_checksum_valid=comp_checksum_valid,
            method=attempt,
            passport_number=passport_number,
            nationality=nationality if nationality.replace("<", "") else None,
            date_of_birth=dob,
            sex=sex,
            expiry_date=expiry,
            personal_number=personal_number,
            surname=surname,
            given_names=given_names,
            line_confidences=(round(line1_conf, 4), round(line2_conf, 4)),
            quality_score=quality_score,
            is_canonical=is_canonical,
        )

    @staticmethod
    def _is_likely_line2(text: str) -> bool:
        """Check if a single line structurally resembles MRZ Line 2 rather than Line 1."""
        if len(text) < 30:
            return False
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        norm_line2 = text.ljust(TD3_LINE_LENGTH, "<")[:TD3_LINE_LENGTH]
        pass_num = norm_line2[0:9]
        pass_check = correct_ocr_digits(norm_line2[9:10])
        has_valid_pn_check = (
            pass_check.isdigit()
            and compute_mrz_check_digit(pass_num) == int(pass_check)
        )
        comp_check = correct_ocr_digits(norm_line2[43:44])
        has_valid_comp_check = (
            comp_check.isdigit()
            and compute_td3_composite_check(norm_line2) == int(comp_check)
        )

        return (
            (digit_ratio > 0.35 or has_valid_pn_check or has_valid_comp_check)
            and "<<" not in text[:35]
        )

    def _diagnostic_result(
        self,
        merged_lines: list[tuple[str, float]],
    ) -> PassportMRZResult:
        if len(merged_lines) == 1 and self._is_likely_line2(merged_lines[0][0]):
            line1 = ""
            line2 = merged_lines[0][0].ljust(TD3_LINE_LENGTH, "<")[:TD3_LINE_LENGTH]
        else:
            line1 = merged_lines[0][0].ljust(TD3_LINE_LENGTH, "<")[:TD3_LINE_LENGTH] if merged_lines else ""
            line2 = merged_lines[1][0].ljust(TD3_LINE_LENGTH, "<")[:TD3_LINE_LENGTH] if len(merged_lines) > 1 else ""
        conf = sum(l[1] for l in merged_lines) / max(len(merged_lines), 1) if merged_lines else 0.0

        return PassportMRZResult(
            line1=line1,
            line2=line2,
            confidence=round(conf, 4),
            checksum_valid=False,
            composite_checksum_valid=False,
            method="diagnostic",
            passport_number=None,
            nationality=None,
            date_of_birth=None,
            sex=None,
            expiry_date=None,
            personal_number=None,
            surname=None,
            given_names=None,
            line_confidences=tuple(round(l[1], 4) for l in merged_lines),
            quality_score=0.0,
            is_canonical=False,
        )

    def _not_found(self) -> PassportMRZResult:
        return PassportMRZResult(
            line1="",
            line2="",
            confidence=0.0,
            checksum_valid=False,
            composite_checksum_valid=False,
            method="failed",
            passport_number=None,
            nationality=None,
            date_of_birth=None,
            sex=None,
            expiry_date=None,
            personal_number=None,
            surname=None,
            given_names=None,
            line_confidences=(),
            quality_score=0.0,
            is_canonical=False,
        )

    @classmethod
    def is_structurally_valid(cls, result: PassportMRZResult) -> bool:
        """A TD3 result is valid when lines are 44 chars, Line 1 has a standard P-header, and Line 2 has valid checksums or dates."""
        if not result or not result.line1 or not result.line2:
            return False
        if len(result.line1) != TD3_LINE_LENGTH or len(result.line2) != TD3_LINE_LENGTH:
            return False
        if not cls._has_td3_line1_header(result.line1):
            return False
        if any(w in result.line1 for w in ("SURNAME", "APELLIDO", "PELLIDO", "PASSPORT", "PASAPORTE", "NACIONALIDAD", "SEXO")):
            return False

        has_checksum = bool(result.checksum_valid or result.composite_checksum_valid)
        has_valid_dates = bool(result.date_of_birth and result.expiry_date)
        has_passport_num = is_valid_passport_number(result.passport_number) and not (result.passport_number or "").startswith("P")

        return has_checksum or (has_valid_dates and has_passport_num)
