"""Fixed MRZ field positions for ICAO Doc 9303 TD3 passports.

Passport MRZ is 2 lines x 44 characters.
Line 1: [0] Doc Type ('P'), [1] Subtype, [2:5] Issuing Country ('AZE'), [5:44] Name
Line 2: [0:9] Passport Number, [9] Check, [10:13] Nationality, [13:19] DOB, [19] Check,
        [20] Sex, [21:27] Expiry, [27] Check, [28:42] Personal Number, [42] Check, [43] Composite Check
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSlice:
    """Inclusive-exclusive character window on one MRZ line."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def read(self, line_text: str) -> str:
        if self.end > len(line_text):
            return ""
        return line_text[self.start : self.end]


TD3_LINE_LENGTH = 44

# Line 1 fields
TD3_DOC_TYPE = FieldSlice(0, 1)          # "P"
TD3_DOC_SUBTYPE = FieldSlice(1, 2)       # Subtype or "<"
TD3_ISSUING_COUNTRY = FieldSlice(2, 5)   # "AZE"
TD3_NAME = FieldSlice(5, 44)             # SURNAME<<GIVEN<NAMES<<<...

# Line 2 fields
TD3_PASSPORT_NUMBER = FieldSlice(0, 9)
TD3_PASSPORT_CHECK = FieldSlice(9, 10)
TD3_NATIONALITY = FieldSlice(10, 13)
TD3_DOB = FieldSlice(13, 19)
TD3_DOB_CHECK = FieldSlice(19, 20)
TD3_SEX = FieldSlice(20, 21)
TD3_EXPIRY = FieldSlice(21, 27)
TD3_EXPIRY_CHECK = FieldSlice(27, 28)
TD3_PERSONAL_NUMBER = FieldSlice(28, 42)
TD3_PERSONAL_CHECK = FieldSlice(42, 43)
TD3_COMPOSITE_CHECK = FieldSlice(43, 44)
