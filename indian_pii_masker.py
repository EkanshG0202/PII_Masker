"""
Indian PII Masker
=================
Entities masked
───────────────────────────────────────────────────────────
  AADHAAR        12-digit UID  (4-4-4, any separator)
  PAN            Permanent Account Number  (AAAAA9999A)
  GST            GSTIN  (29ABCDE1234F1Z5)
  IFSC           Bank IFSC code  (HDFC0001234)
  PHONE          Indian mobile numbers (all common formats)
  ACCOUNT        Bank account numbers (context-anchored, 11-18 digits)
  VOTER_ID       EPIC number  (3 letters + 7 digits)
  PASSPORT       Indian passport  (1 letter + 7 digits)
  DL             Driving licence  (SS-RR-YYYY-NNNNNNN)
#   UDYAM          MSME Udyam registration  (UDYAM-XX-DD-NNNNNNN)
  UAN            EPFO Universal Account Number (context-anchored, 12 digits)
  EMAIL          E-mail addresses

Ambiguous formats supported (NEW)
───────────────────────────────────────────────────────────
  All entities now tolerate arbitrary separators (spaces, dashes, dots,
  underscores, slashes) inserted between ANY characters, e.g.:
    E C P P G 0 1 1 1 K       ← space after every character
    E-C-P-P-G-0-1-1-1-K       ← dash after every character
    ECPP G0 111K               ← random groupings
    [Aadhaar Redacted]         ← dashes instead of spaces in Aadhaar
    98765 43210                ← phone split randomly
    MH-27-2012-0034761         ← DL with extra dashes

  A pre-processing step (_normalize_dense_separators) collapses
  sequences where every character is followed by a separator — the
  dominant "manual obfuscation" pattern — into compact tokens before
  the main analysis runs.

Setup
─────
    pip install presidio-analyzer presidio-anonymizer spacy psutil
    python -m spacy download en_core_web_lg
"""

import re
import time
import os
import psutil

from presidio_analyzer import (
    AnalyzerEngine,
    PatternRecognizer,
    Pattern,
    RecognizerResult,
)
from presidio_analyzer.entity_recognizer import EntityRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# =========================================================
# SEPARATOR CONSTANT — inserted between every character/group
# in "ambiguous" regex patterns
# =========================================================

# Matches 0-to-many of: space, tab, dash, dot, underscore, slash, pipe
_S = r'[\s.\-_/|]{0,3}'

# =========================================================
# HELPERS
# =========================================================

def _norm(text: str) -> str:
    """Strip everything except letters and digits (case-preserved)."""
    return re.sub(r'[^A-Za-z0-9]', '', text)


def _sep_pattern(strict: str) -> str:
    """
    Given a strict regex like r'[A-Z]{4}0[A-Z0-9]{6}', inject _S between
    every atom so it tolerates arbitrary separators between characters.

    Handles character classes [...], quantified groups {n}, and bare chars.
    Each top-level atom is kept; _S is inserted between consecutive atoms.

    This is a best-effort approach sufficient for fixed-length ID patterns.
    For more complex patterns, recognizers build their own explicitly.
    """
    # Tokenise into atoms: [...], {n}, bare letter/digit/escaped, ^/$
    token_re = re.compile(
        r'(\[\^?[^\]]*\]\{?\d*,?\d*\}?'   # [class]{n}
        r'|\[\^?[^\]]*\]'                 # [class]
        r'|\([^)]*\)\{?\d*,?\d*\}?'        # (group){n}
        r'|\{?\d+,?\d*\}'                  # bare {n}
        r'|\\.'                              # escaped char
        r'|[^^$.|?*+(){}\\]'                # bare char (not meta)
        r'|[.^$|?*+(){}\\]'                 # meta
        r')'
    )
    tokens = token_re.findall(strict)
    return _S.join(tokens)


# =========================================================
# PRE-PROCESSOR — collapse "space/dash after every char" patterns
# =========================================================

def _normalize_dense_separators(text: str) -> str:
    """
    Collapse sequences where nearly every alphanumeric character is
    separated by a consistent single separator, e.g.:

        "E C P P G 0 1 1 1 K"  →  "ECPPG0111K"
        "E-C-P-P-G-0-1-1-1-K"  →  "ECPPG0111K"
        "2 3 4 5  6 7 8 9  0 1 2 3"  →  "2345678901 23"  (only tight runs)

    Algorithm:
      - Find maximal runs of (single-alnum)(single-separator) followed
        by a final alnum — at least 4 characters long.
      - Replace each run with its stripped version.
      - Preserve surrounding context so span offsets remain consistent
        for the result text (positions shift, but that is fine because
        Presidio operates on the normalised copy).
    """
    # Pattern: alnum, then (sep, alnum) repeated ≥3 times
    # sep = exactly one non-alnum non-newline char (space, dash, dot, underscore…)
    dense = re.compile(
        r'(?<![A-Za-z0-9])'        # not preceded by alnum (word boundary)
        r'([A-Za-z0-9]'            # first char
        r'(?:[^A-Za-z0-9\n][A-Za-z0-9]){3,})'  # (sep + alnum) × 3+
        r'(?![A-Za-z0-9])'         # not followed by alnum
    )

    def _strip(m):
        return re.sub(r'[^A-Za-z0-9]', '', m.group(0))

    return dense.sub(_strip, text)


# =========================================================
# VALIDATORS  (unchanged — all use _norm internally)
# =========================================================

def _valid_aadhaar(text: str) -> bool:
    t = _norm(text)
    return len(t) == 12 and t.isdigit() and t[0] not in "01"


def _valid_pan(text: str) -> bool:
    return bool(re.fullmatch(r'[A-Z]{5}[0-9]{4}[A-Z]', _norm(text).upper()))


def _valid_ifsc(text: str) -> bool:
    return bool(re.fullmatch(r'[A-Z]{4}0[A-Z0-9]{6}', _norm(text).upper()))


def _valid_gst(text: str) -> bool:
    return bool(re.fullmatch(
        r'\d{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]',
        _norm(text).upper(),
    ))


def _valid_phone(text: str) -> bool:
    t = _norm(text)
    if   t.startswith("91")  and len(t) == 12: t = t[2:]
    elif t.startswith("091") and len(t) == 13: t = t[3:]
    elif t.startswith("0")   and len(t) == 11: t = t[1:]
    return len(t) == 10 and t.isdigit() and t[0] in "6789"


def _valid_account(text: str) -> bool:
    digits = re.search(r'\d{11,18}', _norm(text))
    if not digits:
        return False
    t = digits.group()
    return t.isdigit() and 11 <= len(t) <= 18


def _valid_voter_id(text: str) -> bool:
    return bool(re.fullmatch(r'[A-Z]{3}[0-9]{7}', _norm(text).upper()))


def _valid_passport(text: str) -> bool:
    t = _norm(text).upper()
    return bool(re.fullmatch(r'[A-Z][1-9]\d{5}[1-9]', t))


def _valid_dl(text: str) -> bool:
    t = _norm(text).upper()
    return bool(re.fullmatch(r'[A-Z]{2}\d{2}(19|20)\d{2}\d{7}', t))


# def _valid_udyam(text: str) -> bool:
#     return bool(re.fullmatch(
#         r'UDYAM[A-Z]{2}\d{2}\d{7}',
#         _norm(text).upper(),
#     ))


def _valid_uan(text: str) -> bool:
    digits = re.search(r'\d{12}', _norm(text))
    return bool(digits)


# =========================================================
# RECOGNIZERS
# =========================================================

class CustomEmailRecognizer(PatternRecognizer):
    """
    Explicit email recognizer with strict fallbacks for typos and obfuscation.
    Tightened to prevent false positives on regular English text.
    """
    def __init__(self):
        super().__init__(
            supported_entity="EMAIL_ADDRESS",
            patterns=[
                # 1. Standard well-formed email
                Pattern(
                    "email_standard", 
                    r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', 
                    score=1.0
                ),
                
                # 2. Missing '@' but attached directly to a known provider (e.g., ramesh1972gmail.com)
                Pattern(
                    "email_missing_at", 
                    r'(?i)\b[A-Z0-9._%+-]+(?:gmail|yahoo|outlook|hotmail|rediffmail)\.com\b', 
                    score=0.85
                ),
                
                # 3. Explicit bracket obfuscation (e.g., user[at]domain[dot]com)
                Pattern(
                    "email_obfuscated_brackets", 
                    r'(?i)\b[A-Z0-9._%+-]+\s*(?:\[at\]|\(at\))\s*[A-Z0-9.-]+\s*(?:\[dot\]|\(dot\)|\.)\s*(?:com|in|co\.in|org|net)\b', 
                    score=0.80
                ),

                # 4. Spelled out words (e.g., user at gmail dot com)
                Pattern(
                    "email_obfuscated_words", 
                    r'(?i)\b[A-Z0-9._%+-]+\s+at\s+(?:gmail|yahoo|outlook|hotmail|rediffmail)\s+(?:dot|\.)\s+(?:com|in|co\.in|org|net)\b', 
                    score=0.80
                ),
                
                # 5. Missing '@' replaced by a space (e.g., jayjagannath5press gmail.com)
                # {3,} requires the username to be 3+ chars to avoid masking "to gmail.com"
                Pattern(
                    "email_space_missing_at",
                    r'(?i)\b[A-Z0-9._%+-]{3,}\s+(?:gmail|yahoo|outlook|hotmail|rediffmail)\.com\b',
                    score=0.80
                )
            ],
        )

class AadhaarRecognizer(PatternRecognizer):
    """
    Matches Aadhaar in all separator variants:
      • 4-4-4 with standard separators  (original)
      • 4-4-4 with arbitrary separators per group
      • Every digit separated individually: [Aadhaar Redacted]
      • Mixed: [Aadhaar Redacted]
    """
    _G4  = r'\d' + _S + r'\d' + _S + r'\d' + _S + r'\d'   # 4 digits with seps
    _SEP = r'[\s.\-_()/]{0,3}'

    def __init__(self):
        g4 = self._G4
        sep = self._SEP
        super().__init__(
            supported_entity="AADHAAR",
            patterns=[
                # Original: 4-4-4 grouped
                Pattern("aadhaar_4_4_4",
                        r'(?<!\d)\d{4}' + sep + r'\d{4}' + sep + r'\d{4}(?!\d)',
                        score=0.85),
                # Fully separated: every digit has a separator
                Pattern("aadhaar_separated",
                        r'(?<![A-Za-z0-9])' + g4 + sep + g4 + sep + g4 + r'(?![A-Za-z0-9])',
                        score=0.80),
                # Bare 12 digits (context-anchored)
                Pattern("aadhaar_12_bare",
                        r'(?<!\d)\d{12}(?!\d)',
                        score=0.55),
            ],
            context=["aadhaar", "aadhar", "uid", "unique identification"],
        )

    def validate_result(self, pattern_text):
        return _valid_aadhaar(pattern_text)


class PANRecognizer(PatternRecognizer):
    """
    PAN card: AAAAA9999A
    Tolerates any separator between every character.
    Examples:
      ECPPG0111K  /  E C P P G 0 1 1 1 K  /  E-C-P-P-G-0-1-1-1-K
      ECPP-G01-11K  /  E.C.P.P.G.0.1.1.1.K
    """
    # 5 alpha, 4 digit, 1 alpha — each char separated by _S
    _ALPHA = r'[A-Za-z]'
    _DIGIT = r'[0-9]'

    def __init__(self):
        a, d, s = self._ALPHA, self._DIGIT, _S
        pan_sep = (
            r'(?<![A-Za-z0-9])'
            + a + s + a + s + a + s + a + s + a  # 5 alpha
            + s
            + d + s + d + s + d + s + d           # 4 digit
            + s
            + a                                    # 1 alpha
            + r'(?![A-Za-z0-9])'
        )
        super().__init__(
            supported_entity="PAN",
            patterns=[
                Pattern("pan_strict",
                        r'(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])',
                        score=0.90),
                Pattern("pan_separated",
                        pan_sep,
                        score=0.85),
            ],
        )

    def validate_result(self, pattern_text):
        return _valid_pan(pattern_text)


class GSTRecognizer(PatternRecognizer):
    """
    GSTIN: 2-digit state + PAN(10) + 1 + Z + 1 = 15 chars
    Tolerates separators between every character.
    """
    _D = r'[0-9]'
    _A = r'[A-Za-z]'
    _AN = r'[A-Za-z0-9]'

    def __init__(self):
        d, a, an, s = self._D, self._A, self._AN, _S

        gst_sep = (
            r'(?<![A-Za-z0-9])'
            # 2-digit state code
            + d + s + d + s
            # 5 alpha (PAN part)
            + a + s + a + s + a + s + a + s + a + s
            # 4 digit
            + d + s + d + s + d + s + d + s
            # 1 alpha
            + a + s
            # 1 alphanumeric (1-9 or A-Z)
            + an + s
            # literal Z
            + r'[Zz]' + s
            # 1 alphanumeric checksum
            + an
            + r'(?![A-Za-z0-9])'
        )
        super().__init__(
            supported_entity="GST",
            patterns=[
                Pattern("gst_strict",
                        r'(?<!\w)\d{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z](?!\w)',
                        score=0.95),
                Pattern("gst_separated",
                        gst_sep,
                        score=0.88),
            ],
        )

    def validate_result(self, pattern_text):
        return _valid_gst(pattern_text)


class IFSCRecognizer(PatternRecognizer):
    """
    IFSC: 4 alpha + 0 + 6 alphanumeric
    Tolerates separators between every character.
    Examples:
      HDFC0001234  /  H D F C 0 0 0 1 2 3 4  /  HDFC-0-001234
    """
    _A = r'[A-Za-z]'
    _AN = r'[A-Za-z0-9]'

    def __init__(self):
        a, an, s = self._A, self._AN, _S
        ifsc_sep = (
            r'(?<![A-Za-z0-9])'
            + a + s + a + s + a + s + a   # 4 alpha
            + s + r'0' + s                # literal 0
            + an + s + an + s + an + s + an + s + an + s + an  # 6 alphanum
            + r'(?![A-Za-z0-9])'
        )
        super().__init__(
            supported_entity="IFSC",
            patterns=[
                Pattern("ifsc_strict",
                        r'(?<![A-Z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Z0-9])',
                        score=0.90),
                Pattern("ifsc_separated",
                        ifsc_sep,
                        score=0.85),
            ],
        )

    def validate_result(self, pattern_text):
        return _valid_ifsc(pattern_text)


class IndianPhoneRecognizer(PatternRecognizer):
    """
    All common Indian mobile formats + fully separated digit runs.
    Examples:
      9876543210  /  98765 43210  /  9 8 7 6 5 4 3 2 1 0
      +91-98765-43210  /  +91 9 8 7 6 5 4 3 2 1 0
    """
    _CC  = r'(?:(?:\+|0{0,2})91[\s()\-]*)?'
    _S10 = (                                                 # 10 digits, any sep
        r'[6-9]' + _S
        + r'\d' + _S + r'\d' + _S + r'\d' + _S + r'\d'   # 5 digits
        + _S
        + r'\d' + _S + r'\d' + _S + r'\d' + _S + r'\d' + _S + r'\d'  # 5 digits
    )
    _P1  = r'(?<!\d)' + _CC + r'[6-9]\d{4}[\s\-.]?\d{5}(?!\d)'
    _P2  = r'(?:(?:\+|0{0,2})91[\s]*)\(\d{3}\)[\s\-]*\d{3}[\s\-]*\d{4}'
    _P3  = r'(?<!\d)0\d{2}[\s\-.]?\d{3}[\s\-.]?\d{5}(?!\d)'
    _P4  = r'(?<!\d)[6-9]\d{2}[\s\-]\d{3}[\s\-]\d{4}(?!\d)'

    def __init__(self):
        super().__init__(
            supported_entity="PHONE",
            patterns=[
                Pattern("phone_p1", self._P1, score=0.85),
                Pattern("phone_p2", self._P2, score=0.85),
                Pattern("phone_p3", self._P3, score=0.80),
                Pattern("phone_p4", self._P4, score=0.75),
                # Fully separated 10-digit run (with optional country code)
                Pattern("phone_sep10",
                        r'(?<![A-Za-z0-9])' + self._CC + self._S10 + r'(?![A-Za-z0-9])',
                        score=0.78),
            ],
        )

    def validate_result(self, pattern_text):
        return _valid_phone(pattern_text)


class VoterIdRecognizer(PatternRecognizer):
    """
    EPIC: 3 uppercase letters + 7 digits
    Tolerates separators between every character.
    Examples:
      XGN3002623  /  X G N 3 0 0 2 6 2 3  /  XGN-3002623  /  X-G-N-3-0-0-2-6-2-3
    """
    _A = r'[A-Za-z]'
    _D = r'[0-9]'

    def __init__(self):
        a, d, s = self._A, self._D, _S
        voter_sep = (
            r'(?<![A-Za-z0-9])'
            + a + s + a + s + a   # 3 alpha
            + s
            + d + s + d + s + d + s + d + s + d + s + d + s + d  # 7 digits
            + r'(?![A-Za-z0-9])'
        )
        super().__init__(
            supported_entity="VOTER_ID",
            patterns=[
                Pattern("voter_id_strict",
                        r'(?<![A-Z0-9])[A-Z]{3}[0-9]{7}(?![A-Z0-9])',
                        score=0.85),
                Pattern("voter_id_separated",
                        voter_sep,
                        score=0.80),
            ],
        )

    def validate_result(self, pattern_text):
        return _valid_voter_id(pattern_text)


class PassportRecognizer(PatternRecognizer):
    """
    Indian passport: 1 letter + 7 digits (first & last digit non-zero)
    Tolerates separators between every character.
    Examples:
      A2345671  /  A 2 3 4 5 6 7 1  /  A-2345671  /  A-2-3-4-5-6-7-1
    """
    _A = r'[A-Za-z]'
    _NZ = r'[1-9]'   # non-zero digit
    _D  = r'[0-9]'

    def __init__(self):
        a, nz, d, s = self._A, self._NZ, self._D, _S
        pp_sep = (
            r'(?<![A-Za-z0-9])'
            + a + s + nz + s          # letter + non-zero
            + d + s + d + s + d + s + d + s + d  # 5 middle digits
            + s + nz                   # non-zero last
            + r'(?![A-Za-z0-9])'
        )
        super().__init__(
            supported_entity="PASSPORT",
            patterns=[
                Pattern("passport_strict",
                        r'(?<![A-Z0-9])[A-Z][1-9]\d{5}[1-9](?![A-Z0-9])',
                        score=0.85),
                Pattern("passport_separated",
                        pp_sep,
                        score=0.80),
            ],
        )

    def validate_result(self, pattern_text):
        return _valid_passport(pattern_text)


class DrivingLicenceRecognizer(PatternRecognizer):
    """
    Indian DL: SS-RR-YYYY-NNNNNNN (2+2+4+7 = 15 chars)
    Original pattern already tolerates some separators; now also matches
    fully character-separated variants.
    Examples:
      MH272012 0034761  /  MH-27-2012-0034761  /  M H 2 7 2 0 1 2 0 0 3 4 7 6 1
    """
    _DL_STRICT = (
        r'(?<![A-Z0-9])'
        r'[A-Z]{2}[\s\-]?\d{2}[\s\-]?(19|20)\d{2}[\s\-]?\d{7}'
        r'(?!\d)'
    )
    _A = r'[A-Za-z]'
    _D = r'[0-9]'

    def __init__(self):
        a, d, s = self._A, self._D, _S
        dl_sep = (
            r'(?<![A-Za-z0-9])'
            + a + s + a                                       # 2-char state
            + s
            + d + s + d                                       # 2-digit RTO
            + s
            + r'(?:19|20)' + s + d + s + d           # year
            + s
            + d + s + d + s + d + s + d + s + d + s + d + s + d  # 7-digit serial
            + r'(?![A-Za-z0-9])'
        )
        super().__init__(
            supported_entity="DL",
            patterns=[
                Pattern("dl_strict", self._DL_STRICT, score=0.85),
                Pattern("dl_separated", dl_sep, score=0.80),
            ],
        )

    def validate_result(self, pattern_text):
        return _valid_dl(pattern_text)


# class UdyamRecognizer(PatternRecognizer):
#     """
#     UDYAM-XX-DD-NNNNNNN
#     The UDYAM prefix is distinctive; also tolerates separators within each segment.
#     Examples:
#       UDYAM-DL-04-0012345  /  U D Y A M - D L - 0 4 - 0 0 1 2 3 4 5
#     """
#     def __init__(self):
#         s = _S
#         udyam_sep = (
#             r'(?i)(?<!\w)'
#             r'U' + s + r'D' + s + r'Y' + s + r'A' + s + r'M'
#             + s + r'[\-]?' + s
#             + r'[A-Za-z]' + s + r'[A-Za-z]'
#             + s + r'[\-]?' + s
#             + r'[0-9]' + s + r'[0-9]'
#             + s + r'[\-]?' + s
#             + r'[0-9]' + s + r'[0-9]' + s + r'[0-9]' + s
#             + r'[0-9]' + s + r'[0-9]' + s + r'[0-9]' + s + r'[0-9]'
#             + r'(?!\w)'
#         )
#         super().__init__(
#             supported_entity="UDYAM",
#             patterns=[
#                 Pattern("udyam_strict",
#                         r'(?i)(?<!\w)UDYAM[\-][A-Z]{2}[\-]\d{2}[\-]\d{7}(?!\w)',
#                         score=0.99),
#                 Pattern("udyam_separated",
#                         udyam_sep,
#                         score=0.92),
#             ],
#         )
#
#     def validate_result(self, pattern_text):
#         return _valid_udyam(pattern_text)


# =========================================================
# CONTEXT-ANCHORED RECOGNIZERS (unchanged logic, but digit
# patterns now also tolerate separators within digit spans)
# =========================================================

class AccountNumberRecognizer(EntityRecognizer):
    """
    Indian bank account numbers, 11–18 digits.
    Also matches digits written with spaces/dashes between them
    when preceded by an account-number keyword.
    Only the digit span is masked.
    """
    # Digit string with optional single separators between digits (11–18 digits)
    _DIG_SEP = r'\d(?:[\s.\-_]?\d){10,17}'

    _FULL = re.compile(
        r'(?i)(?:'
            r'account\s*(?:no\.?|number|#)?'
            r'|a\s*[/\-]?\s*c\s*(?:no\.?)?'
            r'|acct\.?'
            r'|acc\b\.?'
        r')\s*[:\-#]?\s*'
        r'(' + _DIG_SEP + r')',
    )

    def __init__(self):
        super().__init__(
            supported_entities=["ACCOUNT_NUMBER"],
            name="AccountNumberRecognizer",
        )

    def load(self):
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for m in self._FULL.finditer(text):
            digit_text = m.group(1)
            if not _valid_account(digit_text):
                continue
            results.append(RecognizerResult(
                entity_type="ACCOUNT_NUMBER",
                start=m.start(1),
                end=m.end(1),
                score=0.88,
            ))
        return results


class UANRecognizer(EntityRecognizer):
    """
    EPFO Universal Account Number — 12 digits.
    Also matches digits with separators when preceded by a UAN keyword.
    Only the digit span is masked.
    """
    _DIG_SEP = r'\d(?:[\s.\-_]?\d){11}'

    _FULL = re.compile(
        r'(?i)(?:'
            r'uan'
            r'|universal\s+account\s*(?:no\.?|number)?'
            r'|epfo\s*(?:no\.?|number)?'
            r'|pf\s+(?:uan|account)\s*(?:no\.?|number)?'
        r')\s*[:\-]?\s*'
        r'(' + _DIG_SEP + r')',
    )

    def __init__(self):
        super().__init__(
            supported_entities=["UAN"],
            name="UANRecognizer",
        )

    def load(self):
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for m in self._FULL.finditer(text):
            digit_text = m.group(1)
            if not _valid_uan(digit_text):
                continue
            results.append(RecognizerResult(
                entity_type="UAN",
                start=m.start(1),
                end=m.end(1),
                score=0.92,
            ))
        return results


# =========================================================
# OVERLAP RESOLUTION
# =========================================================

def _resolve_overlaps(results: list) -> list:
    """Keep the best (highest score, then longest span) non-overlapping results."""
    sorted_r = sorted(results, key=lambda r: (r.score, r.end - r.start), reverse=True)
    kept: list[RecognizerResult] = []
    for candidate in sorted_r:
        if not any(
            not (candidate.end <= k.start or candidate.start >= k.end)
            for k in kept
        ):
            kept.append(candidate)
    return kept


# =========================================================
# WHITESPACE REPAIR
# =========================================================

def _repair_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'(\w)\[', r'\1 [', text)
    text = re.sub(r'\]\s+([,.:;!?])', r']\1', text)
    return text


# =========================================================
# ENGINE SETUP
# =========================================================

def _build_analyzer() -> AnalyzerEngine:
    engine = None
    for model in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        try:
            import spacy
            spacy.load(model)
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model}],
            })
            engine = AnalyzerEngine(nlp_engine=provider.create_engine())
            print(f"[masking] spaCy model loaded: {model}")
            break
        except Exception:
            pass

    if engine is None:
        print("[masking] No spaCy model — regex-only mode.")
        engine = AnalyzerEngine()

    for cls in (
        CustomEmailRecognizer,
        AadhaarRecognizer,
        PANRecognizer,
        GSTRecognizer,
        IFSCRecognizer,
        IndianPhoneRecognizer,
        AccountNumberRecognizer,
        VoterIdRecognizer,
        PassportRecognizer,
        DrivingLicenceRecognizer,
        # UdyamRecognizer,
        UANRecognizer,
    ):
        engine.registry.add_recognizer(cls())

    return engine


_analyzer   = _build_analyzer()
_anonymizer = AnonymizerEngine()

_ENTITIES = [
    "EMAIL_ADDRESS",
    "AADHAAR", "PAN", "GST", "IFSC",
    "PHONE", "ACCOUNT_NUMBER",
    "VOTER_ID", "PASSPORT", "DL",
    # "UDYAM",
    "UAN",
]

_OPERATORS = {
    "EMAIL_ADDRESS":  OperatorConfig("replace", {"new_value": "[EMAIL]"}),
    "AADHAAR":        OperatorConfig("replace", {"new_value": "[AADHAAR]"}),
    "PAN":            OperatorConfig("replace", {"new_value": "[PAN]"}),
    "GST":            OperatorConfig("replace", {"new_value": "[GST]"}),
    "IFSC":           OperatorConfig("replace", {"new_value": "[IFSC]"}),
    "PHONE":          OperatorConfig("replace", {"new_value": "[PHONE]"}),
    "ACCOUNT_NUMBER": OperatorConfig("replace", {"new_value": "[ACCOUNT]"}),
    "VOTER_ID":       OperatorConfig("replace", {"new_value": "[VOTER_ID]"}),
    "PASSPORT":       OperatorConfig("replace", {"new_value": "[PASSPORT]"}),
    "DL":             OperatorConfig("replace", {"new_value": "[DL]"}),
    # "UDYAM":          OperatorConfig("replace", {"new_value": "[UDYAM]"}),
    "UAN":            OperatorConfig("replace", {"new_value": "[UAN]"}),
}


# =========================================================
# RESOURCE MONITOR
# =========================================================

class ResourceMonitor:
    """
    Context manager that measures CPU time, wall time, memory delta,
    peak memory, and thread count for any block of code.
    """

    def __init__(self, label: str = "block", print_report: bool = True):
        self.label        = label
        self.print_report = print_report
        self.stats: dict  = {}
        self._proc        = psutil.Process(os.getpid())

    def __enter__(self):
        mem_info          = self._proc.memory_info()
        self._wall_start  = time.perf_counter()
        self._cpu_start   = self._proc.cpu_times()
        self._mem_start   = mem_info.rss
        return self

    def __exit__(self, *_):
        wall_end  = time.perf_counter()
        cpu_end   = self._proc.cpu_times()
        mem_info  = self._proc.memory_info()

        wall_elapsed  = wall_end - self._wall_start
        cpu_user      = cpu_end.user  - self._cpu_start.user
        cpu_sys       = cpu_end.system - self._cpu_start.system
        mem_current   = mem_info.rss
        mem_delta     = mem_current - self._mem_start
        threads       = self._proc.num_threads()

        self.stats = {
            "label":            self.label,
            "wall_time_s":      round(wall_elapsed,  4),
            "cpu_user_s":       round(cpu_user,      4),
            "cpu_sys_s":        round(cpu_sys,       4),
            "mem_rss_mb":       round(mem_current  / 1024 / 1024, 2),
            "mem_delta_mb":     round(mem_delta     / 1024 / 1024, 2),
            "threads":          threads,
        }

        if self.print_report:
            self._print()

    def _print(self):
        s = self.stats
        print(
            f"\n{'─' * 50}\n"
            f"  Resource usage — {s['label']}\n"
            f"{'─' * 50}\n"
            f"  Wall time      : {s['wall_time_s']:.4f} s\n"
            f"  CPU user       : {s['cpu_user_s']:.4f} s\n"
            f"  CPU system     : {s['cpu_sys_s']:.4f} s\n"
            f"  Memory (RSS)   : {s['mem_rss_mb']:.2f} MB\n"
            f"  Memory delta   : {s['mem_delta_mb']:+.2f} MB\n"
            f"  Threads        : {s['threads']}\n"
            f"{'─' * 50}"
        )


# =========================================================
# PUBLIC API
# =========================================================

def mask_pii(text: str, monitor: bool = False) -> str:
    """
    Detect and replace Indian PII in *text*. Returns the anonymised string.

    Pipeline:
      1. _normalize_dense_separators  — collapse "char-sep-char-sep" runs
         so the main recognizers see compact tokens.
      2. Presidio analyze on the (possibly normalized) text.
      3. Overlap resolution.
      4. Presidio anonymize.
      5. Whitespace repair.

    Args:
        text:    Input string to mask.
        monitor: If True, prints a resource-usage report after each call.
    """
    with ResourceMonitor("mask_pii", print_report=monitor) as _mon:
        normalised = _normalize_dense_separators(text)
        raw        = _analyzer.analyze(text=normalised, language="en", entities=_ENTITIES)
        clean      = _resolve_overlaps(raw)
        result     = _anonymizer.anonymize(
                         text=normalised,
                         analyzer_results=clean,
                         operators=_OPERATORS,
                     )
        masked     = _repair_whitespace(result.text)
    return masked


# =========================================================
# TESTS
# =========================================================

if __name__ == "__main__":

    tests = [

        ("TEST 1 — Original bare values (no label)", """
        ECPPG0111K
        29ABCDE1234F1Z5
        HDFC0001234
        [Aadhaar Redacted]
        XGN3002623
        A2345671
        MH27 2012 0034761
        # UDYAM-DL-04-0012345
        9876543210
        user@example.com
        """),

        ("TEST 2 — With label prefix", """
        PAN no: ECPPG0111K
        GSTIN: 29ABCDE1234F1Z5
        IFSC HDFC0001234
        Aadhaar: [Aadhaar Redacted]
        Voter ID XGN3002623
        Passport A2345671
        DL MH27 2012 0034761
        # UDYAM-DL-04-0012345
        Mobile: 9876543210
        Email: user@example.com
        """),

        ("TEST 3 — Account / UAN context-anchored", """
        account no: 55678901234567
        UAN: 100234567890
        55678901234567
        100234567890
        """),

        ("TEST 4 — Mixed prose", """
        Hi, my Aadhaar is [Aadhaar Redacted] and my phone is 9 8 7 6 5 4 3 2 1 0.
        """),

        # ── NEW AMBIGUOUS-FORMAT TESTS ──────────────────────────────────

        ("TEST 5 — Space after every character", """
        PAN:  E C P P G 0 1 1 1 K
        IFSC: H D F C 0 0 0 1 2 3 4
        Voter ID: X G N 3 0 0 2 6 2 3
        Passport: A 2 3 4 5 6 7 1
        """),

        ("TEST 6 — Dash after every character", """
        PAN:  E-C-P-P-G-0-1-1-1-K
        GSTIN: 2-9-A-B-C-D-E-1-2-3-4-F-1-Z-5
        IFSC: H-D-F-C-0-0-0-1-2-3-4
        Aadhaar: [Aadhaar Redacted]
        Phone: 9-8-7-6-5-4-3-2-1-0
        """),

        ("TEST 7 — Random groupings / mixed separators", """
        PAN: ECPP G0 111K
        IFSC: HD FC00 01234
        Aadhaar: [Aadhaar Redacted]
        DL: MH-27 2012-003 4761
        # UDYAM: UDYAM DL 04 0012345
        account no: 5 5 6 7 8 9 0 1 2 3 4 5 6 7
        UAN: 1 0 0 2 3 4 5 6 7 8 9 0
        """),

        ("TEST 8 — Dot and underscore separators", """
        E.C.P.P.G.0.1.1.1.K
        IFSC: H_D_F_C_0_0_0_1_2_3_4
        Phone: 9.8.7.6.5.4.3.2.1.0
        Voter ID: X.G.N.3.0.0.2.6.2.3
        """),

        ("TEST 9 — In flowing prose with ambiguous IDs", """
        Dear team, the customer's PAN E C P P G 0 1 1 1 K was flagged.
        Their Aadhaar [Aadhaar Redacted] is on record. Please reach them at
        9 8 7 6 5 4 3 2 1 0 or user@example.com.
        Their driving licence MH-27-2012-0034761 expires next year.
        """),

    ]

    for label, text in tests:
        print("\n" + "=" * 60)
        print(label)
        print("=" * 60)
        print("INPUT:\n", text)
        print("OUTPUT:\n", mask_pii(text, monitor=False))