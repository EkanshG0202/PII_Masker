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
  UDYAM          MSME Udyam registration  (UDYAM-XX-DD-NNNNNNN)
  UDYOG          Udyog Aadhaar  (UAP19D0000001 / context-anchored 12-char)
  UAM            Udyog Aadhaar Memorandum  (same format as UDYOG; context-anchored)
  UAN            EPFO Universal Account Number (context-anchored, 12 digits)
  EMAIL          E-mail addresses

Ambiguous formats supported
───────────────────────────────────────────────────────────
  All entities tolerate arbitrary separators (spaces, dashes, dots,
  underscores, slashes) inserted between ANY characters, e.g.:
    E C P P G 0 1 1 1 K       ← space after every character
    E-C-P-P-G-0-1-1-1-K       ← dash after every character
    ECPP G0 111K               ← random groupings
    U D Y A M - D L - 0 4 - 0 0 1 2 3 4 5   ← fully separated UDYAM
    UAP 19D 000 0001           ← grouped Udyog Aadhaar
    U-A-P-1-9-D-0-0-0-0-0-0-1 ← dash-separated UAM

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
    token_re = re.compile(
        r'(\[\^?[^\]]*\]\{?\d*,?\d*\}?'
        r'|\[\^?[^\]]*\]'
        r'|\([^)]*\)\{?\d*,?\d*\}?'
        r'|\{?\d+,?\d*\}'
        r'|\\.'
        r'|[^^$.|?*+(){}\\]'
        r'|[.^$|?*+(){}\\]'
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
        "U D Y A M - D L - 0 4 - 0 0 1 2 3 4 5"  →  "UDYAM-DL-04-0012345"

    Algorithm:
      - Find maximal runs of (single-alnum)(single-separator) followed
        by a final alnum — at least 4 characters long.
      - Replace each run with its stripped version.
    """
    dense = re.compile(
        r'(?<![A-Za-z0-9])'
        r'([A-Za-z0-9]'
        r'(?:[^A-Za-z0-9\n][A-Za-z0-9]){3,})'
        r'(?![A-Za-z0-9])'
    )

    def _strip(m):
        return re.sub(r'[^A-Za-z0-9]', '', m.group(0))

    return dense.sub(_strip, text)


# =========================================================
# VALIDATORS
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


def _valid_udyam(text: str) -> bool:
    """UDYAM-XX-DD-NNNNNNN: prefix UDYAM + 2-alpha state + 2-digit district + 7-digit serial."""
    t = _norm(text).upper()
    return bool(re.fullmatch(r'UDYAM[A-Z]{2}\d{2}\d{7}', t))


# Indian state codes used in Udyog Aadhaar / UAM
_UA_STATE_CODES = {
    'AN','AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ','HP','HR',
    'JH','JK','KA','KL','LA','LD','MH','ML','MN','MP','MZ','NL','OD','OR',
    'PB','PY','RJ','SK','TG','TN','TR','TS','UP','UT','WB',
}

def _valid_udyog_uam(text: str) -> bool:
    """
    Udyog Aadhaar / UAM: UA + 2-char state code + 2-digit year + 1 alpha category
    + 7-digit serial = 14 chars total.
    Also accepts the compact 12-char variant (UA + state + 9 alphanum).
    """
    t = _norm(text).upper()
    # Full 14-char format: UA + state(2) + year(2) + category(1) + serial(7)
    m14 = re.fullmatch(r'UA([A-Z]{2})(\d{2})([A-Z])(\d{7})', t)
    if m14:
        return m14.group(1) in _UA_STATE_CODES
    # Compact 12-char: UA + state(2) + 8 alphanum (some older formats)
    m12 = re.fullmatch(r'UA([A-Z]{2})[A-Z0-9]{8}', t)
    if m12:
        return m12.group(1) in _UA_STATE_CODES
    return False


def _valid_uan(text: str) -> bool:
    digits = re.search(r'\d{12}', _norm(text))
    return bool(digits)


# =========================================================
# RECOGNIZERS (existing — unchanged)
# =========================================================

class CustomEmailRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="EMAIL_ADDRESS",
            patterns=[
                Pattern("email_standard",
                        r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
                        score=1.0),
                Pattern("email_missing_at",
                        r'(?i)\b[A-Z0-9._%+-]+(?:gmail|yahoo|outlook|hotmail|rediffmail)\.com\b',
                        score=0.85),
                Pattern("email_obfuscated_brackets",
                        r'(?i)\b[A-Z0-9._%+-]+\s*(?:\[at\]|\(at\))\s*[A-Z0-9.-]+\s*(?:\[dot\]|\(dot\)|\.)\s*(?:com|in|co\.in|org|net)\b',
                        score=0.80),
                Pattern("email_obfuscated_words",
                        r'(?i)\b[A-Z0-9._%+-]+\s+at\s+(?:gmail|yahoo|outlook|hotmail|rediffmail)\s+(?:dot|\.)\s+(?:com|in|co\.in|org|net)\b',
                        score=0.80),
                Pattern("email_space_missing_at",
                        r'(?i)\b[A-Z0-9._%+-]{3,}\s+(?:gmail|yahoo|outlook|hotmail|rediffmail)\.com\b',
                        score=0.80),
            ],
        )


class AadhaarRecognizer(PatternRecognizer):
    _G4  = r'\d' + _S + r'\d' + _S + r'\d' + _S + r'\d'
    _SEP = r'[\s.\-_()/]{0,3}'

    def __init__(self):
        g4 = self._G4
        sep = self._SEP
        super().__init__(
            supported_entity="AADHAAR",
            patterns=[
                Pattern("aadhaar_4_4_4",
                        r'(?<!\d)\d{4}' + sep + r'\d{4}' + sep + r'\d{4}(?!\d)',
                        score=0.85),
                Pattern("aadhaar_separated",
                        r'(?<![A-Za-z0-9])' + g4 + sep + g4 + sep + g4 + r'(?![A-Za-z0-9])',
                        score=0.80),
                Pattern("aadhaar_12_bare",
                        r'(?<!\d)\d{12}(?!\d)',
                        score=0.55),
            ],
            context=["aadhaar", "aadhar", "uid", "unique identification"],
        )

    def validate_result(self, pattern_text):
        return _valid_aadhaar(pattern_text)


class PANRecognizer(PatternRecognizer):
    _ALPHA = r'[A-Za-z]'
    _DIGIT = r'[0-9]'

    def __init__(self):
        a, d, s = self._ALPHA, self._DIGIT, _S
        pan_sep = (
            r'(?<![A-Za-z0-9])'
            + a + s + a + s + a + s + a + s + a
            + s
            + d + s + d + s + d + s + d
            + s
            + a
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
    _D = r'[0-9]'
    _A = r'[A-Za-z]'
    _AN = r'[A-Za-z0-9]'

    def __init__(self):
        d, a, an, s = self._D, self._A, self._AN, _S
        gst_sep = (
            r'(?<![A-Za-z0-9])'
            + d + s + d + s
            + a + s + a + s + a + s + a + s + a + s
            + d + s + d + s + d + s + d + s
            + a + s
            + an + s
            + r'[Zz]' + s
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
    _A = r'[A-Za-z]'
    _AN = r'[A-Za-z0-9]'

    def __init__(self):
        a, an, s = self._A, self._AN, _S
        ifsc_sep = (
            r'(?<![A-Za-z0-9])'
            + a + s + a + s + a + s + a
            + s + r'0' + s
            + an + s + an + s + an + s + an + s + an + s + an
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
    _CC  = r'(?:(?:\+|0{0,2})91[\s()\-]*)?'
    _S10 = (
        r'[6-9]' + _S
        + r'\d' + _S + r'\d' + _S + r'\d' + _S + r'\d'
        + _S
        + r'\d' + _S + r'\d' + _S + r'\d' + _S + r'\d' + _S + r'\d'
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
                Pattern("phone_sep10",
                        r'(?<![A-Za-z0-9])' + self._CC + self._S10 + r'(?![A-Za-z0-9])',
                        score=0.78),
            ],
        )

    def validate_result(self, pattern_text):
        return _valid_phone(pattern_text)


class VoterIdRecognizer(PatternRecognizer):
    _A = r'[A-Za-z]'
    _D = r'[0-9]'

    def __init__(self):
        a, d, s = self._A, self._D, _S
        voter_sep = (
            r'(?<![A-Za-z0-9])'
            + a + s + a + s + a
            + s
            + d + s + d + s + d + s + d + s + d + s + d + s + d
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
    _A = r'[A-Za-z]'
    _NZ = r'[1-9]'
    _D  = r'[0-9]'

    def __init__(self):
        a, nz, d, s = self._A, self._NZ, self._D, _S
        pp_sep = (
            r'(?<![A-Za-z0-9])'
            + a + s + nz + s
            + d + s + d + s + d + s + d + s + d
            + s + nz
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
            + a + s + a
            + s
            + d + s + d
            + s
            + r'(?:19|20)' + s + d + s + d
            + s
            + d + s + d + s + d + s + d + s + d + s + d + s + d
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


# =========================================================
# NEW — UDYAM RECOGNIZER
# Format: UDYAM-XX-DD-NNNNNNN
#   UDYAM  : literal prefix (5 chars)
#   XX     : 2-char state code (alpha)
#   DD     : 2-digit district number
#   NNNNNNN: 7-digit serial
# Total stripped: 5+2+2+7 = 16 alphanum chars
#
# Handles:
#   • Standard:      UDYAM-DL-04-0012345
#   • No dashes:     UDYAMDL040012345
#   • Space-sep:     U D Y A M D L 0 4 0 0 1 2 3 4 5
#   • Dash-sep:      U-D-Y-A-M-D-L-0-4-0-0-1-2-3-4-5
#   • Mixed/grouped: UDYAM DL 04 0012345 / UDYAM-DL 04-0012345
#   • Case-insensitive
# =========================================================

class UdyamRecognizer(PatternRecognizer):
    """
    MSME Udyam Registration Number: UDYAM-XX-DD-NNNNNNN
    Tolerates:
      - Any separator (space, dash, dot, underscore) between segments OR between every char
      - Case-insensitive
      - Bare (no separator), grouped, or fully separated
    """
    _A = r'[A-Za-z]'
    _D = r'[0-9]'

    def __init__(self):
        a, d, s = self._A, self._D, _S

        # Strict: UDYAM (with optional single separator between segments) - XX - DD - NNNNNNN
        # Allows separator between each segment block
        udyam_strict = (
            r'(?i)(?<!\w)'
            r'UDYAM'
            + r'[\s.\-_]{0,3}'   # sep after UDYAM
            + r'[A-Z]{2}'        # state code
            + r'[\s.\-_]{0,3}'   # sep
            + r'\d{2}'           # district
            + r'[\s.\-_]{0,3}'   # sep
            + r'\d{7}'           # serial
            + r'(?!\w)'
        )

        # Fully character-separated: U-D-Y-A-M-D-L-0-4-0-0-1-2-3-4-5
        # Each char of UDYAM followed by _S, then state (2), district (2), serial (7)
        udyam_sep = (
            r'(?i)(?<![A-Za-z0-9])'
            # U D Y A M  (each letter + optional sep)
            + r'[Uu]' + s + r'[Dd]' + s + r'[Yy]' + s + r'[Aa]' + s + r'[Mm]'
            + s
            # state code: 2 alpha
            + a + s + a
            + s
            # district: 2 digits
            + d + s + d
            + s
            # serial: 7 digits
            + d + s + d + s + d + s + d + s + d + s + d + s + d
            + r'(?![A-Za-z0-9])'
        )

        super().__init__(
            supported_entity="UDYAM",
            patterns=[
                Pattern("udyam_strict",    udyam_strict, score=0.99),
                Pattern("udyam_separated", udyam_sep,    score=0.92),
            ],
            context=["udyam", "msme", "registration", "udyam certificate",
                     "msme registration", "udyam number"],
        )

    def validate_result(self, pattern_text: str) -> bool:
        return _valid_udyam(pattern_text)


# =========================================================
# NEW — UDYOG / UAM RECOGNIZER
# Udyog Aadhaar (UA) and Udyog Aadhaar Memorandum share the same format:
#   UA + 2-char state code + 2-digit year + 1-char category + 7-digit serial
#   Example: UAP19D0000001  (UA + AP + 19 + D + 0000001)
#            UADL04A0034567
#
# Also accepts compact 12-char: UA + state(2) + 8 alphanum (older issuances)
#
# Context-anchored when the number alone is ambiguous; standalone UA* matched
# at high confidence when state code is valid.
#
# Handles:
#   • Standard:      UAP19D0000001
#   • Space-sep:     U A P 1 9 D 0 0 0 0 0 0 1
#   • Dash-sep:      U-A-P-1-9-D-0-0-0-0-0-0-1
#   • Grouped:       UA-P19-D000-0001 / UAP 19D 0000001
#   • Context-form:  Udyog Aadhaar: UAP19D0000001
#   • UAM prefix:    UAM No: UAP19D0000001  (keyword disambiguates)
# =========================================================

class UdyogUAMRecognizer(PatternRecognizer):
    """
    Udyog Aadhaar / UAM number — all three forms, separated variants included.
    """
    _A  = r'[A-Za-z]'
    _AN = r'[A-Za-z0-9]'
    _D  = r'[0-9]'

    _STATE_ALT = '|'.join(sorted(_UA_STATE_CODES, key=len, reverse=True))

    def __init__(self):
        a, an, d, s = self._A, self._AN, self._D, _S

        # ── Form B: UA + valid-state + YY + alpha-cat + 7-digit serial (14 chars) ──
        udyog_strict14 = (
            r'(?i)(?<![A-Za-z0-9])'
            r'UA'
            + r'(?:' + self._STATE_ALT + r')'
            + r'\d{2}'
            + r'[A-Z]'
            + r'\d{7}'
            + r'(?![A-Za-z0-9])'
        )

        # ── Form C: UA + valid-state + 8 alphanum compact (12 chars) ──
        udyog_strict12 = (
            r'(?i)(?<![A-Za-z0-9])'
            r'UA'
            + r'(?:' + self._STATE_ALT + r')'
            + r'[A-Z0-9]{8}'
            + r'(?![A-Za-z0-9])'
        )

        # ── Form B separated: U A <state> <yy> <cat> <7d> ──
        udyog_sep14 = (
            r'(?i)(?<![A-Za-z0-9])'
            + r'[Uu]' + s + r'[Aa]'
            + s + a + s + a          # state
            + s + d + s + d          # year
            + s + a                  # category
            + s + d + s + d + s + d + s + d + s + d + s + d + s + d  # 7 digits
            + r'(?![A-Za-z0-9])'
        )

        # ── Form C separated: U A <state> <8 alphanum> ──
        udyog_sep12 = (
            r'(?i)(?<![A-Za-z0-9])'
            + r'[Uu]' + s + r'[Aa]'
            + s + a + s + a
            + s + an + s + an + s + an + s + an + s + an + s + an + s + an + s + an
            + r'(?![A-Za-z0-9])'
        )

        # ── Form A (no UA prefix): SS YY X NNNNNNN ──
        # Strict 12-char: state(2) + year(2) + category(1 alpha) + serial(7 digits)
        # Valid state codes make the alternation selective enough for standalone use.
        udyog_noua_bare = (
            r'(?<![A-Za-z0-9])'
            + r'(?:' + self._STATE_ALT + r')'   # valid 2-char state
            + r'\d{2}'                           # 2-digit year
            + r'[A-Za-z]'                        # category letter
            + r'\d{7}'                           # 7-digit serial
            + r'(?![A-Za-z0-9])'
        )

        # Separated Form A: each char optionally separated
        udyog_noua_sep = (
            r'(?<![A-Za-z0-9])'
            + a + s + a              # state (2 alpha)
            + s + d + s + d          # year (2 digits)
            + s + a                  # category (1 alpha)
            + s + d + s + d + s + d + s + d + s + d + s + d + s + d  # 7 digits
            + r'(?![A-Za-z0-9])'
        )

        super().__init__(
            supported_entity="UDYOG_UAM",
            patterns=[
                # UA-prefix variants — high confidence standalone
                Pattern("udyog_strict14",    udyog_strict14, score=0.97),
                Pattern("udyog_strict12",    udyog_strict12, score=0.93),
                Pattern("udyog_sep14",       udyog_sep14,    score=0.90),
                Pattern("udyog_sep12",       udyog_sep12,    score=0.86),
                # No-UA-prefix variants — validator enforces state+year constraints
                Pattern("udyog_noua_bare",   udyog_noua_bare, score=0.88),
                Pattern("udyog_noua_sep",    udyog_noua_sep,  score=0.82),
            ],
            context=[
                "udyog", "udyog aadhaar", "uam", "udyog aadhaar memorandum",
                "ua number", "msme", "udyam", "ssme", "small enterprise",
                "micro enterprise", "medium enterprise",
                "uam no", "uam number", "uan number",
                "registration", "certificate",
            ],
        )

    def validate_result(self, pattern_text: str) -> bool:
        return _valid_udyog_uam(pattern_text)


# =========================================================
# CONTEXT-ANCHORED RECOGNIZERS (unchanged)
# =========================================================

class AccountNumberRecognizer(EntityRecognizer):
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
        UdyamRecognizer,
        UdyogUAMRecognizer,
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
    "UDYAM", "UDYOG_UAM",
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
    "UDYAM":          OperatorConfig("replace", {"new_value": "[UDYAM]"}),
    "UDYOG_UAM":      OperatorConfig("replace", {"new_value": "[UDYOG_UAM]"}),
    "UAN":            OperatorConfig("replace", {"new_value": "[UAN]"}),
}


# =========================================================
# RESOURCE MONITOR
# =========================================================

class ResourceMonitor:
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
            "label":        self.label,
            "wall_time_s":  round(wall_elapsed,  4),
            "cpu_user_s":   round(cpu_user,      4),
            "cpu_sys_s":    round(cpu_sys,       4),
            "mem_rss_mb":   round(mem_current  / 1024 / 1024, 2),
            "mem_delta_mb": round(mem_delta     / 1024 / 1024, 2),
            "threads":      threads,
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
      2. Presidio analyze on the normalised text.
      3. Overlap resolution.
      4. Presidio anonymize.
      5. Whitespace repair.
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

        # ── EXISTING TESTS (unchanged) ─────────────────────────────────

        ("TEST 1 — Original bare values (no label)", """
        ECPPG0111K
        29ABCDE1234F1Z5
        HDFC0001234
        2389 4539 1048
        XGN3002623
        A2345671
        MH27 2012 0034761
        9876543210
        user@example.com
        """),

        ("TEST 2 — With label prefix", """
        PAN no: ECPPG0111K
        GSTIN: 29ABCDE1234F1Z5
        IFSC HDFC0001234
        Aadhaar: 2389 4539 1048
        Voter ID XGN3002623
        Passport A2345671
        DL MH27 2012 0034761
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
        Hi, my Aadhaar is 2389 4539 1048 and my phone is 9 8 7 6 5 4 3 2 1 0.
        """),

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
        Aadhaar: 2-3-8-9-4-5-3-9-1-0-4-8
        Phone: 9-8-7-6-5-4-3-2-1-0
        """),

        ("TEST 7 — Random groupings / mixed separators", """
        PAN: ECPP G0 111K
        IFSC: HD FC00 01234
        Aadhaar: 2389 4539 1048
        DL: MH-27 2012-003 4761
        account no: 5 5 6 7 8 9 0 1 2 3 4 5 6 7
        UAN: 1 0 0 2 3 4 5 6 7 8 9 0
        """),

        ("TEST 8 — Dot and underscore separators", """
        E.C.P.P.G.0.1.1.1.K
        IFSC: H_D_F_C_0_0_0_1_2_3_4
        Phone: 9.8.7.6.5.4.3.2.1.0
        Voter ID: X.G.N.3.0.0.2.6.2.3
        """),

        # ── NEW UDYAM / UDYOG / UAM TESTS ─────────────────────────────

        ("TEST 9 — UDYAM standard formats", """
        Registration: UDYAM-DL-04-0012345
        My MSME number is UDYAM-MH-27-0098765
        udyam-ka-05-0001234
        UDYAM-UP-09-0034567
        """),

        ("TEST 10 — UDYAM bare (no dashes)", """
        UDYAMDL040012345
        UDYAMMH270098765
        """),

        ("TEST 11 — UDYAM space-separated (every character)", """
        U D Y A M D L 0 4 0 0 1 2 3 4 5
        U D Y A M M H 2 7 0 0 9 8 7 6 5
        """),

        ("TEST 12 — UDYAM dash-separated (every character)", """
        U-D-Y-A-M-D-L-0-4-0-0-1-2-3-4-5
        u-d-y-a-m-m-h-2-7-0-0-9-8-7-6-5
        """),

        ("TEST 13 — UDYAM random groupings / mixed separators", """
        UDYAM DL 04 0012345
        UDYAM-DL 04-0012345
        UDYAM.DL.04.0012345
        UDYAM_DL_04_0012345
        UDYAM/DL/04/0012345
        """),

        ("TEST 14 — UDYOG Aadhaar standard (14-char)", """
        Udyog Aadhaar: UAP19D0000001
        UA Number: UADL04A0034567
        UAMH27B0098765
        UAUP09C0001234
        """),

        ("TEST 15 — UDYOG compact legacy (12-char)", """
        Udyog Aadhaar: UAAP00000001
        UAM No: UADL00034567
        """),

        ("TEST 16 — UDYOG / UAM space-separated (every character)", """
        Udyog Aadhaar: U A P 1 9 D 0 0 0 0 0 0 1
        UAM: U A D L 0 4 A 0 0 3 4 5 6 7
        """),

        ("TEST 17 — UDYOG / UAM dash-separated (every character)", """
        U-A-P-1-9-D-0-0-0-0-0-0-1
        U-A-D-L-0-4-A-0-0-3-4-5-6-7
        """),

        ("TEST 18 — UDYOG / UAM mixed groupings", """
        UAP 19D 0000001
        UA-P19-D000-0001
        UA.DL.04A.0034567
        UA_MH_27B_0098765
        """),

        ("TEST 19 — All three in flowing prose", """
        Dear Sir, I am writing regarding my MSME enterprises.
        My Udyam registration is UDYAM-DL-04-0012345 and my Udyog Aadhaar
        number is UAP19D0000001. I also have an older UAM number UADL00034567.
        My PAN is ECPPG0111K and mobile is 9876543210.
        Please contact me at owner@mybusiness.in for any queries.
        """),

        ("TEST 20 — Obfuscated mix of all three", """
        Udyam: U-D-Y-A-M-D-L-0-4-0-0-1-2-3-4-5
        Udyog: U A P 1 9 D 0 0 0 0 0 0 1
        UAM: UA.DL.04A.0034567
        PAN: E C P P G 0 1 1 1 K
        Phone: 9-8-7-6-5-4-3-2-1-0
        """),

    ]

    for label, text in tests:
        print("\n" + "=" * 60)
        print(label)
        print("=" * 60)
        print("INPUT:\n", text)
        print("OUTPUT:\n", mask_pii(text, monitor=False))