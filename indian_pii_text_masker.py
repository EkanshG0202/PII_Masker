"""
Indian Text PII Masker (GLiNER + Presidio)
==========================================
Extracts and masks unstructured text-based entities using GLiNER
(Generalist and Lightweight Indicator for NER) integrated into Microsoft Presidio.

Entities masked:
  PERSON         Person names (contextually deduced by GLiNER)
  ORGANIZATION   Company/org names (contextually deduced by GLiNER)
  ADDRESS        Postal addresses and landmarks (GLiNER)
  LOCATION       Geographical locations, cities, states (GLiNER)

Note: Phone, email, bank account, IFSC, GST, PAN, and Aadhaar masking are
handled by separate dedicated maskers.
"""

import re
import os
import time
import uuid
import psutil
from typing import List, Dict, Tuple, Optional

import torch
from presidio_analyzer import AnalyzerEngine, RecognizerResult, EntityRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from gliner import GLiNER


# =========================================================
# SENTINEL HELPER
# =========================================================

def _sentinel(tag: str) -> str:
    """Generate a unique placeholder that won't appear in real text."""
    return f"__PII_{tag}_{uuid.uuid4().hex[:10]}__"


# =========================================================
# WHITELISTS — terms that must NEVER be masked
# =========================================================

_GOV_SCHEMES = {
    "msme", "msme portal", "udyam", "udyam portal", "udyam registration",
    "udyog aadhaar", "udyog aadhar", "uam", "urc",
    # Udyam Aadhar / Aadhaar variants (the specific phrase that was mis-tagged)
    "udyam aadhar", "udyam aadhaar",
    "udyam aadhar portal", "udyam aadhaar portal",
    "udaym aadhar", "udaym aadhaar",
    "udhyam aadhar", "udhyam aadhaar",
    "udyam assist", "udyam assist platform",
    "pm vishwakarma", "pm vishvkarma",
    "vishwakarma", "vishvkarma",
    "vishwakarma yojana", "vishvkarma yojna", "vishvkarma yojana",
    "vishwakarma scheme",
    "pm vishwakarma yojana", "pm vishvkarma yojna",
    "pradhan mantri kisan sampada yojana",
    "startup india", "make in india", "jan dhan", "pmegp", "cgtmse",
    "nsic", "kvic", "sidbi", "gem portal", "government e-marketplace",
    "income tax portal", "gst portal", "mca portal", "epfo", "esic",
    "district industries centre",
    "the udyog aadhaar registration authority",
    "udyog aadhaar registration authority",
    "ax-momsme",
}

_GENERIC_ROLES = {
    "sir", "madam", "mam", "dear sir", "dear madam", "respected sir",
    "dear sir/madam", "dear sir / madam",
    "proprietor", "sole proprietor", "applicant",
    "owner", "partner", "director", "manager", "officer", "authority",
    "gram pradhan", "pradhan", "sarpanch", "panchayat", "officer in charge",
    "registration authority",
    "karta", "huf", "authorized signatory", "authorised signatory",
    "nodal officer",
    # Document / letter terms mistaken for names
    "letter head", "letterhead",
    # Relational / generic nouns GLiNER grabs as PERSON
    "spouse", "my spouse", "wife", "husband",
    "persan", "persan name",           # OCR/typo variant seen in data
    "tehsil",                          # administrative label, not a name
}

_GENERIC_ACRONYMS = {
    "otp", "sms", "email", "email id", "mail id", "mobile number",
    "mobile no", "pan", "pan number", "pan no", "aadhaar", "aadhar",
    "uan", "urc", "gstin", "gst", "ifsc", "neft", "rtgs",
    "pdf", "otp number", "registration number", "application number",
    "certificate", "udyam certificate", "udyam number",
    "llp", "pvt", "ltd", "pvt ltd", "private limited",
    # Fiscal-year label variants
    "f.y.", "f.y", "fy",
    # Common abbreviations misread as location/name
    "d.o.b", "d.o.b.", "dob",
}

KNOWN_TERMS: set = _GOV_SCHEMES | _GENERIC_ROLES | _GENERIC_ACRONYMS

# Single-token terms that GLiNER frequently misclassifies
_SINGLE_TOKEN_BLACKLIST = {
    # Words mistaken for PERSON names
    "udyam", "udhyam", "udyog", "registration", "cancel", "cancell",
    "solve", "mam", "latitude", "longitude", "i", "already", "forgot",
    "alrady",        # misspelling of "already" — seen tagged as LOCATION
    "tehsil",        # administrative label
    "ho",            # fragment ("ho to") tagged as NAME
    # Words mistaken for LOCATION
    "us", "f.y.", "fy", "f.y",
    "alrady",        # also tagged as LOCATION in one case
    # Words mistaken for ORG
    "cement business", "d.o.b", "dob", "llp",
    "udyog adhar", "udyog aadhar",   # portal names, not organisations
}


def _in_whitelist(text: str) -> bool:
    norm = text.strip().lower()
    if norm in KNOWN_TERMS or norm in _SINGLE_TOKEN_BLACKLIST:
        return True
    cleaned = re.sub(r"[^\w\s]", "", norm).strip()
    if cleaned in KNOWN_TERMS or cleaned in _SINGLE_TOKEN_BLACKLIST:
        return True
    hyphenless = re.sub(r"[-]", " ", norm).strip()
    for term in KNOWN_TERMS:
        if (norm.startswith(term + " ")
                or norm == term
                or cleaned.startswith(term + " ")
                or cleaned == term
                or hyphenless.startswith(term + " ")):
            return True
    # Also reject if the span IS a known term with trailing punctuation stripped
    punct_stripped = norm.rstrip(".,;:!? ")
    if punct_stripped in KNOWN_TERMS or punct_stripped in _SINGLE_TOKEN_BLACKLIST:
        return True
    return False


# =========================================================
# PERSON NAME PLAUSIBILITY GUARD
# =========================================================

_COMMON_NON_NAMES = {
    "dear", "hello", "hi", "kindly", "please", "thank", "thanks",
    "regards", "warm", "best", "yours", "truly", "sincerely",
    "note", "subject", "reference", "re", "sub", "enclosure",
    "attach", "attachment", "enclosed", "copy", "date", "place",
    "from", "to", "cc", "bcc",
    "bank", "loan", "business", "trade", "company", "firm", "enterprise",
    "certificate", "registration", "scheme", "portal", "platform",
    "application", "applicant", "account", "amount", "rupees", "inr",
    "lakh", "crore", "total",
    # Pronouns / common words frequently mis-tagged
    "udyam", "udhyam", "udyog", "i", "we", "our", "my",
    "cancel", "cancell", "solve", "already", "forgot", "mam",
    "alrady",        # misspelling of "already"
    "latitude", "longitude", "email", "otp",
    # Document / structural labels
    "letter head", "letterhead",
    "tehsil",
    # Relational nouns falsely tagged as PERSON
    "spouse", "wife", "husband",
    "persan",        # OCR/typo for "person"
    # Short imperative verbs / phrases seen in data
    "ho",
}

# Multi-word spans that should never be masked as PERSON (checked after tokenising)
_NON_NAME_PHRASES = {
    "solve my problem",
    "letter head",
    "udyam aadhar",
    "udyam aadhaar",
    "udyam registration",
    "udyog aadhar",
    "udyog aadhaar",
    "my spouse",
    "solve my",
    "d.o.b.",
    "d.o.b",
    "persan name",
}

_ALLOWED_LOWERCASE_PARTICLES = {"ji", "kumar", "devi", "bai", "lal", "ram"}


def _is_plausible_name(text: str) -> bool:
    stripped = text.strip()
    norm = stripped.lower()

    # Explicit multi-word non-name phrases
    if norm in _NON_NAME_PHRASES:
        return False

    # Pure digits / only whitespace
    if re.fullmatch(r"[\d\s]+", stripped):
        return False
    # Short alphanumeric (codes, IDs)
    if re.search(r"\d", stripped) and len(stripped) <= 20:
        return False
    tokens = norm.split()
    # Single common non-name word
    if len(tokens) == 1 and tokens[0] in _COMMON_NON_NAMES:
        return False
    # Single all-lowercase word that is not a known Indian particle
    if len(tokens) == 1 and stripped == stripped.lower():
        if tokens[0] not in _ALLOWED_LOWERCASE_PARTICLES:
            return False
    # Single character
    if len(stripped.strip()) <= 1:
        return False
    return True


# =========================================================
# ROLE-SUFFIX TRIMMER
# =========================================================

_ROLE_SUFFIX_RE = re.compile(
    r"""
    [,\s]* \b(
        proprietor  | sole\s+proprietor |
        owner       | co[- ]?owner      |
        partner     | managing\s+partner |
        director    | managing\s+director | executive\s+director |
        manager     | general\s+manager  |
        applicant   | complainant        |
        founder     | co[- ]?founder     |
        trustee     | secretary          |
        chairman    | chairperson        |
        karta       | authorised\s+signatory | authorized\s+signatory
    )\b
    [,\s]*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _trim_role_suffix(span_text: str, start: int, end: int):
    m = _ROLE_SUFFIX_RE.search(span_text)
    if not m:
        return span_text, start, end
    trimmed = span_text[: m.start()].rstrip(" ,")
    if not trimmed:
        return "", start, start
    return trimmed, start, start + len(trimmed)


# =========================================================
# REGEX PRE-PASS — protect structured tokens before GLiNER
# =========================================================
# These tokens are restored to their original value after GLiNER runs.
# Nothing here is masked — masking of phone/email/bank/IFSC/GST/PAN/Aadhaar
# is the responsibility of other maskers.

_PROTECT_PATTERNS: List[Tuple[str, int]] = [
    # UDYAM / UAM registration numbers
    (
        r"\b(?:UDYAM|UDAYM|UDHYAM)[- ][A-Z]{2}[- ]\d{2}[- ]\d{7}"
        r"(?:/[A-Z]/\d{5})?",
        re.IGNORECASE,
    ),
    (
        r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-CE-Z]\d{7}(?!\w)",
        re.IGNORECASE,
    ),
    # Dates  dd/mm/yyyy or dd-mm-yyyy
    (
        r"\b(0?[1-9]|[12]\d|3[01])[/\-](0?[1-9]|1[0-2])[/\-](19|20)\d{2}\b",
        0,
    ),
    # IFSC codes
    (r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE),
    # GST numbers
    (r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b", re.IGNORECASE),
    # Full emails  — must come before phone so digit-heavy local-parts are shielded first
    (r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b", 0),
    # Partial emails with embedded digit run (e.g. pk8804398358gmail.com)
    # Must also run before the phone pattern for the same reason.
    (
        r"\b[a-zA-Z][a-zA-Z0-9._%+\-]*\d+[a-zA-Z0-9._%+\-]*"
        r"(?:gmail|yahoo|hotmail|outlook|rediffmail|ymail)(?:\.com)?\b",
        re.IGNORECASE,
    ),
    # Plain partial emails without digits (e.g. abcgmail.com)
    (
        r"\b[a-zA-Z0-9._%+\-]{4,}(?:gmail|yahoo|hotmail|outlook|rediffmail|ymail)(?:\.com)?\b",
        re.IGNORECASE,
    ),
    # Phone numbers  (+91/91/0 prefix optional, 10 digits starting 6-9)
    # Runs AFTER all email patterns so embedded digit runs in email prefixes
    # have already been replaced by sentinels.
    (
        r"(?<!\d)(?:\+91[\s\-]?|91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)",
        0,
    ),
    # Bank account numbers (9–18 digits)
    (r"(?<![_\d])\d{9,18}(?![_\d])", 0),
]


def _regex_protect(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace structured tokens with sentinels so GLiNER never sees them.
    All sentinels are restored verbatim after GLiNER runs.

    Returns:
        protected_text — text with sentinels substituted
        restore_map    — {sentinel: original_text}
    """
    restore_map: Dict[str, str] = {}

    for pattern, flags in _PROTECT_PATTERNS:
        def _repl(m, _pattern=pattern):
            s = _sentinel("PROTECT")
            restore_map[s] = m.group(0)
            return s
        text = re.sub(pattern, _repl, text, flags=flags)

    return text, restore_map


# =========================================================
# GLINER RECOGNIZER
# =========================================================

class GlinerRecognizer(EntityRecognizer):
    def __init__(self, model_name: str = "urchade/gliner_medium-v2.1", threshold: float = 0.82):
        self.label_mapping = {
            "person name":    "PERSON",
            "company name":   "ORGANIZATION",
            "postal address": "ADDRESS",
            "location":       "LOCATION",
        }
        self.gliner_labels = list(self.label_mapping.keys())
        self.threshold = threshold

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"[text-masking] Loading GLiNER on {device.upper()}...")
        self.model = GLiNER.from_pretrained(model_name).to(device)
        print("[text-masking] GLiNER model loaded and ready.")

        super().__init__(
            supported_entities=list(self.label_mapping.values()),
            name="GlinerRecognizer",
        )

    def load(self):
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []

        predictions = self.model.predict_entities(text, self.gliner_labels, threshold=self.threshold)

        for pred in predictions:
            presidio_entity = self.label_mapping.get(pred["label"])
            if not presidio_entity or presidio_entity not in entities:
                continue

            span_text = text[pred["start"]: pred["end"]]

            # ── Global whitelist ──────────────────────────────────────────────
            if _in_whitelist(span_text):
                continue

            # ── Skip sentinel tokens (protected structured data) ─────────────
            if re.match(r"^__PII_", span_text.strip()):
                continue

            # ── Entity-specific guards ────────────────────────────────────────
            pred_start, pred_end = pred["start"], pred["end"]

            if presidio_entity == "PERSON":
                span_text, pred_start, pred_end = _trim_role_suffix(
                    span_text, pred_start, pred_end
                )
                if not span_text:
                    continue
                if not _is_plausible_name(span_text):
                    continue

            elif presidio_entity == "ORGANIZATION":
                norm = span_text.strip().lower()
                cleaned_norm = re.sub(r"[^\w\s]", "", norm)
                if norm in _GOV_SCHEMES or cleaned_norm in _GOV_SCHEMES:
                    continue
                if norm in {
                    "i", "we", "my", "d.o.b", "dob", "llp",
                    "district industries centre",
                    "udyog adhar", "udyog aadhar", "udyog aadhaar",
                }:
                    continue
                # Reject single-word spans that are in the blacklist
                if norm.strip() in _SINGLE_TOKEN_BLACKLIST:
                    continue

            elif presidio_entity == "LOCATION":
                norm = span_text.strip().lower()
                if norm in {
                    "us", "f.y.", "fy", "f.y", "i", "we",
                    "alrady",   # misspelling of "already"
                }:
                    continue
                if re.match(r"^f\.?y\.?\s*\d{4}", norm):
                    continue
                # Reject single-word spans that are in the blacklist
                if norm.strip() in _SINGLE_TOKEN_BLACKLIST:
                    continue

            results.append(
                RecognizerResult(
                    entity_type=presidio_entity,
                    start=pred_start,
                    end=pred_end,
                    score=pred["score"],
                )
            )

        return results


# =========================================================
# ENGINE SETUP
# =========================================================

def _build_analyzer() -> AnalyzerEngine:
    engine = AnalyzerEngine()
    engine.registry.add_recognizer(GlinerRecognizer(
        model_name="urchade/gliner_medium-v2.1",
        threshold=0.82,
    ))
    return engine


_analyzer   = _build_analyzer()
_anonymizer = AnonymizerEngine()

_ENTITIES = ["PERSON", "ORGANIZATION", "ADDRESS", "LOCATION"]

_OPERATORS = {
    "PERSON":       OperatorConfig("replace", {"new_value": "[NAME]"}),
    "ORGANIZATION": OperatorConfig("replace", {"new_value": "[ORG]"}),
    "ADDRESS":      OperatorConfig("replace", {"new_value": "[ADDRESS]"}),
    "LOCATION":     OperatorConfig("replace", {"new_value": "[LOCATION]"}),
}


# =========================================================
# UTILITIES
# =========================================================

class ResourceMonitor:
    def __init__(self, label: str = "block", print_report: bool = True):
        self.label = label
        self.print_report = print_report
        self._proc = psutil.Process(os.getpid())

    def __enter__(self):
        self._wall_start = time.perf_counter()
        self._cpu_start  = self._proc.cpu_times()
        self._mem_start  = self._proc.memory_info().rss
        return self

    def __exit__(self, *_):
        wall_end = time.perf_counter()
        cpu_end  = self._proc.cpu_times()
        mem_cur  = self._proc.memory_info().rss
        self.stats = {
            "label":        self.label,
            "wall_time_s":  round(wall_end - self._wall_start, 4),
            "cpu_user_s":   round(cpu_end.user - self._cpu_start.user, 4),
            "mem_delta_mb": round((mem_cur - self._mem_start) / 1024 / 1024, 2),
        }
        if self.print_report:
            print(
                f"\nExecution stats — {self.stats['label']}\n"
                f"  Wall time : {self.stats['wall_time_s']:.4f} s\n"
                f"  CPU user  : {self.stats['cpu_user_s']:.4f} s\n"
                f"  RAM delta : {self.stats['mem_delta_mb']:+.2f} MB\n"
            )


def _repair_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(\w)\[", r"\1 [", text)
    text = re.sub(r"]\s+([,.:;!?])", r"]\1", text)
    return text.strip()


# =========================================================
# PUBLIC API
# =========================================================

def mask_text_entities(text: str, monitor: bool = False) -> str:
    """
    Scans and redacts PERSON, ORGANIZATION, ADDRESS, and LOCATION entities
    using GLiNER via Presidio.

    Structured tokens (phone, email, bank account, IFSC, GST, UDYAM, dates)
    are temporarily shielded with sentinels before GLiNER runs and restored
    verbatim afterwards — ensuring this function does not double-mask or
    corrupt tokens that are the responsibility of other maskers.
    """
    with ResourceMonitor("mask_text_entities", print_report=monitor):

        # ── Step 1: Protect structured tokens from GLiNER ──────────────────
        protected_text, restore_map = _regex_protect(text)

        # ── Step 2: GLiNER + Presidio pass ─────────────────────────────────
        raw_hits = _analyzer.analyze(
            text=protected_text,
            language="en",
            entities=_ENTITIES,
        )
        redacted = _anonymizer.anonymize(
            text=protected_text,
            analyzer_results=raw_hits,
            operators=_OPERATORS,
        )
        output = redacted.text

        # ── Step 3: Restore all protected tokens verbatim ───────────────────
        for sentinel, original in restore_map.items():
            output = output.replace(sentinel, original)

        output = _repair_whitespace(output)

    return output