"""
Indian Text PII Masker (GLiNER + Presidio)
==========================================
Extracts and masks unstructured text-based entities using GLiNER
(Generalist and Lightweight Indicator for NER) integrated into Microsoft Presidio.

Entities masked:
  PERSON         Person names (contextually deduced)
  ORGANIZATION   Company/org names (contextually deduced)
  ADDRESS        Postal addresses and landmarks (contextually deduced)
  LOCATION       Geographical locations, cities, and states

v2 changes
----------
- Raised GLiNER confidence threshold (0.60 → 0.70) to cut low-confidence
  false positives.
- Added post-analysis filter that drops any span whose matched text appears
  in the KNOWN_TERMS whitelist.
- Added _is_plausible_name() guard on PERSON hits.
- Presidio's built-in recognizers for PHONE_NUMBER and EMAIL_ADDRESS 
  are explicitly registered. (Note: PAN and Aadhaar masking have been disabled).

v3 changes
----------
- _in_whitelist() now also matches compound phrases whose *prefix* is a
  known gov-scheme term.
- Added _trim_role_suffix() to preserve correct name masking while stopping 
  the role word leaking into the [NAME] placeholder.
"""

import re
import time
import os
import psutil
from typing import List

from presidio_analyzer import AnalyzerEngine, RecognizerResult, EntityRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from gliner import GLiNER


# =========================================================
# WHITELISTS  — terms that must never be masked
# =========================================================

# Government scheme / portal / programme names
_GOV_SCHEMES = {
    "msme", "msme portal", "udyam", "udyam portal", "udyam registration",
    "udyog aadhaar", "udyog aadhar", "uam", "urc", "udyam assist",
    "udyam assist platform", "pm vishwakarma", "pm vishvkarma",
    "vishwakarma yojana", "vishvkarma yojna", "vishwakarma scheme",
    "startup india", "make in india", "jan dhan", "pmegp", "cgtmse",
    "nsic", "kvic", "sidbi", "gem portal", "government e-marketplace",
    "income tax portal", "gst portal", "mca portal", "epfo", "esic",
}

# Generic role/title/salutation words that are NOT person names
_GENERIC_ROLES = {
    "sir", "madam", "dear sir", "dear madam", "respected sir",
    "dear sir/madam", "proprietor", "sole proprietor", "applicant",
    "owner", "partner", "director", "manager", "officer", "authority",
    "gram pradhan", "pradhan", "sarpanch", "panchayat", "officer in charge",
    "registration authority", "udyog aadhaar registration authority",
}

# Common acronyms and technical terms that are not org or person names
_GENERIC_ACRONYMS = {
    "otp", "sms", "email", "email id", "mail id", "mobile number",
    "mobile no", "pan", "pan number", "pan no", "aadhaar", "aadhar",
    "uan", "urc", "gstin", "gst", "ifsc", "neft", "rtgs",
    "pdf", "otp number", "registration number", "application number",
    "certificate", "udyam certificate", "udyam number",
}

# Build a single flat set for O(1) lookup (lowercase)
KNOWN_TERMS: set = _GOV_SCHEMES | _GENERIC_ROLES | _GENERIC_ACRONYMS


def _in_whitelist(text: str) -> bool:
    """
    Return True if the span text (case-insensitive) should never be masked.
    """
    normalised = text.strip().lower()

    # Tier 1 — exact
    if normalised in KNOWN_TERMS:
        return True

    # Tier 2 — exact after punctuation strip
    cleaned = re.sub(r"[^\w\s]", "", normalised).strip()
    if cleaned in KNOWN_TERMS:
        return True

    # Tier 3 — prefix match
    hyphen_normalised = re.sub(r"[-]", " ", normalised).strip()
    for term in KNOWN_TERMS:
        if (normalised.startswith(term + " ")
                or cleaned.startswith(term + " ")
                or hyphen_normalised.startswith(term + " ")):
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
    "application", "applicant",
}


def _is_plausible_name(text: str) -> bool:
    """
    Returns False for spans that cannot reasonably be a person name.
    """
    stripped = text.strip()

    if re.fullmatch(r"\d+", stripped):
        return False

    if re.search(r"\d", stripped) and len(stripped) <= 15:
        return False

    tokens = stripped.lower().split()

    if len(tokens) == 1 and tokens[0] in _COMMON_NON_NAMES:
        return False

    _ALLOWED_LOWERCASE_PARTICLES = {"ji", "kumar", "devi", "bai", "lal", "ram"}
    if len(tokens) == 1 and stripped == stripped.lower():
        if tokens[0] not in _ALLOWED_LOWERCASE_PARTICLES:
            return False

    return True


# =========================================================
# ROLE-SUFFIX TRIMMER
# =========================================================

_ROLE_SUFFIX_RE = re.compile(
    r"""
    [,\s]*          
    \b(
        proprietor | sole\s+proprietor |
        owner       | co[- ]?owner      |
        partner     | managing\s+partner |
        director    | managing\s+director | executive\s+director |
        manager     | general\s+manager  |
        applicant   | complainant        |
        founder     | co[- ]?founder     |
        trustee     | secretary          |
        chairman    | chairperson
    )\b
    [,\s]*$         
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _trim_role_suffix(span_text: str, start: int, end: int):
    """
    If the span ends with a role word, shrink the span to exclude it.
    """
    m = _ROLE_SUFFIX_RE.search(span_text)
    if not m:
        return span_text, start, end

    trimmed = span_text[: m.start()].rstrip(" ,")
    if not trimmed:
        return "", start, start

    new_end = start + len(trimmed)
    return trimmed, start, new_end


class GlinerRecognizer(EntityRecognizer):
    def __init__(
        self,
        model_name: str = "urchade/gliner_medium-v2.1",
        threshold: float = 0.70,      
    ):
        self.label_mapping = {
            "person":       "PERSON",
            "organization": "ORGANIZATION",
            "address":      "ADDRESS",
            "location":     "LOCATION",
        }
        self.gliner_labels = list(self.label_mapping.keys())
        self.threshold = threshold

        print(f"[text-masking] Initialising GLiNER model: {model_name}...")
        self.model = GLiNER.from_pretrained(model_name)
        print("[text-masking] GLiNER model loaded and ready.")

        super().__init__(
            supported_entities=list(self.label_mapping.values()),
            name="GlinerRecognizer",
        )

    def load(self):
        pass  

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts=None,
    ) -> List[RecognizerResult]:
        results = []

        predictions = self.model.predict_entities(
            text, self.gliner_labels, threshold=self.threshold
        )

        for pred in predictions:
            gliner_label = pred["label"]
            presidio_entity = self.label_mapping.get(gliner_label)

            if not presidio_entity or presidio_entity not in entities:
                continue

            span_text = text[pred["start"]: pred["end"]]

            if _in_whitelist(span_text):
                continue

            if presidio_entity == "PERSON":
                span_text, pred_start, pred_end = _trim_role_suffix(
                    span_text, pred["start"], pred["end"]
                )
                if not span_text:
                    continue
                if not _is_plausible_name(span_text):
                    continue

            if presidio_entity == "ORGANIZATION":
                norm = span_text.strip().lower()
                if norm in _GOV_SCHEMES or re.sub(r"[^\w\s]", "", norm) in _GOV_SCHEMES:
                    continue

            final_start = pred_start if presidio_entity == "PERSON" else pred["start"]
            final_end   = pred_end   if presidio_entity == "PERSON" else pred["end"]

            results.append(
                RecognizerResult(
                    entity_type=presidio_entity,
                    start=final_start,
                    end=final_end,
                    score=pred["score"],
                )
            )

        return results


# =========================================================
# ENGINE SETUP
# =========================================================

def _build_analyzer() -> AnalyzerEngine:
    engine = AnalyzerEngine()

    gliner_recognizer = GlinerRecognizer(
        model_name="urchade/gliner_medium-v2.1",
        threshold=0.70,
    )
    engine.registry.add_recognizer(gliner_recognizer)

    return engine


_analyzer = _build_analyzer()
_anonymizer = AnonymizerEngine()

_GLINER_ENTITIES = ["PERSON", "ORGANIZATION", "ADDRESS", "LOCATION"]

# Removed IN_PAN and IN_AADHAAR to blacklist them from being masked
_STRUCTURED_ENTITIES = [
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
]

_ALL_ENTITIES = _GLINER_ENTITIES + _STRUCTURED_ENTITIES

# Removed IN_PAN and IN_AADHAAR replacement operators
_OPERATORS = {
    "PERSON":        OperatorConfig("replace", {"new_value": "[NAME]"}),
    "ORGANIZATION":  OperatorConfig("replace", {"new_value": "[ORG]"}),
    "ADDRESS":       OperatorConfig("replace", {"new_value": "[ADDRESS]"}),
    "LOCATION":      OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    "PHONE_NUMBER":  OperatorConfig("replace", {"new_value": "[PHONE]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
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
        self._cpu_start = self._proc.cpu_times()
        self._mem_start = self._proc.memory_info().rss
        return self

    def __exit__(self, *_):
        wall_end = time.perf_counter()
        cpu_end = self._proc.cpu_times()
        mem_current = self._proc.memory_info().rss

        self.stats = {
            "label":        self.label,
            "wall_time_s":  round(wall_end - self._wall_start, 4),
            "cpu_user_s":   round(cpu_end.user - self._cpu_start.user, 4),
            "mem_delta_mb": round((mem_current - self._mem_start) / 1024 / 1024, 2),
        }

        if self.print_report:
            print(
                f"\nExecution stats — {self.stats['label']}\n"
                f"  Wall time: {self.stats['wall_time_s']:.4f} s | "
                f"  CPU user : {self.stats['cpu_user_s']:.4f} s | "
                f"  RAM delta: {self.stats['mem_delta_mb']:+.2f} MB\n"
            )


def _repair_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(\w)\[", r"\1 [", text)
    text = re.sub(r"]\s+([,.:;!?])", r"]\1", text)
    return text


# =========================================================
# PUBLIC INTERFACE API
# =========================================================

def mask_text_entities(text: str, monitor: bool = False) -> str:
    """
    Scans and redacts Names, Orgs, Locations, Addresses, phone numbers,
    and email addresses. PAN and Aadhaar masking have been explicitly disabled.
    """
    with ResourceMonitor("mask_text_entities", print_report=monitor):
        raw_hits = _analyzer.analyze(
            text=text,
            language="en",
            entities=_ALL_ENTITIES,
        )

        redacted = _anonymizer.anonymize(
            text=text,
            analyzer_results=raw_hits,
            operators=_OPERATORS,
        )

        output = _repair_whitespace(redacted.text)

    return output