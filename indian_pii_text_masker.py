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
    "udyam number", "udyam no",
    "udhyam number", "udhyam no",
    "udyam registration number", "udyam registration no",
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
    "sir/madam", "sir / madam", "sir/mam", "sir / mam",
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
    "unable",        # "unable to" fragment
    "writing",       # "writing to request" fragment
    "trying",        # "trying to register" fragment
    "clicking",      # "clicking on" fragment
    "applying",      # "applying to" fragment
    "email",         # "Email ID" - email should never be a name
    "otp",           # "OTP" sometimes mis-tagged as NAME
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
    "udyam", "udhyam", "udyog", "udaym",
    "i", "we", "our", "my",
    "cancel", "cancell", "cancellation", "solve", "already", "forgot", "mam",
    "alrady",        # misspelling of "already"
    "latitude", "longitude", "email", "otp",
    # Aadhaar label word (not the number — that is protected by sentinel)
    "aadhaar", "aadhar", "adhar", "addhar",
    # Document / structural labels
    "letter head", "letterhead",
    "tehsil", "district", "nagar", "village", "taluka",
    # Relational nouns falsely tagged as PERSON
    "spouse", "wife", "husband",
    "persan",        # OCR/typo for "person"
    # Short imperative verbs / phrases seen in data
    "ho",
    # Commonly mis-tagged action words
    "update", "change", "reset", "verify", "register",
    "download", "upload", "migrate", "retrieve",
    # Verb-phrase fragments that appear after "I am"
    "unable", "writing", "trying", "applying", "clicking", "attempting",
    "running", "still",
}

# Multi-word spans that should never be masked as PERSON.
# Matched via regex word-boundary search (see _is_non_name_phrase) so
# minor spacing or punctuation differences don't cause misses.
_NON_NAME_PHRASES = {
    "solve my problem",
    "solve my",
    "letter head",
    "udyam aadhar",
    "udyam aadhaar",
    "udyam registration",
    "udhyam registration",
    "udaym registration",
    "udyog aadhar",
    "udyog aadhaar",
    "udyog adhar",
    "my spouse",
    "d.o.b.",
    "d.o.b",
    "persan name",
    # Imperative verb phrases seen mis-tagged as names
    "cancel my",
    "cancell my",
    "please cancel",
    "please cancell",
    "please solve",
    "please update",
    "please change",
    "please register",
    "please verify",
    "please reset",
    "please retrieve",
    "please download",
    "please upload",
    "please migrate",
    # Verb-phrase fragments falsely captured by "I am ..." regex pattern
    "unable to",
    "not able to",
    "trying to",
    "writing to",
    "writing to request",
    "applying to",
    "attempting to",
    "clicking on",
    "continuously trying",
    "still not",
    # Action words that appear mid-sentence
    "cancell",
    "cancel",
}

# Pre-compiled regex for phrase matching — word-boundary anchored so
# "udyam registration" inside a longer sentence is still caught.
_NON_NAME_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in sorted(_NON_NAME_PHRASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _is_non_name_phrase(text: str) -> bool:
    """Return True if the entire span matches a known non-name phrase."""
    norm = text.strip().lower()
    # Exact set membership first (fast path)
    if norm in _NON_NAME_PHRASES:
        return True
    # Regex match that covers the whole span
    m = _NON_NAME_PHRASE_RE.fullmatch(norm)
    return m is not None

_ALLOWED_LOWERCASE_PARTICLES = {"ji", "kumar", "devi", "bai", "lal", "ram"}

# Known Indian cities, districts, and state names that GLiNER sometimes tags
# as PERSON. Checked case-insensitively. Not exhaustive — covers the most
# frequent offenders seen in grievance data.
_KNOWN_PLACE_NAMES = {
    # States / UTs
    "maharashtra", "gujarat", "rajasthan", "karnataka", "kerala",
    "tamilnadu", "tamil nadu", "andhra pradesh", "telangana", "odisha",
    "west bengal", "uttar pradesh", "madhya pradesh", "bihar", "jharkhand",
    "chhattisgarh", "uttarakhand", "himachal pradesh", "punjab", "haryana",
    "delhi", "goa", "assam", "manipur", "meghalaya", "tripura", "nagaland",
    "mizoram", "arunachal pradesh", "sikkim", "jammu", "kashmir",
    "chandigarh", "puducherry", "pondicherry",
    # Major cities / district HQs that appear frequently in data
    "mumbai", "pune", "nagpur", "nashik", "aurangabad",
    "ahmedabad", "surat", "vadodara", "rajkot",
    "jaipur", "jodhpur", "udaipur", "bikaner", "kota",
    "bangalore", "bengaluru", "mysuru", "mysore", "hubli", "mangalore",
    "hyderabad", "secunderabad", "warangal", "visakhapatnam", "vizag",
    "ernakulam", "kochi", "cochin", "kozhikode", "calicut", "thiruvananthapuram",
    "thrissur", "palakkad", "kollam", "kannur", "malappuram", "alappuzha",
    "chennai", "madurai", "coimbatore", "salem", "trichy", "tiruchirappalli",
    "kolkata", "howrah", "siliguri", "durgapur", "asansol",
    "lucknow", "kanpur", "agra", "varanasi", "allahabad", "prayagraj",
    "patna", "gaya", "bhagalpur",
    "bhopal", "indore", "gwalior", "jabalpur",
    "bhubaneswar", "cuttack", "rourkela",
    "ranchi", "dhanbad", "jamshedpur",
    "raipur", "bilaspur",
    "dehradun", "haridwar", "roorkee",
    "chandigarh", "ludhiana", "amritsar", "jalandhar",
    "gurgaon", "gurugram", "faridabad", "ambala",
    "noida", "ghaziabad", "meerut", "bareilly", "moradabad",
    "srinagar", "jammu", "leh",
    "udhampur", "belgachia", "bikaner",
    "kishanganj",
}


def _is_plausible_name(text: str) -> bool:
    stripped = text.strip()
    norm = stripped.lower()

    # ── Reject if the whole span is a known non-name phrase ──────────────────
    if _is_non_name_phrase(norm):
        return False

    # ── Reject if the span (or its cleaned form) is in any whitelist ─────────
    # This catches "Udyam", "Aadhaar", "OTP", etc. regardless of casing.
    cleaned = re.sub(r"[^\w\s]", "", norm).strip()
    if (norm in KNOWN_TERMS or cleaned in KNOWN_TERMS
            or norm in _SINGLE_TOKEN_BLACKLIST or cleaned in _SINGLE_TOKEN_BLACKLIST):
        return False
    # Also check with trailing punctuation stripped
    punct_stripped = norm.rstrip(".,;:!? ")
    if punct_stripped in KNOWN_TERMS or punct_stripped in _SINGLE_TOKEN_BLACKLIST:
        return False

    # ── Pure digits / only whitespace ────────────────────────────────────────
    if re.fullmatch(r"[\d\s]+", stripped):
        return False
    # ── Short alphanumeric codes / IDs ────────────────────────────────────────
    if re.search(r"\d", stripped) and len(stripped) <= 20:
        return False

    tokens = norm.split()

    # ── Single-token checks ───────────────────────────────────────────────────
    if len(tokens) == 1:
        t = tokens[0]
        # In _COMMON_NON_NAMES
        if t in _COMMON_NON_NAMES:
            return False
        # All-uppercase single token whose lowercased form is blacklisted
        if stripped.isupper() and t in _SINGLE_TOKEN_BLACKLIST:
            return False
        # All-lowercase single token that is not a known Indian particle
        if stripped == stripped.lower() and t not in _ALLOWED_LOWERCASE_PARTICLES:
            return False
        # Slash-delimited salutation like "Sir/Mam" — reject each part
        if "/" in stripped:
            parts = [p.strip().lower() for p in stripped.split("/")]
            if all(p in _COMMON_NON_NAMES or p in _GENERIC_ROLES for p in parts if p):
                return False

    # ── Single character ──────────────────────────────────────────────────────
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
    # Partial emails: any prefix immediately followed by a known domain (no space).
    # Pattern is intentionally broad — anchored at \b and at domain end — so it
    # catches things like pk8804398358gmail.com and abcgmail.com in one pass.
    # Must run BEFORE the phone pattern so embedded digit runs are protected first.
    (
        r"\b[a-zA-Z0-9._%+\-]{2,}"
        r"(?:gmail|yahoo|hotmail|outlook|rediffmail|ymail|live|icloud)"
        r"(?:\.com)?\b",
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
# LOCATION / ADDRESS CONTEXT GUARD
# =========================================================

# Words that appear NEAR a genuine location/address span (context signals).
# These legitimise a nearby span but should NOT themselves be masked.
_LOCATION_CONTEXT_RE = re.compile(
    r"\b(at|in|near|from|to|of|"
    r"district|village|taluka|tehsil|city|state|"
    r"pin|pincode|pin\s*code|ward|nagar|nagara|"
    r"road|street|marg|lane|gali|colony|sector|phase|plot|block|flat|floor|"
    r"house|h\.?no|door|building|bldg|tower|"
    r"locality|area|zone|region|"
    r"post|p\.?o\.?|po\b|thana|mandal|"
    r"india|state|province|country)\b",
    re.IGNORECASE,
)

# Single-token spans that are themselves location label words, not place names.
# These must never be the masked span — they are context words, not PII.
_LOCATION_LABEL_WORDS = {
    "tehsil", "district", "nagar", "nagara", "village", "taluka",
    "ward", "sector", "phase", "plot", "block", "locality", "area",
    "zone", "region", "thana", "mandal", "colony", "road", "street",
    "lane", "marg", "gali", "state", "city", "country", "province",
}

_LOCATION_CONTEXT_WINDOW: int = 80   # characters on each side of the span


def _has_location_context(text: str, start: int, end: int) -> bool:
    """Return True if the surrounding text contains location-signal words."""
    snippet = text[max(0, start - _LOCATION_CONTEXT_WINDOW):
                   end + _LOCATION_CONTEXT_WINDOW]
    return bool(_LOCATION_CONTEXT_RE.search(snippet))


# =========================================================
# MULTI-TOKEN NAME PROPER-NOUN GUARD
# =========================================================

def _has_name_token(tokens: List[str]) -> bool:
    """
    For a multi-token span to be accepted as a PERSON name, at least one token
    must start with an uppercase letter or be a known Indian name particle.
    This rejects spans like "cancel registration" or "solve my problem" that
    happen to have no capitalisation.
    """
    for t in tokens:
        if not t:
            continue
        if t[0].isupper():
            return True
        if t.lower() in _ALLOWED_LOWERCASE_PARTICLES:
            return True
    return False


# =========================================================
# OVERLAP DEDUPLICATION
# =========================================================

def _dedup_results(results: List[RecognizerResult]) -> List[RecognizerResult]:
    """
    Greedy deduplication: keep the highest-scoring non-overlapping span.
    Ties broken by span length (longer span wins).
    """
    # Sort: highest score first, then longest span first
    ranked = sorted(
        results,
        key=lambda r: (r.score, r.end - r.start),
        reverse=True,
    )
    kept: List[RecognizerResult] = []
    for r in ranked:
        if not any(r.start < k.end and r.end > k.start for k in kept):
            kept.append(r)
    return kept


# =========================================================
# GLINER RECOGNIZER
# =========================================================


# Per-entity confidence thresholds.
# ADDRESS and LOCATION are the noisiest labels so they demand higher confidence.
_ENTITY_THRESHOLDS: Dict[str, float] = {
    "PERSON":       0.85,
    "ORGANIZATION": 0.85,
    "ADDRESS":      0.92,
    "LOCATION":     0.90,
}

# Minimum number of whitespace-separated tokens a span must contain for
# ADDRESS predictions to be accepted.  Single- or double-word ADDRESS hits
# (e.g. "Delhi", "my office") are almost always false positives.
_ADDRESS_MIN_TOKENS: int = 4


class GlinerRecognizer(EntityRecognizer):
    def __init__(self, model_name: str = "urchade/gliner_medium-v2.1", threshold: float = 0.85):
        self.label_mapping = {
            "person name":              "PERSON",
            "company name":             "ORGANIZATION",
            "postal address":           "ADDRESS",
            "geographical place name":  "LOCATION",
        }
        self.gliner_labels = list(self.label_mapping.keys())
        # `threshold` is used as the floor passed to GLiNER; per-entity gates
        # are applied afterwards in analyze() using _ENTITY_THRESHOLDS.
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

        # Use the lowest per-entity threshold as the GLiNER floor so the model
        # returns all candidates; per-entity gates are applied below.
        gliner_floor = min(_ENTITY_THRESHOLDS.values())
        predictions = self.model.predict_entities(text, self.gliner_labels, threshold=gliner_floor)

        for pred in predictions:
            presidio_entity = self.label_mapping.get(pred["label"])
            if not presidio_entity or presidio_entity not in entities:
                continue

            # ── Per-entity confidence gate ────────────────────────────────────
            entity_threshold = _ENTITY_THRESHOLDS.get(presidio_entity, self.threshold)
            if pred["score"] < entity_threshold:
                continue

            span_text = text[pred["start"]: pred["end"]]

            # ── Global whitelist ──────────────────────────────────────────────
            if _in_whitelist(span_text):
                continue

            # ── Skip sentinel tokens (protected structured data) ─────────────
            # A span may start with, end with, or fully contain a sentinel if
            # GLiNER absorbs surrounding characters — catch all cases.
            # Also catch bare hex tokens (8-16 hex chars) that are the inner
            # part of a partially-split sentinel __PII_PROTECT_<hex>__
            if "__PII_" in span_text:
                continue
            if re.fullmatch(r"[0-9a-f]{8,16}", span_text.strip(), re.IGNORECASE):
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
                # Multi-token spans must contain at least one proper-noun token
                tokens = span_text.split()
                if len(tokens) > 1 and not _has_name_token(tokens):
                    continue
                # Reject if the entire span is a known place name
                if span_text.strip().lower() in _KNOWN_PLACE_NAMES:
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
                if norm.strip() in _SINGLE_TOKEN_BLACKLIST:
                    continue

            elif presidio_entity == "ADDRESS":
                # Reject suspiciously short ADDRESS spans — real postal
                # addresses contain at least a house number, street, and area.
                if len(span_text.split()) < _ADDRESS_MIN_TOKENS:
                    continue
                # Must have at least one location-signal word nearby
                if not _has_location_context(text, pred_start, pred_end):
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
                # Reject fiscal year label "F.Y. 2023-24" style spans
                if re.match(r"^f\.?y\.?(\s*\d{2,4})?(\s*[-–]\s*\d{2,4})?$", norm):
                    continue
                if norm.strip() in _SINGLE_TOKEN_BLACKLIST:
                    continue
                # Reject spans that are just location label words, not place names
                if norm.strip() in _LOCATION_LABEL_WORDS:
                    continue
                # Require location-signal context for LOCATION spans as well
                if not _has_location_context(text, pred_start, pred_end):
                    continue

            results.append(
                RecognizerResult(
                    entity_type=presidio_entity,
                    start=pred_start,
                    end=pred_end,
                    score=pred["score"],
                )
            )

        # Remove overlapping spans, keeping the highest-confidence prediction
        return _dedup_results(results)


# =========================================================
# ENGINE SETUP
# =========================================================

def _build_analyzer() -> AnalyzerEngine:
    engine = AnalyzerEngine()
    engine.registry.add_recognizer(GlinerRecognizer(
        model_name="urchade/gliner_medium-v2.1",
        # threshold here is only the GLiNER floor; per-entity gates in
        # _ENTITY_THRESHOLDS are applied inside analyze().
        threshold=min(_ENTITY_THRESHOLDS.values()),
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
# REGEX NAME PRE-PASS  (fix #10)
# =========================================================
# Patterns like "MY NAME IS ASHISH KUMAR" or "I am Leela" are very reliable
# name declarations that don't need ML inference. Catching them here ensures
# all-caps names and other hard cases for GLiNER are still redacted.

_NAME_DECLARATION_RE = re.compile(
    r"""
    (?:
        # "My name is X" / "My name: X"
        \b(?:my\s+name\s+(?:is|:|was))\s+
        ((?:[A-Z][a-zA-Z]*\.?\s*){1,6})   # 1–6 capitalised tokens
    |
        # "I am Mr/Mrs/Dr <Name>" — ONLY with explicit title prefix
        \bi\s+am\s+(?:mr\.?|mrs\.?|ms\.?|dr\.?|smt\.?|shri\.?)\s+
        ((?:[A-Z][a-zA-Z]*\.?\s*){1,6})
    |
        # "Name: X X" / "Name - X X"  (label pattern)
        \b(?:name\s*[:\-]\s*)
        ((?:[A-Z][a-zA-Z]*\.?\s*){1,6})
    |
        # Signature block: "Regards,\n  First Last" or "Sincerely,\n  Name"
        (?:regards|sincerely|yours\s+(?:truly|faithfully|sincerely)|warm\s+regards)
        \s*[,\n]\s*
        ((?:[A-Z][a-zA-Z]*\.?\s*){1,4})
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Minimum character length for a regex-detected name span (avoids single initials)
_REGEX_NAME_MIN_LEN = 3


def _regex_name_hits(text: str) -> List[RecognizerResult]:
    """
    Return Presidio RecognizerResult objects for names found via explicit
    declaration patterns. These complement GLiNER especially for all-caps text
    and very long names.
    """
    results = []
    for m in _NAME_DECLARATION_RE.finditer(text):
        # Take whichever capture group matched
        for g in (1, 2, 3, 4):
            span = m.group(g)
            if not span:
                continue
            span = span.strip()
            if len(span) < _REGEX_NAME_MIN_LEN:
                continue
            if _in_whitelist(span):
                continue
            if not _is_plausible_name(span):
                continue
            # Locate the span inside the full match
            span_start = text.index(span, m.start())
            span_end   = span_start + len(span)
            results.append(RecognizerResult(
                entity_type="PERSON",
                start=span_start,
                end=span_end,
                score=0.95,   # High confidence: pattern is very precise
            ))
            break
    return results


# =========================================================
# POST-OUTPUT CLEANUP  (fix #9)
# =========================================================
# If a partial email was split by the phone sentinel (e.g. "pk [PHONE]gmail.com"),
# the phone mask replaces the digits with [PHONE] leaving a dangling prefix.
# Clean up by collapsing these artefacts into a single [EMAIL] tag.

_SPLIT_EMAIL_RE = re.compile(
    r"\b[a-zA-Z][a-zA-Z0-9._%+\-]*\s*\[(?:PHONE|EMAIL)\]\s*"
    r"(?:gmail|yahoo|hotmail|outlook|rediffmail|ymail|live|icloud)(?:\.com)?\b",
    re.IGNORECASE,
)


def _fix_split_emails(text: str) -> str:
    """Replace split-email artefacts like 'pk[PHONE]gmail.com' with [EMAIL]."""
    return _SPLIT_EMAIL_RE.sub("[EMAIL]", text)




# =========================================================
# PUBLIC API
# =========================================================

def mask_text_entities(text: str, monitor: bool = False) -> str:
    """
    Scans and redacts PERSON, ORGANIZATION, ADDRESS, and LOCATION entities
    using GLiNER via Presidio, supplemented by a regex name pre-pass.

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

        # ── Step 3: Regex name pre-pass — catches all-caps names and
        #            explicit declarations that GLiNER may miss ──────────────
        regex_hits = _regex_name_hits(protected_text)

        # Merge and deduplicate; regex hits included with GLiNER results
        all_hits = _dedup_results(list(raw_hits) + regex_hits)

        redacted = _anonymizer.anonymize(
            text=protected_text,
            analyzer_results=all_hits,
            operators=_OPERATORS,
        )
        output = redacted.text

        # ── Step 4: Restore all protected tokens verbatim ───────────────────
        for sentinel, original in restore_map.items():
            output = output.replace(sentinel, original)

        # ── Step 5: Clean up split-email artefacts ──────────────────────────
        output = _fix_split_emails(output)

        output = _repair_whitespace(output)

    return output