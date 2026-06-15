"""
Indian Text PII Masker (GLiNER + Presidio)
==========================================
Extracts and masks unstructured text-based entities using
GLiNER integrated into Microsoft Presidio.

Entities masked:
  PERSON        Person names
  ORGANIZATION  Company/org names
  ADDRESS       Postal addresses and landmarks
"""

import re
from typing import List, Dict, Optional

import torch

from presidio_analyzer import AnalyzerEngine, RecognizerResult, EntityRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from gliner import GLiNER


# =========================================================
# STRUCTURED IDENTIFIER PROTECTION
# =========================================================
# These patterns are replaced with sentinels BEFORE GLiNER runs,
# then restored verbatim AFTER masking.

_PROTECT_PATTERNS = [
    # AADHAAR — 12 digits in 4-4-4 groups (space, hyphen, or none)
    r"(?<!\d)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)",
    # PAN — AAAAA9999A
    r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])",
    # GST — 29ABCDE1234F1Z5
    r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b",
    # IFSC — 4 letters + 0 + 6 alphanumeric
    r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    # PHONE — Indian mobile numbers
    r"(?<!\d)(?:\+91[\s\-]?|91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)",
    # ACCOUNT — context-anchored bank account
    r"(?<=account[\s:])\s*\d{11,18}(?!\d)",
    # VOTER_ID — 3 letters + 7 digits
    r"(?<![A-Z0-9])[A-Z]{3}\d{7}(?![A-Z0-9])",
    # PASSPORT — 1 letter + 7 digits
    r"(?<![A-Z0-9])[A-Z]\d{7}(?![A-Z0-9])",
    # DRIVING LICENCE — SS-RR-YYYY-NNNNNNN
    r"\b[A-Z]{2}[- ]\d{2}[- ]\d{4}[- ]\d{7}\b",
    # UDYAM number
    r"\b(?:UDYAM|UDAYM|UDHYAM)[- ][A-Z]{2}[- ]\d{2}[- ]\d{7}(?:/[A-Z]/\d{5})?",
    # UAM/URC numbers (alphanumeric codes like BR26D0018623, KR01A0000045)
    r"\b[A-Z]{2}\d{2}[A-Z]\d{7}\b",
    # UAN — context-anchored 12 digits
    r"(?<=uan[\s:])\s*\d{12}(?!\d)",
    # EMAIL
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    # URLs
    r"https?://\S+",
    # ── Non-PII keywords GLiNER commonly mislabels as PERSON/ORG ──
    # Aadhaar/aadhar word variants
    r"\b(?:aadhaar|aadhar|adhar|addhar|e-aasdhr)"
    r"(?:\s+(?:number|no\.?|card|link|update|correction|enrollment|enrolment|seeding|linked|registered|validation|verified))?\b",
    # Udyam / Udyog / Udhyog keyword phrases
    r"\b(?:"
    r"udyam|udhyam|udaym|udayam"
    r"|udyam\s+registration|udhyam\s+registration|udaym\s+registration"
    r"|udyam\s+portal|udyam\s+number|udyam\s+no|udyam\s+certificate"
    r"|udyam\s+assist(?:\s+platform)?"
    r"|ud(?:y|h?y)og\s+a[ad]ha?ar"
    r"|ud(?:y|h?y)og\s+a[ad]ha?ar\s+(?:number|no\.?|registration|portal|certificate|memorandum|uam)"
    r"|udhyog\s+aadhaar\s+memorandum|udyog\s+aadhar\s+memorandum"
    r"|uam|urc"
    r")\b",
    # Registration / email / OTP / action words
    r"\b(?:registration|registartion|registrations)\b",
    r"\b(?:email|e-mail|email\s+id|mail\s+id)\b",
    r"\b(?:otp|otp\s+number)\b",
    r"\b(?:cancell?(?:ation)?|cancel(?:l?ed)?)\b",
    r"\b(?:solve|solved|solution|resolve|resolved)\b",
    r"\b(?:alr?ea?dy|alrady)\b",
    r"\b(?:unable|trying|clicking|applying|attempting|writing|didt)\b",
    r"\b(?:sir|madam|mam|dear\s+sir|dear\s+madam|dear\s+sir\s*/\s*madam)\b",
    r"\b(?:proprietor|sole\s+proprietor|respondent|applicant|complainant)\b",
    r"\b(?:application\s+)?dt\.?\b",
    r"\b(?:latitude|longitude|geolocation|geo\s*tag(?:ging)?)\b",
    r"\b(?:letter\s*head|letterhead)\b",
    r"\b(?:msme|msefc|msmed|pmegp|kvic|sidbi|nsic|cgtmse)\b",
]

_PROTECT_RE = re.compile(
    "|".join(f"(?:{p})" for p in _PROTECT_PATTERNS),
    re.IGNORECASE,
)

_SENTINEL_PREFIX = "__PROTECT_"
_SENTINEL_SUFFIX = "__"


def _protect_identifiers(text: str) -> tuple[str, Dict[str, str]]:
    restore_map: Dict[str, str] = {}
    counter = [0]

    def _repl(m):
        key = f"{_SENTINEL_PREFIX}{counter[0]}{_SENTINEL_SUFFIX}"
        restore_map[key] = m.group(0)
        counter[0] += 1
        return key

    return _PROTECT_RE.sub(_repl, text), restore_map


def _restore_identifiers(text: str, restore_map: Dict[str, str]) -> str:
    for sentinel, original in restore_map.items():
        text = text.replace(sentinel, original)
    return text


# =========================================================
# PERSON SPAN VALIDATOR
# =========================================================
# Post-GLiNER guard: rejects PERSON predictions that are clearly not names.

# Business/generic nouns that end up in 2-token "X Business" spans
_BUSINESS_TAIL_WORDS = {
    "business", "enterprise", "firm", "company", "shop", "store",
    "agency", "bureau", "centre", "center", "services", "service",
    "trading", "industries", "industry", "corporation", "associates",
    "ventures", "venture", "works", "solutions", "consultancy",
    "suppliers", "supplier", "dealers", "dealer", "products",
    "exports", "imports", "logistics", "technologies", "tech",
    "systems", "system", "group", "holding", "holdings",
}

# Single tokens that look superficially name-like but aren't
_SINGLE_TOKEN_NON_NAMES = {
    # Misspellings / typos / action words
    "claear", "didt", "canu", "forgut", "alrady", "cancell",
    "issie", "privde", "provied", "provied", "recev", "resev",
    "resubmit", "migrate", "retrieve", "download", "upload",
    # Document/portal words
    "attachment", "letterhead", "certificate", "document", "letter",
    "annexure", "enclosure", "affidavit", "invoice", "receipt",
    "bharatmapservice", "webgis", "portal", "website",
    # Commodities / generic nouns
    "milk", "cement", "rice", "wheat", "cotton", "gold", "silver",
    "iron", "steel", "wood", "cloth", "cloth",
    # Roles already in protect list but belt-and-suspenders
    "respondent", "petitioner", "complainant", "proprietor",
    "applicant", "director", "manager", "owner", "partner",
}

# Name honorifics / prefixes — single-token spans with these are NOT names
_HONORIFICS = {"mr", "mrs", "ms", "dr", "shri", "smt", "prof", "er"}

# Name particles allowed as single lowercase tokens
_NAME_PARTICLES = {"ji", "kumar", "devi", "bai", "lal", "ram", "singh",
                   "devi", "ben", "bhai", "rao", "nair", "das"}


def _is_valid_person_span(span_text: str) -> bool:
    """Return False if the span is clearly not a person name."""
    raw = span_text.strip()
    if not raw:
        return False

    # Contains a sentinel — GLiNER tagged a protected token
    if "__PROTECT_" in raw:
        return False

    # Pure digits / punctuation
    if re.fullmatch(r"[\d\s\.\-/,;:]+", raw):
        return False

    # Strip trailing punctuation artifacts (e.g. "Rishi/s" → "Rishi")
    cleaned = re.sub(r"[/\\.,;:!?\s]+$", "", raw).strip()
    if not cleaned:
        return False

    norm = cleaned.lower()
    tokens = norm.split()

    # Single token checks
    if len(tokens) == 1:
        t = tokens[0]
        # All-lowercase single token that's not a known particle
        if raw == raw.lower() and t not in _NAME_PARTICLES:
            return False
        # Known non-name single token
        if t in _SINGLE_TOKEN_NON_NAMES:
            return False
        # Honorific alone (no actual name)
        if t.rstrip(".") in _HONORIFICS:
            return False
        # All-uppercase token that's 6+ chars and contains no vowels
        # (likely an acronym or garbled word, e.g. CLAEAR, DIDT)
        if raw.isupper() and len(t) >= 4:
            vowels = set("aeiou")
            if not any(c in vowels for c in t):
                return False

    # Multi-token: last token is a generic business noun → ORG, not NAME
    if len(tokens) >= 2 and tokens[-1] in _BUSINESS_TAIL_WORDS:
        return False

    # Multi-token: ALL tokens are lowercase and none are name particles
    if all(tok == tok.lower() for tok in tokens):
        if not any(tok in _NAME_PARTICLES for tok in tokens):
            return False

    # Span is longer than 6 tokens — very unlikely to be a single person name
    if len(tokens) > 6:
        return False

    return True


# =========================================================
# POST-MASK CLEANUP
# =========================================================

_BROKEN_SENTINEL_RE = re.compile(
    r"(?:\[)?__(?:PROTECT_\d+|\[(?:NAME|ORG|ADDRESS)\])__(?:\])?",
    re.IGNORECASE,
)
_DOUBLE_BRACKET_RE = re.compile(r"\[\[([A-Z]+)\]\]")


def _cleanup_artefacts(text: str) -> str:
    text = _BROKEN_SENTINEL_RE.sub("", text)
    text = _DOUBLE_BRACKET_RE.sub(r"[\1]", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# =========================================================
# GLINER RECOGNIZER
# =========================================================

_ENTITY_THRESHOLDS: Dict[str, float] = {
    "PERSON":       0.75,
    "ORGANIZATION": 0.85,
    "ADDRESS":      0.80,
}

_ADDRESS_MIN_TOKENS: int = 2


class GlinerRecognizer(EntityRecognizer):
    FINETUNED_LABEL_MAPPING = {
        "full_name":      "PERSON",
        "company_name":   "ORGANIZATION",
        "postal_address": "ADDRESS",
    }

    def __init__(self, finetuned_model_path: str = "./gliner_pii_finetuned"):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"[text-masking] Loading GLiNER from '{finetuned_model_path}' on {device.upper()} ...")
        self.finetuned_model = GLiNER.from_pretrained(finetuned_model_path).to(device)
        self.finetuned_labels = list(self.FINETUNED_LABEL_MAPPING.keys())
        print("[text-masking] GLiNER model loaded and ready.")

        super().__init__(
            supported_entities=list(self.FINETUNED_LABEL_MAPPING.values()),
            name="GlinerRecognizer",
        )

    def load(self):
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []
        gliner_floor = min(_ENTITY_THRESHOLDS.values())

        for pred in self.finetuned_model.predict_entities(
            text, self.finetuned_labels, threshold=gliner_floor
        ):
            presidio_entity = self.FINETUNED_LABEL_MAPPING.get(pred["label"])
            if not presidio_entity or presidio_entity not in entities:
                continue
            if pred["score"] < _ENTITY_THRESHOLDS[presidio_entity]:
                continue

            span_text = pred["text"]

            # Reject if span overlaps a sentinel
            if "__PROTECT_" in span_text:
                continue

            # Entity-specific validation
            if presidio_entity == "PERSON":
                if not _is_valid_person_span(span_text):
                    continue

            elif presidio_entity == "ADDRESS":
                if len(span_text.split()) < _ADDRESS_MIN_TOKENS:
                    continue

            results.append(RecognizerResult(
                entity_type=presidio_entity,
                start=pred["start"],
                end=pred["end"],
                score=pred["score"],
            ))

        return _dedup_results(results)


def _dedup_results(results: List[RecognizerResult]) -> List[RecognizerResult]:
    ranked = sorted(results, key=lambda r: (r.score, r.end - r.start), reverse=True)
    kept: List[RecognizerResult] = []
    for r in ranked:
        if not any(r.start < k.end and r.end > k.start for k in kept):
            kept.append(r)
    return kept


# =========================================================
# ENGINE SETUP
# =========================================================

_FINETUNED_PATH = "C:/College/PS-1/PII Masking/gliner_pii_finetuned"

_analyzer = AnalyzerEngine()
_analyzer.registry.add_recognizer(GlinerRecognizer(finetuned_model_path=_FINETUNED_PATH))
_anonymizer = AnonymizerEngine()

_ENTITIES = ["PERSON", "ORGANIZATION", "ADDRESS"]

_OPERATORS = {
    "PERSON":       OperatorConfig("replace", {"new_value": "[NAME]"}),
    "ORGANIZATION": OperatorConfig("replace", {"new_value": "[ORG]"}),
    "ADDRESS":      OperatorConfig("replace", {"new_value": "[ADDRESS]"}),
}


# =========================================================
# PUBLIC API
# =========================================================

def mask_text_entities(text: str) -> str:
    # Step 1: Protect structured identifiers and known non-PII keywords
    protected_text, restore_map = _protect_identifiers(text)

    # Step 2: Run GLiNER via Presidio
    hits = _analyzer.analyze(text=protected_text, language="en", entities=_ENTITIES)
    redacted = _anonymizer.anonymize(
        text=protected_text,
        analyzer_results=hits,
        operators=_OPERATORS,
    )

    # Step 3: Restore protected originals
    output = _restore_identifiers(redacted.text, restore_map)

    # Step 4: Clean up broken sentinel artefacts
    output = _cleanup_artefacts(output)
    return output


# =========================================================
# TESTING & EXAMPLES
# =========================================================

if __name__ == "__main__":
    print("Initializing PII Masker... (This may take a moment to load the models)")

    test_cases = [
        # Correct masking
        "My name is Rahul Sharma and my contact number is +91-9876543210. Please email me at rahul.sharma@gmail.com.",
        "The sole proprietor, Mr. Amit Kumar, applied for UDYAM registration. Udyam number: UDYAM-MH-18-0123456.",
        "I am Dr. Sneha Desai. I live at Flat No 402, Sunshine Tower, MG Road, Bangalore.",
        "Tata Consultancy Services is located in Pune. The Managing Director, Rajesh Gopinathan, signed the document.",
        # False positive cases from real data
        "DEAR SIR, I AM A PROPRIETOR HAVING PAN: GZDPD7433L AND AADHAR NUMBER: 9759 9062 6267.",
        "I am running a Cement Business and want to get registered with MSME.",
        "then I failed to enter OTP as my earlier mobile number has been changed.",
        "Udyam Registration (UDYAM-WB-05-0001331) Certificate.",
        "I want to update my phone number and email id in my UAM that is BR26D0018623.",
        "When i am trying to do registration after Aadhar validation successful.",
        "Application dt.06/06/2024 is still not converted in case.",
        "UDHYOG AADHAAR MEMORANDUM HAS ALREADY BEEN DONE THROUGH THIS AADHAR NUMBER.",
        "Udyog Aadhaar Number: MH18D0032045  Enterprise Name: P. O. P. Decorator  Date of Registration: 20/05/2018",
    ]

    print("\n" + "=" * 60)
    print("RUNNING PII MASKING EXAMPLES")
    print("=" * 60 + "\n")

    for i, text in enumerate(test_cases, 1):
        print(f"--- Example {i} ---")
        print(f"ORIGINAL: {text}")
        masked = mask_text_entities(text)
        print(f"MASKED:   {masked}\n")

    print("Testing complete.")