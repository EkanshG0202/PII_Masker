"""
Indian Text PII Masker (GLiNER + IndicNER + Presidio)
=====================================================
Extracts and masks unstructured text-based entities using a hybrid
approach of GLiNER and IndicNER integrated into Microsoft Presidio.

Entities masked:
  PERSON        Person names (IndicNER for semantic, GLiNER for context)
  ORGANIZATION  Company/org names
  ADDRESS       Postal addresses and landmarks (GLiNER)
  LOCATION      Geographical locations, cities, states
"""

import re
import os
import time
import uuid
import psutil
from typing import List, Dict, Tuple, Optional

import torch

# =========================================================
# SECURITY GUARD: PyTorch Version Enforcement
# =========================================================
# PyTorch versions < 2.6 contain a severe arbitrary code execution
# vulnerability in torch.load() via pickle. We strictly enforce 2.6+.
_torch_v = torch.__version__.split('+')[0].split('.')
if int(_torch_v[0]) < 2 or (int(_torch_v[0]) == 2 and int(_torch_v[1]) < 6):
    raise SystemExit(
        f"🚨 SECURITY HALT: Current PyTorch version is {torch.__version__}.\n"
        f"Due to arbitrary code execution risks in `torch.load()`, you MUST upgrade to PyTorch 2.6.0 or higher.\n"
        f"Please run: pip install --upgrade torch"
    )

from transformers import pipeline
from huggingface_hub import login

# 1. Force the authentication at the environment level
my_token = "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" # <-- PASTE YOUR REAL TOKEN HERE
login(token=my_token)
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
    "letter head", "letterhead",
    "spouse", "my spouse", "wife", "husband",
    "persan", "persan name",
    "tehsil",
}

_GENERIC_ACRONYMS = {
    "otp", "sms", "email", "email id", "mail id", "mobile number",
    "mobile no", "pan", "pan number", "pan no", "aadhaar", "aadhar",
    "uan", "urc", "gstin", "gst", "ifsc", "neft", "rtgs",
    "pdf", "otp number", "registration number", "application number",
    "certificate", "udyam certificate", "udyam number",
    "llp", "pvt", "ltd", "pvt ltd", "private limited",
    "f.y.", "f.y", "fy",
    "d.o.b", "d.o.b.", "dob",
}

KNOWN_TERMS: set = _GOV_SCHEMES | _GENERIC_ROLES | _GENERIC_ACRONYMS

_SINGLE_TOKEN_BLACKLIST = {
    "udyam", "udhyam", "udyog", "registration", "cancel", "cancell",
    "solve", "mam", "latitude", "longitude", "i", "already", "forgot",
    "alrady", "tehsil", "ho", "unable", "writing", "trying", "clicking",
    "applying", "email", "otp", "us", "f.y.", "fy", "f.y", "alrady",
    "cement business", "d.o.b", "dob", "llp", "udyog adhar", "udyog aadhar",
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
    "udyam", "udhyam", "udyog", "udaym",
    "i", "we", "our", "my",
    "cancel", "cancell", "cancellation", "solve", "already", "forgot", "mam",
    "alrady", "latitude", "longitude", "email", "otp",
    "aadhaar", "aadhar", "adhar", "addhar",
    "letter head", "letterhead",
    "tehsil", "district", "nagar", "village", "taluka",
    "spouse", "wife", "husband", "persan", "ho",
    "update", "change", "reset", "verify", "register",
    "download", "upload", "migrate", "retrieve",
    "unable", "writing", "trying", "applying", "clicking", "attempting",
    "running", "still",
}

_NON_NAME_PHRASES = {
    "solve my problem", "solve my", "letter head",
    "udyam aadhar", "udyam aadhaar", "udyam registration",
    "udhyam registration", "udaym registration", "udyog aadhar",
    "udyog aadhaar", "udyog adhar", "my spouse", "d.o.b.", "d.o.b",
    "persan name", "cancel my", "cancell my", "please cancel",
    "please cancell", "please solve", "please update", "please change",
    "please register", "please verify", "please reset", "please retrieve",
    "please download", "please upload", "please migrate", "unable to",
    "not able to", "trying to", "writing to", "writing to request",
    "applying to", "attempting to", "clicking on", "continuously trying",
    "still not", "cancell", "cancel",
}

_NON_NAME_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in sorted(_NON_NAME_PHRASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

def _is_non_name_phrase(text: str) -> bool:
    norm = text.strip().lower()
    if norm in _NON_NAME_PHRASES:
        return True
    m = _NON_NAME_PHRASE_RE.fullmatch(norm)
    return m is not None

_ALLOWED_LOWERCASE_PARTICLES = {"ji", "kumar", "devi", "bai", "lal", "ram"}

_KNOWN_PLACE_NAMES = {
    "maharashtra", "gujarat", "rajasthan", "karnataka", "kerala",
    "tamilnadu", "tamil nadu", "andhra pradesh", "telangana", "odisha",
    "west bengal", "uttar pradesh", "madhya pradesh", "bihar", "jharkhand",
    "chhattisgarh", "uttarakhand", "himachal pradesh", "punjab", "haryana",
    "delhi", "goa", "assam", "manipur", "meghalaya", "tripura", "nagaland",
    "mizoram", "arunachal pradesh", "sikkim", "jammu", "kashmir",
    "chandigarh", "puducherry", "pondicherry",
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
    "ludhiana", "amritsar", "jalandhar",
    "gurgaon", "gurugram", "faridabad", "ambala",
    "noida", "ghaziabad", "meerut", "bareilly", "moradabad",
    "srinagar", "leh", "udhampur", "belgachia", "kishanganj",
}

def _is_plausible_name(text: str) -> bool:
    stripped = text.strip()
    norm = stripped.lower()

    if _is_non_name_phrase(norm):
        return False

    cleaned = re.sub(r"[^\w\s]", "", norm).strip()
    if (norm in KNOWN_TERMS or cleaned in KNOWN_TERMS
            or norm in _SINGLE_TOKEN_BLACKLIST or cleaned in _SINGLE_TOKEN_BLACKLIST):
        return False

    punct_stripped = norm.rstrip(".,;:!? ")
    if punct_stripped in KNOWN_TERMS or punct_stripped in _SINGLE_TOKEN_BLACKLIST:
        return False

    if re.fullmatch(r"[\d\s]+", stripped):
        return False

    if re.search(r"\d", stripped) and len(stripped) <= 20:
        return False

    tokens = norm.split()
    if len(tokens) == 1:
        t = tokens[0]
        if t in _COMMON_NON_NAMES:
            return False
        if stripped.isupper() and t in _SINGLE_TOKEN_BLACKLIST:
            return False
        if stripped == stripped.lower() and t not in _ALLOWED_LOWERCASE_PARTICLES:
            return False
        if "/" in stripped:
            parts = [p.strip().lower() for p in stripped.split("/")]
            if all(p in _COMMON_NON_NAMES or p in _GENERIC_ROLES for p in parts if p):
                return False

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
# REGEX PRE-PASS
# =========================================================

_PROTECT_PATTERNS: List[Tuple[str, int]] = [
    (
        r"\b(?:UDYAM|UDAYM|UDHYAM)[- ][A-Z]{2}[- ]\d{2}[- ]\d{7}"
        r"(?:/[A-Z]/\d{5})?",
        re.IGNORECASE,
    ),
    (
        r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-CE-Z]\d{7}(?!\w)",
        re.IGNORECASE,
    ),
    (
        r"\b(0?[1-9]|[12]\d|3[01])[/\-](0?[1-9]|1[0-2])[/\-](19|20)\d{2}\b",
        0,
    ),
    (r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE),
    (r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b", re.IGNORECASE),
    (r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b", 0),
    (
        r"\b[a-zA-Z0-9._%+\-]{2,}"
        r"(?:gmail|yahoo|hotmail|outlook|rediffmail|ymail|live|icloud)"
        r"(?:\.com)?\b",
        re.IGNORECASE,
    ),
    (
        r"(?<!\d)(?:\+91[\s\-]?|91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)",
        0,
    ),
    (r"(?<![_\d])\d{9,18}(?![_\d])", 0),
]

def _regex_protect(text: str) -> Tuple[str, Dict[str, str]]:
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

_LOCATION_LABEL_WORDS = {
    "tehsil", "district", "nagar", "nagara", "village", "taluka",
    "ward", "sector", "phase", "plot", "block", "locality", "area",
    "zone", "region", "thana", "mandal", "colony", "road", "street",
    "lane", "marg", "gali", "state", "city", "country", "province",
}

_LOCATION_CONTEXT_WINDOW: int = 80

def _has_location_context(text: str, start: int, end: int) -> bool:
    snippet = text[max(0, start - _LOCATION_CONTEXT_WINDOW):
                   end + _LOCATION_CONTEXT_WINDOW]
    return bool(_LOCATION_CONTEXT_RE.search(snippet))

def _has_name_token(tokens: List[str]) -> bool:
    for t in tokens:
        if not t:
            continue
        if t[0].isupper():
            return True
        if t.lower() in _ALLOWED_LOWERCASE_PARTICLES:
            return True
    return False

def _dedup_results(results: List[RecognizerResult]) -> List[RecognizerResult]:
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
# RECOGNIZERS
# =========================================================

_ENTITY_THRESHOLDS: Dict[str, float] = {
    "PERSON":       0.85,
    "ORGANIZATION": 0.85,
    "ADDRESS":      0.92,
    "LOCATION":     0.90,
}

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
        self.threshold = threshold

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"[text-masking] Loading GLiNER on {device.upper()}...")
        # Explicitly request safetensors for security
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
        gliner_floor = min(_ENTITY_THRESHOLDS.values())
        predictions = self.model.predict_entities(text, self.gliner_labels, threshold=gliner_floor)

        for pred in predictions:
            presidio_entity = self.label_mapping.get(pred["label"])
            if not presidio_entity or presidio_entity not in entities:
                continue

            entity_threshold = _ENTITY_THRESHOLDS.get(presidio_entity, self.threshold)
            if pred["score"] < entity_threshold:
                continue

            span_text = text[pred["start"]: pred["end"]]

            if _in_whitelist(span_text):
                continue

            if "__PII_" in span_text or re.fullmatch(r"[0-9a-f]{8,16}", span_text.strip(), re.IGNORECASE):
                continue

            pred_start, pred_end = pred["start"], pred["end"]

            if presidio_entity == "PERSON":
                span_text, pred_start, pred_end = _trim_role_suffix(span_text, pred_start, pred_end)
                if not span_text or not _is_plausible_name(span_text):
                    continue
                tokens = span_text.split()
                if len(tokens) > 1 and not _has_name_token(tokens):
                    continue
                if span_text.strip().lower() in _KNOWN_PLACE_NAMES:
                    continue

            elif presidio_entity == "ORGANIZATION":
                norm = span_text.strip().lower()
                cleaned_norm = re.sub(r"[^\w\s]", "", norm)
                if norm in _GOV_SCHEMES or cleaned_norm in _GOV_SCHEMES:
                    continue
                if norm in {"i", "we", "my", "d.o.b", "dob", "llp", "district industries centre", "udyog adhar", "udyog aadhar", "udyog aadhaar"}:
                    continue
                if norm.strip() in _SINGLE_TOKEN_BLACKLIST:
                    continue

            elif presidio_entity == "ADDRESS":
                if len(span_text.split()) < _ADDRESS_MIN_TOKENS:
                    continue
                if not _has_location_context(text, pred_start, pred_end):
                    continue

            elif presidio_entity == "LOCATION":
                norm = span_text.strip().lower()
                if norm in {"us", "f.y.", "fy", "f.y", "i", "we", "alrady"}:
                    continue
                if re.match(r"^f\.?y\.?\s*\d{4}", norm) or re.match(r"^f\.?y\.?(\s*\d{2,4})?(\s*[-–]\s*\d{2,4})?$", norm):
                    continue
                if norm.strip() in _SINGLE_TOKEN_BLACKLIST or norm.strip() in _LOCATION_LABEL_WORDS:
                    continue
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

        return _dedup_results(results)


class IndicNerRecognizer(EntityRecognizer):
    def __init__(self, model_name: str = "ai4bharat/IndicNER", threshold: float = 0.85):
        self.label_mapping = {
            "PER": "PERSON",
            "ORG": "ORGANIZATION",
            "LOC": "LOCATION",
        }
        self.threshold = threshold

        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1

        print(f"[text-masking] Loading IndicNER on device {device}...")
        
        # Load normally. PyTorch 2.6+ handles the secure loading of the .bin file automatically.
        self.nlp = pipeline(
            "ner", 
            model=model_name, 
            aggregation_strategy="simple", 
            device=device
        )
            
        print("[text-masking] IndicNER model loaded and ready.")

        super().__init__(
            supported_entities=list(self.label_mapping.values()),
            name="IndicNerRecognizer",
        )

    def load(self):
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []
        predictions = self.nlp(text)

        for pred in predictions:
            mapped_entity = self.label_mapping.get(pred.get("entity_group"))
            
            if not mapped_entity or mapped_entity not in entities:
                continue

            if pred["score"] < self.threshold:
                continue

            span_text = pred["word"]
            pred_start = pred["start"]
            pred_end = pred["end"]

            if _in_whitelist(span_text):
                continue
            if "__PII_" in span_text or re.fullmatch(r"[0-9a-f]{8,16}", span_text.strip(), re.IGNORECASE):
                continue

            if mapped_entity == "PERSON":
                span_text, pred_start, pred_end = _trim_role_suffix(span_text, pred_start, pred_end)
                if not span_text or not _is_plausible_name(span_text):
                    continue
                tokens = span_text.split()
                if len(tokens) > 1 and not _has_name_token(tokens):
                    continue
                if span_text.strip().lower() in _KNOWN_PLACE_NAMES:
                    continue

            elif mapped_entity == "ORGANIZATION":
                norm = span_text.strip().lower()
                cleaned_norm = re.sub(r"[^\w\s]", "", norm)
                if norm in _GOV_SCHEMES or cleaned_norm in _GOV_SCHEMES:
                    continue
                if norm.strip() in _SINGLE_TOKEN_BLACKLIST:
                    continue

            elif mapped_entity == "LOCATION":
                norm = span_text.strip().lower()
                if norm in {"us", "f.y.", "fy", "f.y", "i", "we", "alrady"} or norm.strip() in _SINGLE_TOKEN_BLACKLIST:
                    continue
                if norm.strip() in _LOCATION_LABEL_WORDS:
                    continue

            results.append(
                RecognizerResult(
                    entity_type=mapped_entity,
                    start=pred_start,
                    end=pred_end,
                    score=float(pred["score"]),
                )
            )

        return _dedup_results(results)


# =========================================================
# ENGINE SETUP
# =========================================================

def _build_analyzer() -> AnalyzerEngine:
    engine = AnalyzerEngine()
    engine.registry.add_recognizer(GlinerRecognizer(
        model_name="urchade/gliner_medium-v2.1",
        threshold=min(_ENTITY_THRESHOLDS.values()),
    ))
    engine.registry.add_recognizer(IndicNerRecognizer(
        model_name="ai4bharat/IndicNER",
        threshold=0.85 
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
# REGEX NAME PRE-PASS
# =========================================================

_NAME_DECLARATION_RE = re.compile(
    r"""
    (?:
        \b(?:my\s+name\s+(?:is|:|was))\s+
        ((?:[A-Z][a-zA-Z]*\.?\s*){1,6})
    |
        \bi\s+am\s+(?:mr\.?|mrs\.?|ms\.?|dr\.?|smt\.?|shri\.?)\s+
        ((?:[A-Z][a-zA-Z]*\.?\s*){1,6})
    |
        \b(?:name\s*[:\-]\s*)
        ((?:[A-Z][a-zA-Z]*\.?\s*){1,6})
    |
        (?:regards|sincerely|yours\s+(?:truly|faithfully|sincerely)|warm\s+regards)
        \s*[,\n]\s*
        ((?:[A-Z][a-zA-Z]*\.?\s*){1,4})
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_REGEX_NAME_MIN_LEN = 3

def _regex_name_hits(text: str) -> List[RecognizerResult]:
    results = []
    for m in _NAME_DECLARATION_RE.finditer(text):
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
            span_start = text.index(span, m.start())
            span_end   = span_start + len(span)
            results.append(RecognizerResult(
                entity_type="PERSON",
                start=span_start,
                end=span_end,
                score=0.95,
            ))
            break
    return results


# =========================================================
# POST-OUTPUT CLEANUP
# =========================================================

_SPLIT_EMAIL_RE = re.compile(
    r"\b[a-zA-Z][a-zA-Z0-9._%+\-]*\s*\[(?:PHONE|EMAIL)\]\s*"
    r"(?:gmail|yahoo|hotmail|outlook|rediffmail|ymail|live|icloud)(?:\.com)?\b",
    re.IGNORECASE,
)

def _fix_split_emails(text: str) -> str:
    return _SPLIT_EMAIL_RE.sub("[EMAIL]", text)


# =========================================================
# PUBLIC API
# =========================================================

def mask_text_entities(text: str, monitor: bool = False) -> str:
    with ResourceMonitor("mask_text_entities", print_report=monitor):
        protected_text, restore_map = _regex_protect(text)

        raw_hits = _analyzer.analyze(
            text=protected_text,
            language="en",
            entities=_ENTITIES,
        )

        regex_hits = _regex_name_hits(protected_text)

        all_hits = _dedup_results(list(raw_hits) + regex_hits)

        redacted = _anonymizer.anonymize(
            text=protected_text,
            analyzer_results=all_hits,
            operators=_OPERATORS,
        )
        output = redacted.text

        for sentinel, original in restore_map.items():
            output = output.replace(sentinel, original)

        output = _fix_split_emails(output)
        output = _repair_whitespace(output)

    return output