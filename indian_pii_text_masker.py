"""
Indian Text PII Masker (GLiNER + Presidio)
==========================================
Masks PERSON, ORGANIZATION, and ADDRESS entities using a fine-tuned
GLiNER model via Microsoft Presidio.

Input is assumed to be pre-processed — structured identifiers (Aadhaar,
PAN, phone numbers, etc.) are already masked before this runs.

spaCy / Presidio's built-in NER is intentionally disabled — all entity
detection is handled exclusively by the GLiNER model.
"""

import re
from typing import List, Dict

import torch

from presidio_analyzer import AnalyzerEngine, RecognizerResult, EntityRecognizer, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from gliner import GLiNER


# =========================================================
# GLINER RECOGNIZER
# =========================================================

_ENTITY_THRESHOLDS: Dict[str, float] = {
    "PERSON":       0.75,
    "ORGANIZATION": 0.85,
    "ADDRESS":      0.80,
}

# Matches any already-masked placeholder like [PHONE], [EMAIL], [NAME], etc.
_ALREADY_MASKED = re.compile(r'^\[[A-Z_]+\]$')


class GlinerRecognizer(EntityRecognizer):
    LABEL_MAPPING = {
        "full_name":      "PERSON",
        "company_name":   "ORGANIZATION",
        "postal_address": "ADDRESS",
    }

    def __init__(self, model_path: str = "./gliner_pii_finetuned"):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"[text-masking] Loading GLiNER from '{model_path}' on {device.upper()} ...")
        self.model = GLiNER.from_pretrained(model_path).to(device)
        self.gliner_labels = list(self.LABEL_MAPPING.keys())
        print("[text-masking] GLiNER model loaded and ready.")

        super().__init__(
            supported_entities=list(self.LABEL_MAPPING.values()),
            name="GlinerRecognizer",
        )

    def load(self):
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        gliner_floor = min(_ENTITY_THRESHOLDS.values())
        results = []

        raw_preds = self.model.predict_entities(text, self.gliner_labels, threshold=gliner_floor)
        print(f"[DEBUG] Input: {text!r}")
        for p in raw_preds:
            print(f"[DEBUG]   label={p['label']!r}  score={p['score']:.4f}  text={p['text']!r}")

        for pred in raw_preds:
            presidio_entity = self.LABEL_MAPPING.get(pred["label"])
            if not presidio_entity or presidio_entity not in entities:
                continue

            # Option 1: skip spans that are already-masked placeholders
            if _ALREADY_MASKED.match(pred["text"].strip()):
                print(f"[DEBUG]   SKIPPED (already masked placeholder): label={pred['label']!r}  text={pred['text']!r}")
                continue

            if pred["score"] < _ENTITY_THRESHOLDS[presidio_entity]:
                print(f"[DEBUG]   DROPPED (below threshold): label={pred['label']!r}  score={pred['score']:.4f}  text={pred['text']!r}")
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
# A fresh empty RecognizerRegistry + nlp_engine=None means Presidio
# has no built-in recognizers and spaCy never runs.
# GlinerRecognizer is the sole source of entity predictions.

_MODEL_PATH = "C:/College/PS-1/PII Masking/gliner_pii_finetuned"

_registry = RecognizerRegistry()

_analyzer = AnalyzerEngine(
    registry=_registry,
    nlp_engine=None,
)

_analyzer.registry.recognizers.clear()
_analyzer.registry.add_recognizer(GlinerRecognizer(model_path=_MODEL_PATH))
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
    # Option 3: stash existing placeholders so GLiNER never sees them,
    # then restore them after anonymization.
    placeholder_pattern = re.compile(r'\[[A-Z_]+\]')
    stash: Dict[str, str] = {}

    def _stash(m: re.Match) -> str:
        key = f"__SLOT{len(stash)}__"
        stash[key] = m.group(0)
        return key

    clean_text = placeholder_pattern.sub(_stash, text)

    hits = _analyzer.analyze(text=clean_text, language="en", entities=_ENTITIES)
    redacted = _anonymizer.anonymize(
        text=clean_text,
        analyzer_results=hits,
        operators=_OPERATORS,
    )

    # Restore original placeholders
    result = redacted.text
    for key, original in stash.items():
        result = result.replace(key, original)

    return result


# =========================================================
# TESTING & EXAMPLES
# =========================================================

if __name__ == "__main__":
    print("Initializing PII Masker... (This may take a moment to load the models)")

    test_cases = [
        "My name is Rahul Sharma and my contact number is +91-9876543210. Please email me at rahul.sharma@gmail.com.",
        "The sole proprietor, Mr. Amit Kumar, applied for UDYAM registration. Udyam number: UDYAM-MH-18-0123456.",
        "I am Dr. Sneha Desai. I live at Flat No 402, Sunshine Tower, MG Road, Bangalore.",
        "Tata Consultancy Services is located in Pune. The Managing Director, Rajesh Gopinathan, signed the document.",
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