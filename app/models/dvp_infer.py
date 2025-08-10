# app/models/dvp_infer.py
from __future__ import annotations

import os
import json
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Optional: FAISS for retrieval of prior procedures (if you mount a dataset)
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# ---------------- Config ----------------
# Point to your fine-tuned DVP checkpoint directory (merged weights or LoRA-merged)
# Override via env var in Cloud Run: DVP_MODEL_DIR=/models/mistral_finetuned_Design_Verification_Protocol
DVP_MODEL_DIR = os.getenv(
    "DVP_MODEL_DIR",
    "/models/mistral_finetuned_Design_Verification_Protocol"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---------------- Globals ----------------
_tokenizer: AutoTokenizer | None = None
_model: AutoModelForCausalLM | None = None
_emb_model: SentenceTransformer | None = None
_faiss_index = None
_faiss_texts: List[str] = []

# ---------------- Heuristics & Lookups ----------------
USABILITY_KWS = ["usability", "user", "human factors", "ui", "interface"]
VISUAL_KWS    = ["label", "marking", "display", "visual", "color"]
TECH_KWS      = ["electrical", "mechanical", "flow", "pressure", "occlusion", "accuracy", "alarm"]

TEST_SPEC_LOOKUP = {
    "insulation": "≥ 50 MΩ at 500 V DC (IEC 60601-1)",
    "leakage": "≤ 100 µA at rated voltage (IEC 60601-1)",
    "dielectric": "1500 V AC for 1 min, no breakdown (IEC 60601-1)",
    "flow": "±5% from set value across 0.1–999 ml/hr (IEC 60601-2-24)",
    "occlusion": "Alarm ≤ 30 s at 100 kPa back pressure (IEC 60601-2-24)",
    "luer": "No leakage under 300 kPa for 30 s (ISO 80369-7)",
    "emc": "±6 kV contact, ±8 kV air (IEC 61000-4-2)",
    "vibration": "10–500 Hz, 0.5 g, 2 h/axis (IEC 60068-2-6)",
}

SEVERITY_TO_N = {5: 50, 4: 40, 3: 30, 2: 20, 1: 10}

# ---------------- Utils ----------------
def _get_verification_method(req_text: str) -> str:
    t = (req_text or "").lower()
    if any(k in t for k in USABILITY_KWS):
        return "NA"
    if any(k in t for k in VISUAL_KWS):
        return "Visual Inspection"
    if any(k in t for k in TECH_KWS):
        return "Physical Testing"
    return "Physical Inspection"

def _get_sample_size(requirement_id: str, ha_items: List[Dict[str, Any]]) -> str:
    """Prefer severity from HA (same Requirement ID); fallback heuristic by digits in ID."""
    sev = None
    for h in ha_items or []:
        if str(h.get("requirement_id") or h.get("Requirement ID")) == str(requirement_id):
            s = h.get("severity_of_harm") or h.get("Severity of Harm")
            try:
                s_int = int(str(s))
                sev = s_int if (sev is None or s_int > sev) else sev
            except Exception:
                continue
    if sev is not None:
        return str(SEVERITY_TO_N.get(int(sev), 30))
    try:
        digit = int(str(requirement_id).split("-")[-1]) % 5
        return str(20 + digit * 5)
    except Exception:
        return "30"

# ---------------- Model loader ----------------
def _load_model():
    global _tokenizer, _model, _emb_model
    if _model is not None:
        return

    if not os.path.isdir(DVP_MODEL_DIR):
        raise RuntimeError(f"DVP model dir not found: {DVP_MODEL_DIR}")

    _tokenizer = AutoTokenizer.from_pretrained(DVP_MODEL_DIR)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # ✅ CPU/GPU-safe: explicit dtype + move to device; avoid device_map="auto"
    _model = AutoModelForCausalLM.from_pretrained(DVP_MODEL_DIR, torch_dtype=DTYPE)
    _model.to(DEVICE)

    # Embedding model is optional; don't fail the service if unavailable
    try:
        _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _emb_model = None

# ---------------- Generation helpers ----------------
def _gen_test_procedure(requirement_text: str) -> str:
    """Use the fine-tuned DVP model to produce 3–4 measurable bullets."""
    _load_model()
    prompt = f"""You are a compliance engineer.

Generate ONLY a Design Verification Test Procedure for the following requirement.

Requirement: {requirement_text}

- Limit output strictly to this requirement.
- Do NOT include unrelated tests.
- Output exactly 3–4 bullet points.
- Each bullet must include measurable values (units, thresholds, counts).
"""
    inputs = _tokenizer(prompt, return_tensors="pt").to(DEVICE)  # type: ignore[arg-type]
    with torch.no_grad():
        outputs = _model.generate(  # type: ignore[union-attr]
            **inputs,
            max_new_tokens=320,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
    decoded = _tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore[union-attr]

    bullets: List[str] = []
    for line in decoded.split("\n"):
        s = line.strip(" -•\t")
        if s and len(s.split()) > 3:
            bullets.append(f"- {s}")
        if len(bullets) == 4:
            break
    return "\n".join(bullets) if bullets else "TBD"

def _hybrid_enrich(requirement_text: str) -> Dict[str, str]:
    """Combine model bullets with standards snippets (simple keyword-based assistance)."""
    bullets = _gen_test_procedure(requirement_text)
    std_hint = None
    lt = requirement_text.lower()
    for kw, spec in TEST_SPEC_LOOKUP.items():
        if kw in lt:
            std_hint = spec
            break
    ac = std_hint or "TBD"
    return {"Test Procedure": bullets, "Acceptance Criteria": ac}

# ---------------- Public API ----------------
def dvp_predict(requirements: List[Dict[str, Any]], ha: List[Dict[str, Any]]):
    """
    Return a list of DVP rows for each input requirement.
    Input `requirements` should be canonical dicts with keys:
    - Requirement ID, Verification ID, Requirements
    """
    _load_model()
    rows: List[Dict[str, Any]] = []

    for r in requirements:
        rid = str(r.get("Requirement ID", "") or "")
        vid = str(r.get("Verification ID", "") or "")
        rtxt = r.get("Requirements", "") or ""

        # Section headings like "Functional Requirements" → NA row
        if rtxt.strip().lower().endswith("requirements") and not vid:
            rows.append({
                "verification_id": vid,
                "requirement_id": rid,
                "requirements": rtxt,
                "verification_method": "NA",
                "sample_size": "NA",
                "test_procedure": "NA",
                "acceptance_criteria": "NA",
            })
            continue

        method  = _get_verification_method(rtxt)
        sample  = _get_sample_size(rid, ha or [])
        enriched = _hybrid_enrich(rtxt)

        rows.append({
            "verification_id": vid,
            "requirement_id": rid,
            "requirements": rtxt,
            "verification_method": method or "NA",
            "sample_size": sample or "NA",
            "test_procedure": enriched.get("Test Procedure", "TBD"),
            "acceptance_criteria": enriched.get("Acceptance Criteria", "TBD"),
        })

    return rows
