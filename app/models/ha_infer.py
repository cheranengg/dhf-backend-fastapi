# app/models/ha_infer.py
from __future__ import annotations

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

# ------------------------------- config flags -------------------------------

# Toggle model usage via env
USE_HA_MODEL = os.getenv("USE_HA_MODEL", "0") == "1"

# Directories / IDs
HA_MODEL_MERGED_DIR = os.getenv("HA_MODEL_MERGED_DIR", "/models/mistral_finetuned_Hazard_Analysis_MERGED")
BASE_MODEL_ID       = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
LORA_HA_DIR         = os.getenv("LORA_HA_DIR", "/models/mistral_finetuned_Hazard_Analysis")

# Hugging Face cache and token (these are set in main.py bootstrap; we reuse)
HF_CACHE_DIR = (
    os.environ.get("HF_HOME")
    or os.environ.get("HF_HUB_CACHE")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or os.environ.get("TRANSFORMERS_CACHE")
)

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
TOKEN_KW = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}
CACHE_KW = {"cache_dir": HF_CACHE_DIR} if HF_CACHE_DIR else {}

# -------------------------- optional embedding layer ------------------------

try:
    from sentence_transformers import SentenceTransformer
    import faiss  # type: ignore
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

# ------------------------------ default risks --------------------------------

_DEFAULT_RISKS = [
    "Air Embolism",
    "Allergic response",
    "Infection",
    "Overdose",
    "Underdose",
    "Delay of therapy",
    "Environmental Hazard",
    "Incorrect Therapy",
    "Trauma",
    "Particulate",
]

# -------------------------- model objects (lazy load) ------------------------

_tokenizer = None        # type: ignore
_model     = None        # type: ignore
_emb       = None        # type: ignore

if USE_HA_MODEL:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
else:
    # keep names defined for type hints
    AutoTokenizer = AutoModelForCausalLM = PeftModel = None  # type: ignore
    torch = None  # type: ignore
    DEVICE = DTYPE = None  # type: ignore

# ------------------------------- utilities -----------------------------------

# Map free-text severity to numeric and compute derived fields (PoH, Risk Index)
_SEVERITY_MAP = {"negligible": 1, "minor": 2, "moderate": 3, "serious": 4, "critical": 5}

def _calculate_risk_fields(parsed: Dict[str, Any]) -> Tuple[int, str, str, str, str]:
    sev_txt = str(parsed.get("Severity of Harm", "Moderate")).lower().strip()
    severity = _SEVERITY_MAP.get(sev_txt, 3)

    p0 = str(parsed.get("P0", "Medium")).title()
    p1 = str(parsed.get("P1", "Medium")).title()

    poh_matrix = {
        ("Very Low","Very Low"):"Very Low", ("Very Low","Low"):"Very Low",
        ("Very Low","Medium"):"Low", ("Low","Very Low"):"Very Low",
        ("Low","Low"):"Low", ("Low","Medium"):"Medium", ("Medium","Medium"):"Medium",
        ("Medium","High"):"High", ("High","Medium"):"High", ("High","High"):"High",
        ("Very High","High"):"Very High", ("Very High","Very High"):"Very High",
    }
    poh = poh_matrix.get((p0, p1), "Medium")

    if severity == 5 and poh in ("High", "Very High"):
        risk_index = "Extreme"
    elif severity >= 3 and poh in ("High", "Very High"):
        risk_index = "High"
    else:
        risk_index = "Medium"

    return severity, p0, p1, poh, risk_index

_JSON_OBJ = re.compile(r"\{[\s\S]*?\}")

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_OBJ.findall(text)
    if not m:
        return None
    js = m[-1]
    js = js.replace("'", '"').replace("\\n", " ")
    js = re.sub(r"\s+", " ", js)
    js = re.sub(r",\s*\}", "}", js)
    js = re.sub(r",\s*\]", "]", js)
    try:
        return json.loads(js)
    except Exception:
        return None

# ------------------------------- model loading --------------------------------

def _load_model() -> None:
    """Lazy load HA generator (merged or base+LoRA) and optional embedder."""
    global _tokenizer, _model, _emb
    if not USE_HA_MODEL or _model is not None:
        return

    # Prefer merged fine-tune if present, else base+LoRA
    if os.path.isdir(HA_MODEL_MERGED_DIR):
        _tokenizer = AutoTokenizer.from_pretrained(HA_MODEL_MERGED_DIR, **TOKEN_KW, **CACHE_KW)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(
            HA_MODEL_MERGED_DIR, torch_dtype=DTYPE, **TOKEN_KW, **CACHE_KW
        )
    else:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **TOKEN_KW, **CACHE_KW)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, **TOKEN_KW, **CACHE_KW)
        _model = PeftModel.from_pretrained(base, LORA_HA_DIR, **TOKEN_KW, **CACHE_KW)

    _model.to(DEVICE)  # type: ignore

    # Optional embedder for nearest requirement hint
    if _HAS_EMB:
        try:
            _emb = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=HF_CACHE_DIR or None)
        except Exception:
            _emb = None

# ------------------------------ generation prompt -----------------------------

_PROMPT = """Return ONLY valid JSON for the following risk in an infusion pump.

Risk: {risk}

JSON fields:
{{
  "Hazard": "...",
  "Hazardous Situation": "...",
  "Harm": "...",
  "Sequence of Events": "...",
  "Severity of Harm": "Negligible|Minor|Moderate|Serious|Critical",
  "P0": "Very Low|Low|Medium|High|Very High",
  "P1": "Very Low|Low|Medium|High|Very High"
}}
"""

def _gen_json_for_risk(risk: str) -> Dict[str, Any]:
    """Generate a single risk JSON object using the loaded LLM (when enabled)."""
    _load_model()
    import torch  # local import to avoid when USE_HA_MODEL=0

    inputs = _tokenizer(_PROMPT.format(risk=risk), return_tensors="pt").to(DEVICE)  # type: ignore
    with torch.no_grad():
        out = _model.generate(  # type: ignore
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore
    return _extract_json(decoded) or {}

# -------------------------- nearest requirement hint --------------------------

def _nearest_req_control(requirements: List[Dict[str, Any]], hint_text: str) -> str:
    """
    Use lightweight embeddings to pick the most relevant requirement as a 'risk control' hint.
    Falls back to a standard phrase if embedding stack is unavailable.
    """
    if not _HAS_EMB or _emb is None:
        return "Refer to IEC 60601 and ISO 14971 risk controls"
    try:
        corpus = [str(r.get("Requirements") or "") for r in requirements]
        ids    = [str(r.get("Requirement ID") or "") for r in requirements]
        if not corpus:
            return "Refer to IEC 60601 and ISO 14971 risk controls"

        vecs = _emb.encode(corpus, convert_to_numpy=True)  # type: ignore
        index = faiss.IndexFlatL2(vecs.shape[1])
        index.add(vecs)

        q = _emb.encode([hint_text or "risk control"], convert_to_numpy=True)  # type: ignore
        _, I = index.search(q, 1)
        i = int(I[0][0])
        if 0 <= i < len(corpus) and corpus[i].strip():
            rid = ids[i] if ids[i] else "N/A"
            return f"{corpus[i]} (Ref: {rid})"
        return "Refer to IEC 60601 and ISO 14971 risk controls"
    except Exception:
        return "Refer to IEC 60601 and ISO 14971 risk controls"

# --------------------------------- fallback -----------------------------------

def _fallback_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic, standards-friendly fallback when USE_HA_MODEL=0."""
    rows: List[Dict[str, Any]] = []
    for r in requirements:
        rid = str(r.get("Requirement ID") or "")
        for risk in _DEFAULT_RISKS:
            rows.append({
                "requirement_id": rid,
                "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
                "risk_to_health": risk,
                "hazard": "Not available",
                "hazardous_situation": "Not available",
                "harm": "Not available",
                "sequence_of_events": "Not available",
                "severity_of_harm": "3",
                "p0": "Medium",
                "p1": "Medium",
                "poh": "Medium",
                "risk_index": "Medium",
                "risk_control": f"Refer to IEC 60601 / ISO 14971 (nearest req: {rid})" if rid else "Refer to IEC 60601 / ISO 14971",
            })
    return rows

# ---------------------------------- API ---------------------------------------

def ha_predict(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Produce Hazard Analysis rows for each requirement and each default risk.
    If USE_HA_MODEL=1, generate fields with the LLM; otherwise use deterministic fallback.
    """
    if not USE_HA_MODEL:
        return _fallback_ha(requirements)

    _load_model()

    out: List[Dict[str, Any]] = []
    for r in requirements:
        rid   = str(r.get("Requirement ID") or "")
        rtext = str(r.get("Requirements") or "")

        for risk in _DEFAULT_RISKS:
            try:
                parsed = _gen_json_for_risk(risk)
            except Exception:
                parsed = {}

            severity, p0, p1, poh, risk_index = _calculate_risk_fields(parsed)
            hint = (parsed.get("Hazardous Situation", "") + " " + parsed.get("Harm", "")).strip() or rtext
            control = _nearest_req_control(requirements, hint)

            out.append({
                "requirement_id": rid,
                "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
                "risk_to_health": risk,
                "hazard": parsed.get("Hazard", "Not available"),
                "hazardous_situation": parsed.get("Hazardous Situation", "Not available"),
                "harm": parsed.get("Harm", "Not available"),
                "sequence_of_events": parsed.get("Sequence of Events", "Not available"),
                "severity_of_harm": str(severity),
                "p0": p0,
                "p1": p1,
                "poh": poh,
                "risk_index": risk_index,
                "risk_control": control,
            })

    return out
