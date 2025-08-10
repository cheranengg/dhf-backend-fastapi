# app/models/ha_infer.py
from __future__ import annotations

import os, re, json
from typing import List, Dict, Any, Optional

# ---------------- Fallback (stub) ----------------
def _fallback_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    default_risks = [
        "Air Embolism","Allergic response","Infection","Overdose","Underdose",
        "Delay of therapy","Environmental Hazard","Incorrect Therapy","Trauma","Particulate",
    ]
    for r in requirements:
        rid = r.get("Requirement ID") or ""
        for risk in default_risks:
            rows.append({
                "requirement_id": rid,
                "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
                "risk_to_health": risk,
                "hazard": "Not available",
                "hazardous_situation": "Not available",
                "harm": "Not available",
                "sequence_of_events": "Not available",
                "severity_of_harm": 3,
                "p0": "Medium",
                "p1": "Medium",
                "poh": "Medium",
                "risk_index": "Medium",
                "risk_control": f"Refer to IEC 60601 / ISO 14971 (nearest req: {rid})" if rid else "Refer to IEC 60601 / ISO 14971",
            })
    return rows

# ---- Toggle heavy model usage (safe default off for Cloud Run CPU) ----
USE_HA_MODEL = os.getenv("USE_HA_MODEL", "0") == "1"

# Optional similarity for risk-control hinting (safe if missing)
try:
    from sentence_transformers import SentenceTransformer
    import faiss  # type: ignore
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

# ---------------- Config ----------------
HA_MODEL_MERGED_DIR = os.getenv("HA_MODEL_MERGED_DIR", "/models/mistral_finetuned_Hazard_Analysis_MERGED")
BASE_MODEL_ID       = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
LORA_HA_DIR         = os.getenv("LORA_HA_DIR", "/models/mistral_finetuned_Hazard_Analysis")

# ---------------- Lazy heavy imports / globals ----------------
if USE_HA_MODEL:
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    from peft import PeftModel  # type: ignore
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
else:
    AutoTokenizer = AutoModelForCausalLM = PeftModel = None  # type: ignore
    torch = None  # type: ignore
    DEVICE = DTYPE = None  # type: ignore

_tokenizer: Optional["AutoTokenizer"] = None
_model: Optional["AutoModelForCausalLM"] = None
_emb: Optional["SentenceTransformer"] = None

# ---------------- Guardrails helpers ----------------
_severity_map = {"negligible": 1, "minor": 2, "moderate": 3, "serious": 4, "critical": 5}

def _calculate_risk_fields(parsed: Dict[str, Any]):
    sev_txt = str(parsed.get("Severity of Harm", "Moderate")).lower()
    severity = _severity_map.get(sev_txt, 3)
    p0 = parsed.get("P0", "Medium")
    p1 = parsed.get("P1", "Medium")
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

_json_obj = re.compile(r"\{[\s\S]*?\}")

def _extract_json(text: str) -> Dict[str, Any] | None:
    m = _json_obj.findall(text or "")
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

# ---------------- Loader ----------------
def _load_model():
    """Load merged fine-tuned model, else base+LoRA. No-op when USE_HA_MODEL=0."""
    global _tokenizer, _model, _emb
    if not USE_HA_MODEL or _model is not None:
        return

    if os.path.isdir(HA_MODEL_MERGED_DIR):
        _tokenizer = AutoTokenizer.from_pretrained(HA_MODEL_MERGED_DIR)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(HA_MODEL_MERGED_DIR, torch_dtype=DTYPE)
    else:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE)
        _model = PeftModel.from_pretrained(base, LORA_HA_DIR)

    _model.to(DEVICE)

    if _HAS_EMB:
        try:
            _emb = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _emb = None

# ---------------- Inference ----------------
_default_risks = [
    "Air Embolism", "Allergic response", "Infection", "Overdose", "Underdose",
    "Delay of therapy", "Environmental Hazard", "Incorrect Therapy", "Trauma", "Particulate",
]

_prompt_template = """Return ONLY valid JSON for the following risk in an infusion pump.

Risk: {risk}

JSON fields:
{
  "Hazard": "...",
  "Hazardous Situation": "...",
  "Harm": "...",
  "Sequence of Events": "...",
  "Severity of Harm": "Negligible|Minor|Moderate|Serious|Critical",
  "P0": "Very Low|Low|Medium|High|Very High",
  "P1": "Very Low|Low|Medium|High|Very High"
}
"""

def _gen_json_for_risk(risk: str) -> Dict[str, Any]:
    _load_model()  # ensures tokenizer/model exist
    import torch  # local to avoid import when stub mode
    inputs = _tokenizer(_prompt_template.format(risk=risk), return_tensors="pt").to(DEVICE)  # type: ignore
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

def _nearest_req_control(reqs: List[Dict[str, Any]], hint_text: str) -> str:
    if not _HAS_EMB:
        return "Refer to IEC 60601 and ISO 14971 risk controls"
    try:
        corpus = [str(r.get("Requirements") or "") for r in reqs]
        ids    = [str(r.get("Requirement ID") or "") for r in reqs]
        if not corpus:
            return "Refer to IEC 60601 and ISO 14971 risk controls"
        vecs = _emb.encode(corpus, convert_to_numpy=True)  # type: ignore
        index = faiss.IndexFlatL2(vecs.shape[1])
        index.add(vecs)
        q = _emb.encode([hint_text or "risk control"], convert_to_numpy=True)  # type: ignore
        D, I = index.search(q, 1)
        i = int(I[0][0])
        return f"{corpus[i]} (Ref: {ids[i]})" if 0 <= i < len(corpus) else "Refer to IEC 60601 and ISO 14971 risk controls"
    except Exception:
        return "Refer to IEC 60601 and ISO 14971 risk controls"

def ha_predict(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Fast stub path for Cloud Run CPU
    if not USE_HA_MODEL:
        return _fallback_ha(requirements)

    # Real model path
    _load_model()
    out: List[Dict[str, Any]] = []
    for r in requirements:
        rid   = r.get("Requirement ID") or ""
        rtext = r.get("Requirements") or ""
        for risk in _default_risks:
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
                "severity_of_harm": severity,
                "p0": p0,
                "p1": p1,
                "poh": poh,
                "risk_index": risk_index,
                "risk_control": control,
            })
    return out
