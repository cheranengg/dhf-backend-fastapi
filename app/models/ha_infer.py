from __future__ import annotations
import os, re, json
from typing import List, Dict, Any, Optional

# ----------------- switches & sources -----------------
USE_HA_MODEL = os.getenv("USE_HA_MODEL", "0") == "1"

HA_MODEL_DIR = os.getenv("HA_MODEL_MERGED_DIR", "").strip() or os.getenv("HA_MODEL_DIR", "").strip()

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
CACHE_DIR = (
    os.getenv("HF_HOME")
    or os.getenv("HF_HUB_CACHE")
    or os.getenv("HUGGINGFACE_HUB_CACHE")
    or os.getenv("TRANSFORMERS_CACHE")
    or "/tmp/hf"
)

os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "8"))

def _token_cache_kwargs():
    kw = {"cache_dir": CACHE_DIR, "local_files_only": False, "resume_download": True}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw

# ----------------- optional embeddings -----------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

_tokenizer = None
_model = None
_emb = None

_default_risks = [
    "Air Embolism","Allergic response","Infection","Overdose","Underdose",
    "Delay of therapy","Environmental Hazard","Incorrect Therapy","Trauma","Particulate",
]
_severity_map = {"negligible": 1, "minor": 2, "moderate": 3, "serious": 4, "critical": 5}

# ----------------- JSON extraction (brace balancer) -----------------
def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Return the last balanced {...} object in `text`, cleaned and parsed.
    Works without recursive-regex features.
    """
    if not text:
        return None
    last_obj = None
    stack = []
    start_idx = None

    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start_idx = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx:i+1]
                    last_obj = candidate  # keep the last complete object

    if not last_obj:
        return None

    js = last_obj.replace("'", '"').replace("\\n", " ")
    js = re.sub(r"\s+", " ", js)
    js = re.sub(r",\s*\}", "}", js)
    js = re.sub(r",\s*\]", "]", js)
    try:
        return json.loads(js)
    except Exception:
        return None

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

# ----------------- model loading (merged-only) -----------------
def _load_tokenizer():
    """Load a safe (slow) tokenizer. Prefer merged repo; fall back to base."""
    from transformers import AutoTokenizer
    sources = []
    if HA_MODEL_DIR:
        sources.append(HA_MODEL_DIR)
    if BASE_MODEL_ID:
        sources.append(BASE_MODEL_ID)

    last_err = None
    for src in sources:
        try:
            # IMPORTANT: use_fast=False avoids the Rust tokenizer JSON parse
            tok = AutoTokenizer.from_pretrained(src, use_fast=False, **_token_cache_kwargs())
            if tok.pad_token is None and getattr(tok, "eos_token", None):
                tok.pad_token = tok.eos_token
            return tok
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Tokenizer load failed. Tried: {sources}. Last error: {last_err}")


def _load_model():
    """
    Load HA model lazily. Use tokenizer from merged repo if possible,
    otherwise from base. Model weights come from merged repo (or base if unset).
    """
    global _tokenizer, _model, _emb
    if not USE_HA_MODEL or _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM

    # device & dtype
    device = "cuda" if torch.cuda.is_available() and os.getenv("FORCE_CPU", "0") != "1" else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # 1) tokenizer (slow) with fallback
    _tokenizer = _load_tokenizer()  # safe loader above

    # 2) model weights (prefer merged)
    src = HA_MODEL_DIR or BASE_MODEL_ID
    if not src:
        raise RuntimeError("HA model load failed: no HA_MODEL_DIR or BASE_MODEL_ID set")

    _model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, **_token_cache_kwargs())
    _model.to(device)

    # optional: embeddings best-effort
    if _HAS_EMB:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _emb = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=CACHE_DIR)  # type: ignore
        except Exception:
            _emb = None

# ----------------- prompting -----------------
_PROMPT = """You are generating a Hazard Analysis row for an infusion pump.
Return ONLY a single JSON object and no extra text.
Fields:
{
  "Hazard": "...",
  "Hazardous Situation": "...",
  "Harm": "...",
  "Sequence of Events": "...",
  "Severity of Harm": "Negligible|Minor|Moderate|Serious|Critical",
  "P0": "Very Low|Low|Medium|High|Very High",
  "P1": "Very Low|Low|Medium|High|Very High"
}
Risk to Health: {risk}
"""

def _gen_json_for_risk(risk: str) -> Dict[str, Any]:
    _load_model()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = _PROMPT.format(risk=risk)
    inputs = _tokenizer(prompt, return_tensors="pt").to(device)  # type: ignore

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            temperature=0.0,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
        )  # type: ignore
    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore
    parsed = _extract_json(decoded)
    if not parsed:
        raise RuntimeError("Model did not return valid JSON")
    return parsed

def _nearest_req_control(reqs: List[Dict[str, Any]], hint_text: str) -> str:
    if not _HAS_EMB or not _emb:
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

# ----------------- public API -----------------
def _fallback_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in requirements:
        rid = r.get("Requirement ID") or ""
        for risk in _default_risks:
            rows.append({
                "requirement_id": rid,
                "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
                "risk_to_health": risk,
                "hazard": "TBD",
                "hazardous_situation": "TBD",
                "harm": "TBD",
                "sequence_of_events": "TBD",
                "severity_of_harm": "3",
                "p0": "Medium",
                "p1": "Medium",
                "poh": "Medium",
                "risk_index": "Medium",
                "risk_control": "Refer to IEC 60601 / ISO 14971",
            })
    return rows

def ha_predict(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not USE_HA_MODEL:
        return _fallback_ha(requirements)

    rows: List[Dict[str, Any]] = []
    last_exc: Optional[Exception] = None

    for r in requirements:
        rid   = r.get("Requirement ID") or ""
        rtext = r.get("Requirements") or ""
        for risk in _default_risks:
            try:
                parsed = _gen_json_for_risk(risk)
                severity, p0, p1, poh, risk_index = _calculate_risk_fields(parsed)
                hint = (parsed.get("Hazardous Situation", "") + " " + parsed.get("Harm", "")).strip() or rtext
                control = _nearest_req_control(requirements, hint)
                rows.append({
                    "requirement_id": rid,
                    "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
                    "risk_to_health": risk,
                    "hazard": parsed.get("Hazard", "TBD"),
                    "hazardous_situation": parsed.get("Hazardous Situation", "TBD"),
                    "harm": parsed.get("Harm", "TBD"),
                    "sequence_of_events": parsed.get("Sequence of Events", "TBD"),
                    "severity_of_harm": str(severity),
                    "p0": p0, "p1": p1, "poh": poh, "risk_index": risk_index,
                    "risk_control": control,
                })
            except Exception as e:
                last_exc = e
                rows.append({
                    "requirement_id": rid,
                    "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
                    "risk_to_health": risk,
                    "hazard": "TBD",
                    "hazardous_situation": "TBD",
                    "harm": "TBD",
                    "sequence_of_events": "TBD",
                    "severity_of_harm": "3",
                    "p0": "Medium",
                    "p1": "Medium",
                    "poh": "Medium",
                    "risk_index": "Medium",
                    "risk_control": "Refer to IEC 60601 / ISO 14971",
                })

    if all(row["hazard"] == "TBD" for row in rows) and last_exc is not None:
        raise RuntimeError(str(last_exc))

    return rows
