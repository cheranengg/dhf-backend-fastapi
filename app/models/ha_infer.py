from __future__ import annotations
import os, re, json
from typing import List, Dict, Any, Optional

# ================== Switches & locations ==================
USE_HA_MODEL = os.getenv("USE_HA_MODEL", "0") == "1"
# Use your merged model repo or on-disk path
HA_MODEL_DIR = os.getenv("HA_MODEL_MERGED_DIR", os.getenv("HA_MODEL_DIR", "")) or "/models/ha-merged"

# Writable cache with safe default
CACHE_DIR = (
    os.getenv("HF_HOME")
    or os.getenv("TRANSFORMERS_CACHE")
    or "/tmp/hf"
)
os.makedirs(CACHE_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN", None)
def _token_cache_kwargs():
    kw = {"cache_dir": CACHE_DIR, "trust_remote_code": True}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw

# ================== Optional embeddings for retrieval ==================
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

# ================== Torch / HF imports (guarded) ==================
if USE_HA_MODEL:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
else:
    torch = None  # type: ignore
    AutoTokenizer = AutoModelForCausalLM = None  # type: ignore

# ================== Module globals ==================
_tokenizer: Optional["AutoTokenizer"] = None
_model: Optional["AutoModelForCausalLM"] = None
_emb: Optional["SentenceTransformer"] = None

# Risks you used in Colab
_DEFAULT_RISKS = [
    "Air Embolism","Allergic response","Infection","Overdose","Underdose",
    "Delay of therapy","Environmental Hazard","Incorrect Therapy","Trauma","Particulate",
]

# (Light) Guardrails
_SEVERITY_MAP = {"negligible":1, "minor":2, "moderate":3, "serious":4, "critical":5}

# JSON extraction (take the LAST JSON block)
_JSON_RE = re.compile(r"\{[\s\S]*?\}")

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_RE.findall(text)
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

def _calculate_risk_fields(parsed: Dict[str, Any]):
    """Your ISO 14971 guardrail logic."""
    sev_txt = str(parsed.get("Severity of Harm", "Moderate")).strip().lower()
    severity = _SEVERITY_MAP.get(sev_txt, 3)
    p0 = parsed.get("P0", "Medium")
    p1 = parsed.get("P1", "Medium")
    # quick PoH matrix
    poh_table = {
        ("Very Low","Very Low"):"Very Low", ("Very Low","Low"):"Very Low",
        ("Very Low","Medium"):"Low", ("Low","Very Low"):"Very Low",
        ("Low","Low"):"Low", ("Low","Medium"):"Medium", ("Medium","Medium"):"Medium",
        ("Medium","High"):"High", ("High","Medium"):"High", ("High","High"):"High",
        ("Very High","High"):"Very High", ("Very High","Very High"):"Very High",
    }
    poh = poh_table.get((p0, p1), "Medium")
    if severity == 5 and poh in ("High", "Very High"):
        risk_index = "Extreme"
    elif severity >= 3 and poh in ("High", "Very High"):
        risk_index = "High"
    else:
        risk_index = "Medium"
    return severity, p0, p1, poh, risk_index

# ================== Model loading ==================
def _load_model():
    """Load merged model + tokenizer only from HA_MODEL_DIR (no base)."""
    global _tokenizer, _model, _emb
    if not USE_HA_MODEL or _model is not None:
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    src = HA_MODEL_DIR  # merged repo; includes tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(src, **_token_cache_kwargs())  # type: ignore
    if getattr(_tokenizer, "pad_token", None) is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, **_token_cache_kwargs())  # type: ignore

    # Safety: ensure embedding size matches tokenizer
    if _model.get_input_embeddings().weight.size(0) != len(_tokenizer):
        _model.resize_token_embeddings(len(_tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device)

    # light retrieval model (nearest requirement)
    if _HAS_EMB:
        try:
            _emb = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=CACHE_DIR)  # type: ignore
        except Exception:
            _emb = None

# ================== Prompt ==================
_PROMPT = """You are performing Hazard Analysis for an infusion pump.
Return ONLY a single valid JSON object with these keys:
{{
  "Hazard": "...",
  "Hazardous Situation": "...",
  "Harm": "...",
  "Sequence of Events": "...",
  "Severity of Harm": "Negligible|Minor|Moderate|Serious|Critical",
  "P0": "Very Low|Low|Medium|High|Very High",
  "P1": "Very Low|Low|Medium|High|Very High"
}}

Risk to Health: {risk}
Make outputs concise and specific to infusion pump context.
"""

def _generate_for_risk(risk: str) -> Dict[str, Any]:
    """LLM call -> JSON parse; mirrors your Colab flow minus LangChain plumbing."""
    _load_model()
    import torch
    device = next(_model.parameters()).device  # type: ignore

    prompt = _PROMPT.format(risk=risk)
    inputs = _tokenizer(prompt, return_tensors="pt").to(device)  # type: ignore
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=320,
            temperature=0.3,
            do_sample=True,
            top_p=0.9
        )  # type: ignore
    text = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore
    return _extract_json(text) or {}

# ================== Retrieval for Risk Control ==================
def _nearest_requirement_control(reqs: List[Dict[str, Any]], hint: str) -> str:
    """
    Use FAISS over the provided product requirements (Requirements text) to pick
    the best risk control, mirroring your FAISS usage.
    """
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
        q = _emb.encode([hint or "risk control"], convert_to_numpy=True)  # type: ignore
        D, I = index.search(q, 1)
        i = int(I[0][0])
        return f"{corpus[i]} (Ref: {ids[i]})" if 0 <= i < len(corpus) else "Refer to IEC 60601 and ISO 14971 risk controls"
    except Exception:
        return "Refer to IEC 60601 and ISO 14971 risk controls"

# ================== Public API ==================
def _fallback_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Used when USE_HA_MODEL=0; produces structured rows with sane defaults."""
    rows = []
    for r in requirements:
        rid = r.get("Requirement ID") or ""
        for risk in _DEFAULT_RISKS:
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
                "risk_control": f"Refer to IEC 60601 / ISO 14971 (nearest req: {rid})" if rid else "Refer to IEC 60601 / ISO 14971",
            })
    return rows

def ha_predict(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Replicates your Colab HA:
    - Generate JSON per Risk to Health using your merged model
    - Apply guardrails to compute Severity/P0/P1/PoH/Risk Index
    - Retrieve nearest requirement as Risk Control (FAISS)
    """
    if not USE_HA_MODEL:
        return _fallback_ha(requirements)

    out: List[Dict[str, Any]] = []
    for r in requirements:
        rid   = str(r.get("Requirement ID") or "")
        rtext = str(r.get("Requirements") or "")

        for risk in _DEFAULT_RISKS:
            try:
                parsed = _generate_for_risk(risk)
            except Exception:
                parsed = {}

            # Guardrail calc
            severity, p0, p1, poh, risk_index = _calculate_risk_fields(parsed)

            # Retrieval for risk control (requirements vector store)
            hint = (parsed.get("Hazardous Situation", "") + " " + parsed.get("Harm", "")).strip() or rtext
            control = _nearest_requirement_control(requirements, hint)

            out.append({
                "requirement_id": rid,
                "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
                "risk_to_health": risk,
                "hazard": parsed.get("Hazard", "TBD"),
                "hazardous_situation": parsed.get("Hazardous Situation", "TBD"),
                "harm": parsed.get("Harm", "TBD"),
                "sequence_of_events": parsed.get("Sequence of Events", "TBD"),
                "severity_of_harm": str(severity),
                "p0": p0,
                "p1": p1,
                "poh": poh,
                "risk_index": risk_index,
                "risk_control": control,
            })

    return out

def debug_status():
    return {
        "use_model": USE_HA_MODEL,
        "src": HA_MODEL_DIR,
        "cache": CACHE_DIR,
        "loaded": _model is not None,
        "device": (str(next(_model.parameters()).device) if _model is not None else None)
    }
