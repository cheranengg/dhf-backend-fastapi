from __future__ import annotations
import os, re, json, sys
from typing import List, Dict, Any, Optional

# ----------------- switches & sources -----------------
USE_HA_MODEL = os.getenv("USE_HA_MODEL", "0") == "1"

# We only use the merged repo or local dir the user provides
HA_MODEL_DIR = os.getenv("HA_MODEL_MERGED_DIR", "").strip()
if not HA_MODEL_DIR:
    # allow legacy name "HA_MODEL_DIR"
    HA_MODEL_DIR = os.getenv("HA_MODEL_DIR", "").strip()

# HF auth/cache
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
CACHE_DIR = (
    os.getenv("HF_HOME")
    or os.getenv("HF_HUB_CACHE")
    or os.getenv("HUGGINGFACE_HUB_CACHE")
    or os.getenv("TRANSFORMERS_CACHE")
    or "/tmp/hf"
)

# Optional: reduce CPU threads if needed
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "8"))

def _token_cache_kwargs():
    kw = {"cache_dir": CACHE_DIR, "local_files_only": False}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    # hub can occasionally wobble; retries help first pull
    kw["resume_download"] = True
    return kw

# ----------------- optional embeddings for nearest req -----------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

# ----------------- guard imports behind switch -----------------
_tokenizer = None
_model = None
_emb = None

_default_risks = [
    "Air Embolism","Allergic response","Infection","Overdose","Underdose",
    "Delay of therapy","Environmental Hazard","Incorrect Therapy","Trauma","Particulate",
]

_severity_map = {"negligible": 1, "minor": 2, "moderate": 3, "serious": 4, "critical": 5}

# robust JSON grabber (last object)
_json_obj = re.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _json_obj.findall(text)
    if not m:
        return None
    js = m[-1].replace("'", '"').replace("\\n", " ")
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
def _load_model():
    """
    Load the merged fine-tuned model from HA_MODEL_MERGED_DIR only.
    If anything fails we raise — no silent fallback to placeholders.
    """
    global _tokenizer, _model, _emb
    if not USE_HA_MODEL:
        return
    if _model is not None:
        return

    if not HA_MODEL_DIR:
        raise RuntimeError("HA_MODEL_MERGED_DIR is not set.")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        # L4 supports bfloat16; T4 prefers float16; CPU = float32
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        _tokenizer = AutoTokenizer.from_pretrained(
            HA_MODEL_DIR, use_fast=True, **_token_cache_kwargs()
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForCausalLM.from_pretrained(
            HA_MODEL_DIR, torch_dtype=dtype, **_token_cache_kwargs()
        )
        _model.to("cuda" if torch.cuda.is_available() else "cpu")

        if _HAS_EMB:
            try:
                _emb = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=CACHE_DIR)  # type: ignore
            except Exception:
                _emb = None
    except Exception as e:
        raise RuntimeError(f"HA model load failed: {e}")

# ----------------- prompting -----------------
_PROMPT = (
    "You are generating a Hazard Analysis row for an infusion pump.\n"
    "Return ONLY a single JSON object and no extra text.\n"
    "Fields:\n"
    "{\n"
    '  "Hazard": "...",\n'
    '  "Hazardous Situation": "...",\n'
    '  "Harm": "...",\n'
    '  "Sequence of Events": "...",\n'
    '  "Severity of Harm": "Negligible|Minor|Moderate|Serious|Critical",\n'
    '  "P0": "Very Low|Low|Medium|High|Very High",\n'
    '  "P1": "Very Low|Low|Medium|High|Very High"\n'
    "}\n"
    "Risk to Health: {risk}\n"
)

def _gen_json_for_risk(risk: str) -> Dict[str, Any]:
    _load_model()
    from transformers import StoppingCriteria, StoppingCriteriaList
    import torch

    # deterministic decoding to improve JSON rate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = _PROMPT.format(risk=risk)
    inputs = _tokenizer(prompt, return_tensors="pt").to(device)  # type: ignore

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,        # << deterministic
            temperature=0.0,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
        )  # type: ignore
    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore
    parsed = _extract_json(decoded)
    if not parsed:
        # Let the caller see we failed rather than silently returning placeholders
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
    # used ONLY when USE_HA_MODEL=0
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
                # Capture error and surface on the API by raising after the loop
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

    # If everything failed on first warm-up, raise to show the cause instead of silent TBDs
    if all(row["hazard"] == "TBD" for row in rows) and last_exc is not None:
        raise RuntimeError(str(last_exc))

    return rows
