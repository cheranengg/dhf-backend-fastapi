from __future__ import annotations
import os, re, json, gc
from typing import List, Dict, Any, Optional

# ---- switches & locations ----
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
USE_HA_MODEL = os.getenv("USE_HA_MODEL", "0") == "1"
HA_MODEL_DIR = os.getenv("HA_MODEL_MERGED_DIR", os.getenv("HA_MODEL_DIR", "")) or "/models/ha-merged"
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
HF_TOKEN = os.getenv("HF_TOKEN", None)

from app.utils.cache_bootstrap import pick_hf_cache_dir
CACHE_DIR = pick_hf_cache_dir()
def _token_cache_kwargs():
    kw = {"cache_dir": CACHE_DIR}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw

# optional embeddings
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

if USE_HA_MODEL:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
else:
    torch = None  # type: ignore
    AutoTokenizer = AutoModelForCausalLM = None  # type: ignore

_tokenizer: Optional["AutoTokenizer"] = None
_model: Optional["AutoModelForCausalLM"] = None
_emb: Optional["SentenceTransformer"] = None
_DEVICE: str = "cpu"

_default_risks = [
    "Air Embolism","Allergic response","Infection","Overdose","Underdose",
    "Delay of therapy","Environmental Hazard","Incorrect Therapy","Trauma","Particulate",
]
_severity_map = {"negligible":1,"minor":2,"moderate":3,"serious":4,"critical":5}
_json_obj = re.compile(r"\{[\s\S]*?\}")

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text: return None
    m = _json_obj.findall(text)
    if not m: return None
    js = m[-1].replace("'", '"').replace("\\n", " ")
    js = re.sub(r"\s+", " ", js); js = re.sub(r",\s*\}", "}", js); js = re.sub(r",\s*\]", "]", js)
    try: return json.loads(js)
    except Exception: return None

def _calculate_risk_fields(parsed: Dict[str, Any]):
    sev_txt = str(parsed.get("Severity of Harm", "Moderate")).lower()
    severity = _severity_map.get(sev_txt, 3)
    p0 = parsed.get("P0", "Medium"); p1 = parsed.get("P1", "Medium")
    poh_matrix = {
        ("Very Low","Very Low"):"Very Low", ("Very Low","Low"):"Very Low",
        ("Very Low","Medium"):"Low", ("Low","Very Low"):"Very Low",
        ("Low","Low"):"Low", ("Low","Medium"):"Medium", ("Medium","Medium"):"Medium",
        ("Medium","High"):"High", ("High","Medium"):"High", ("High","High"):"High",
        ("Very High","High"):"Very High", ("Very High","Very High"):"Very High",
    }
    poh = poh_matrix.get((p0,p1), "Medium")
    if severity==5 and poh in ("High","Very High"): risk_index="Extreme"
    elif severity>=3 and poh in ("High","Very High"): risk_index="High"
    else: risk_index="Medium"
    return severity, p0, p1, poh, risk_index

def _load_tokenizer():
    from transformers import AutoTokenizer
    # Try the merged repo first (self-contained), then BASE_MODEL_ID (if set)
    sources = [HA_MODEL_DIR]
    if BASE_MODEL_ID:
        sources.append(BASE_MODEL_ID)

    last_err = None
    for src in sources:
        try:
            # try fast tokenizer first
            return AutoTokenizer.from_pretrained(src, use_fast=True, **_token_cache_kwargs())
        except Exception as e:
            last_err = e
            try:
                # fall back to non-fast (SentencePiece) tokenizer
                return AutoTokenizer.from_pretrained(src, use_fast=False, **_token_cache_kwargs())
            except Exception as e2:
                last_err = e2
                continue
    # If we get here, nothing worked.
    raise RuntimeError(f"Tokenizer load failed. Tried: {sources}. Last error: {last_err}")

def _load_model():
    global _tokenizer, _model, _emb, _DEVICE
    if not USE_HA_MODEL or _model is not None: return
    from transformers import AutoModelForCausalLM
    import torch

    _tokenizer = _load_tokenizer()
    if _tokenizer.pad_token is None: _tokenizer.pad_token = _tokenizer.eos_token

    want_cuda = torch.cuda.is_available() and (not FORCE_CPU)
    try:
        if want_cuda:
            _DEVICE = "cuda"
            _model = AutoModelForCausalLM.from_pretrained(
                HA_MODEL_DIR, torch_dtype=torch.float16, device_map="auto",
                low_cpu_mem_usage=True, **_token_cache_kwargs()
            )
        else:
            raise RuntimeError("CPU path")
    except Exception:
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
        gc.collect()
        _DEVICE = "cpu"
        _model = AutoModelForCausalLM.from_pretrained(
            HA_MODEL_DIR, torch_dtype=None, low_cpu_mem_usage=True, **_token_cache_kwargs()
        )
        _model.to("cpu")

    if _HAS_EMB:
        try: _emb = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=CACHE_DIR)  # type: ignore
        except Exception: _emb = None

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
    _load_model()
    import torch
    inputs = _tokenizer(_prompt_template.format(risk=risk), return_tensors="pt").to(_DEVICE)  # keep same device
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True, top_p=0.9)
    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)
    return _extract_json(decoded) or {}

def _nearest_req_control(reqs: List[Dict[str, Any]], hint_text: str) -> str:
    if not _HAS_EMB or not _emb: return "Refer to IEC 60601 and ISO 14971 risk controls"
    try:
        corpus = [str(r.get("Requirements") or "") for r in reqs]
        ids    = [str(r.get("Requirement ID") or "") for r in reqs]
        if not corpus: return "Refer to IEC 60601 and ISO 14971 risk controls"
        vecs = _emb.encode(corpus, convert_to_numpy=True)  # type: ignore
        index = faiss.IndexFlatL2(vecs.shape[1]); index.add(vecs)
        q = _emb.encode([hint_text or "risk control"], convert_to_numpy=True)  # type: ignore
        D, I = index.search(q, 1); i = int(I[0][0])
        return f"{corpus[i]} (Ref: {ids[i]})" if 0 <= i < len(corpus) else "Refer to IEC 60601 and ISO 14971 risk controls"
    except Exception:
        return "Refer to IEC 60601 and ISO 14971 risk controls"

def _fallback_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in requirements:
        rid = r.get("Requirement ID") or ""
        for risk in _default_risks:
            rows.append({
                "requirement_id": rid, "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
                "risk_to_health": risk, "hazard": "Not available", "hazardous_situation": "Not available",
                "harm": "Not available", "sequence_of_events": "Not available",
                "severity_of_harm": "3", "p0": "Medium", "p1": "Medium", "poh": "Medium", "risk_index": "Medium",
                "risk_control": f"Refer to IEC 60601 / ISO 14971 (nearest req: {rid})" if rid else "Refer to IEC 60601 / ISO 14971",
            })
    return rows

def ha_predict(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not USE_HA_MODEL: return _fallback_ha(requirements)
    out: List[Dict[str, Any]] = []
    for r in requirements:
        rid = r.get("Requirement ID") or ""; rtext = r.get("Requirements") or ""
        for risk in _default_risks:
            try: parsed = _gen_json_for_risk(risk)
            except Exception: parsed = {}
            severity, p0, p1, poh, risk_index = _calculate_risk_fields(parsed)
            hint = (parsed.get("Hazardous Situation","")+" "+parsed.get("Harm","")).strip() or rtext
            control = _nearest_req_control(requirements, hint)
            out.append({
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
    return out
