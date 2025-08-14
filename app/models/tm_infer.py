from __future__ import annotations

# ---------- HF cache bootstrap (robust + normalized) ----------
from app.utils.cache_bootstrap import pick_hf_cache_dir
CACHE_DIR = pick_hf_cache_dir()  # sets HF_* envs consistently and ensures writability

# ---------- FastAPI app ----------
import os, traceback
from typing import Any, Dict, List
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
# ... (rest of your imports stay the same)

import os, json, re
from typing import List, Dict, Any, Optional

USE_TM_MODEL = os.getenv("USE_TM_MODEL", "0") == "1"
TM_MODEL_DIR = os.getenv("TM_MODEL_DIR", "/models/tm-merged")

HF_TOKEN = os.getenv("HF_TOKEN", None)
CACHE_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/workspace/.cache/hf"
def _token_cache_kwargs():
    kw = {"cache_dir": CACHE_DIR}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw

if USE_TM_MODEL:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
else:
    torch = None  # type: ignore
    AutoTokenizer = AutoModelForCausalLM = None  # type: ignore

_tokenizer: Optional["AutoTokenizer"] = None
_model: Optional["AutoModelForCausalLM"] = None

_json_obj = re.compile(r"\{[\s\S]*?\}")

def _load_tm_model():
    global _tokenizer, _model
    if not USE_TM_MODEL or _model is not None: return
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    src = TM_MODEL_DIR
    _tokenizer = AutoTokenizer.from_pretrained(src, **_token_cache_kwargs())  # type: ignore
    if _tokenizer.pad_token is None: _tokenizer.pad_token = _tokenizer.eos_token
    _model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, **_token_cache_kwargs())  # type: ignore
    _model.to("cuda" if torch.cuda.is_available() else "cpu")

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text: return None
    m = _json_obj.findall(text)
    if not m: return None
    js = (m[-1].replace("'", '"').replace("\\n", " "))
    js = re.sub(r"\s+", " ", js); js = re.sub(r",\s*\}", "}", js); js = re.sub(r",\s*\]", "]", js)
    try: return json.loads(js)
    except Exception: return None

def _is_heading(requirement_text: str) -> bool:
    t = (requirement_text or "").strip().lower()
    return (not t) or t.endswith(" requirements") or t in {
        "functional requirements","performance requirements","safety requirements",
        "usability requirements","environmental requirements","design inputs","general requirements"
    }

def _join_unique(values: List[str]) -> str:
    vals = [str(v).strip() for v in values if str(v).strip() and str(v).strip().upper() != "NA"]
    seen: List[str] = []
    for v in vals:
        if v not in seen: seen.append(v)
    return ", ".join(seen) if seen else "NA"

def _compose_fallback(rid: str, vid: str, rtxt: str,
                      ha_slice: List[Dict[str, Any]], drow: Dict[str, Any]) -> Dict[str, Any]:
    risk_ids = _join_unique([h.get("risk_id") or h.get("Risk ID") or "" for h in ha_slice])
    risks_to_health = _join_unique([h.get("risk_to_health") or h.get("Risk to Health") or "" for h in ha_slice])
    risk_controls = _join_unique([h.get("risk_control") or h.get("HA Risk Control") or "" for h in ha_slice])
    method = drow.get("verification_method") or drow.get("Verification Method") or "TBD - Human / SME input"
    criteria = drow.get("acceptance_criteria") or drow.get("Acceptance Criteria") or "TBD - Human / SME input"
    return {
        "verification_id": vid, "requirement_id": rid, "requirements": rtxt,
        "risk_ids": risk_ids if risk_ids != "NA" else "TBD - Human / SME input",
        "risks_to_health": risks_to_health if risks_to_health != "NA" else "TBD - Human / SME input",
        "ha_risk_controls": risk_controls if risk_controls != "NA" else "TBD - Human / SME input",
        "verification_method": method, "acceptance_criteria": criteria,
    }

def tm_predict(requirements: List[Dict[str, Any]], ha: List[Dict[str, Any]], dvp: List[Dict[str, Any]]):
    _load_tm_model()
    rows: List[Dict[str, Any]] = []

    from collections import defaultdict
    ha_by_req: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in ha or []:
        rid = str(h.get("requirement_id") or h.get("Requirement ID") or "")
        if rid: ha_by_req[rid].append(h)

    dvp_by_vid: Dict[str, Dict[str, Any]] = {}
    for d in dvp or []:
        vid = str(d.get("verification_id") or d.get("Verification ID") or "")
        if vid and vid not in dvp_by_vid: dvp_by_vid[vid] = d

    for r in requirements:
        rid = str(r.get("Requirement ID", ""))
        vid = str(r.get("Verification ID", ""))
        rtxt = r.get("Requirements", "") or ""

        if _is_heading(rtxt):
            rows.append({
                "verification_id": vid or "NA", "requirement_id": rid, "requirements": rtxt,
                "risk_ids": "NA", "risks_to_health": "NA", "ha_risk_controls": "NA",
                "verification_method": "NA", "acceptance_criteria": "NA",
            })
            continue

        ha_slice = ha_by_req.get(rid, [])
        drow = dvp_by_vid.get(vid, {})

        if not USE_TM_MODEL:
            rows.append(_compose_fallback(rid, vid, rtxt, ha_slice, drow))
            continue

        instruction = (
            "You are generating a Traceability Matrix row for an infusion pump.\n"
            "Return ONLY a single JSON object with keys: "
            "{\"verification_id\",\"requirement_id\",\"requirements\","
            "\"risk_ids\",\"risks_to_health\",\"ha_risk_controls\","
            "\"verification_method\",\"acceptance_criteria\"}.\n"
            "Use comma-separated strings for list-like fields."
        )
        context = {"requirement": {"Requirement ID": rid, "Verification ID": vid, "Requirements": rtxt},
                   "ha": ha_slice, "dvp": drow}
        prompt = instruction + "\nINPUT:\n" + json.dumps(context, ensure_ascii=False)

        parsed = None
        try:
            import torch
            inputs = _tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
            with torch.no_grad():
                outputs = _model.generate(**inputs, max_new_tokens=400, temperature=0.2, do_sample=True, top_p=0.9)  # type: ignore
            decoded = _tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore
            parsed = _extract_json(decoded)
        except Exception:
            parsed = None

        if not parsed:
            rows.append(_compose_fallback(rid, vid, rtxt, ha_slice, drow))
            continue

        risk_ids = _join_unique([h.get("risk_id") or h.get("Risk ID") or "" for h in ha_slice])
        risks_to_health = _join_unique([h.get("risk_to_health") or h.get("Risk to Health") or "" for h in ha_slice])
        risk_controls = _join_unique([h.get("risk_control") or h.get("HA Risk Control") or "" for h in ha_slice])
        method = drow.get("verification_method") or drow.get("Verification Method") or "TBD - Human / SME input"
        criteria = drow.get("acceptance_criteria") or drow.get("Acceptance Criteria") or "TBD - Human / SME input"

        parsed.setdefault("verification_id", vid)
        parsed.setdefault("requirement_id", rid)
        parsed.setdefault("requirements", rtxt)
        parsed.setdefault("risk_ids", risk_ids or "TBD - Human / SME input")
        parsed.setdefault("risks_to_health", risks_to_health or "TBD - Human / SME input")
        parsed.setdefault("ha_risk_controls", risk_controls or "TBD - Human / SME input")
        parsed.setdefault("verification_method", method)
        parsed.setdefault("acceptance_criteria", criteria)

        for k in ("risk_ids","risks_to_health","ha_risk_controls","verification_method","acceptance_criteria"):
            if not str(parsed.get(k, "")).strip():
                parsed[k] = "TBD - Human / SME input"

        rows.append(parsed)
    return rows
