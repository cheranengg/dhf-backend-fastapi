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

import os
from typing import List, Dict, Any, Optional

USE_DVP_MODEL = os.getenv("USE_DVP_MODEL", "0") == "1"
DVP_MODEL_DIR = os.getenv("DVP_MODEL_DIR", "/models/dvp-merged")

HF_TOKEN = os.getenv("HF_TOKEN", None)
CACHE_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/workspace/.cache/hf"
def _token_cache_kwargs():
    kw = {"cache_dir": CACHE_DIR}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw

if USE_DVP_MODEL:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
else:
    torch = None  # type: ignore
    AutoTokenizer = AutoModelForCausalLM = None  # type: ignore

_tokenizer: Optional["AutoTokenizer"] = None
_model: Optional["AutoModelForCausalLM"] = None

USABILITY_KWS = ["usability", "user", "human factors", "ui", "interface"]
VISUAL_KWS    = ["label", "marking", "display", "visual", "color"]
TECH_KWS      = ["electrical", "mechanical", "flow", "pressure", "occlusion", "accuracy", "alarm"]

TEST_SPEC_LOOKUP = {
    "insulation": "≥ 50 MΩ at 500 V DC (IEC 60601-1)",
    "leakage": "≤ 100 µA at rated voltage (IEC 60601-1)",
    "dielectric": "1500 V AC for 1 min, no breakdown (IEC 60601-1)",
    "flow": "±5% from set value (IEC 60601-2-24)",
    "occlusion": "Alarm ≤ 30 s at 100 kPa back pressure (IEC 60601-2-24)",
    "luer": "No leakage under 300 kPa for 30 s (ISO 80369-7)",
    "emc": "±6 kV contact, ±8 kV air (IEC 61000-4-2)",
    "vibration": "10–500 Hz, 0.5 g, 2 h/axis (IEC 60068-2-6)",
}
SEVERITY_TO_N = {5: 50, 4: 40, 3: 30, 2: 20, 1: 10}

def _get_verification_method(req_text: str) -> str:
    t = (req_text or "").lower()
    if any(k in t for k in USABILITY_KWS): return "NA"
    if any(k in t for k in VISUAL_KWS):    return "Visual Inspection"
    if any(k in t for k in TECH_KWS):      return "Physical Testing"
    return "Physical Inspection"

def _get_sample_size(requirement_id: str, ha_items: List[Dict[str, Any]]) -> str:
    sev = None
    for h in ha_items or []:
        if str(h.get("requirement_id") or h.get("Requirement ID")) == str(requirement_id):
            s = h.get("severity_of_harm") or h.get("Severity of Harm")
            try:
                s_int = int(str(s))
                sev = s_int if (sev is None or s_int > sev) else sev
            except Exception:
                continue
    if sev is not None: return str(SEVERITY_TO_N.get(int(sev), 30))
    try:
        digit = int(str(requirement_id).split("-")[-1]) % 5
        return str(20 + digit * 5)
    except Exception:
        return "30"

def _load_model():
    global _tokenizer, _model
    if not USE_DVP_MODEL or _model is not None: return
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    src = DVP_MODEL_DIR
    _tokenizer = AutoTokenizer.from_pretrained(src, **_token_cache_kwargs())  # type: ignore
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, **_token_cache_kwargs())  # type: ignore
    _model.to("cuda" if torch.cuda.is_available() else "cpu")

def _gen_test_procedure_model(requirement_text: str) -> str:
    _load_model()
    import torch
    prompt = (
        "You are a compliance engineer.\n\n"
        "Generate ONLY a Design Verification Test Procedure for the following requirement.\n\n"
        f"Requirement: {requirement_text}\n\n"
        "- Output exactly 3–4 concise bullets with measurable values and thresholds."
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=320, temperature=0.3, do_sample=True, top_p=0.9)  # type: ignore
    text = _tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore
    bullets = []
    for line in text.split("\n"):
        s = line.strip(" -•\t")
        if s and len(s.split()) > 3:
            bullets.append(f"- {s}")
        if len(bullets) == 4: break
    return "\n".join(bullets) if bullets else "TBD"

def _gen_test_procedure_stub(requirement_text: str) -> str:
    req = (requirement_text or "").strip() or "the feature"
    return (
        f"- Verify {req} at three setpoints; record measured vs setpoint (n=3).\n"
        f"- Confirm repeatability across 5 cycles; compute max deviation and std dev.\n"
        f"- Boundary test at min/max conditions; log pass/fail vs spec.\n"
        f"- Record equipment IDs & calibration dates; attach raw data."
    )

def _gen_test_procedure(requirement_text: str) -> str:
    return _gen_test_procedure_model(requirement_text) if USE_DVP_MODEL else _gen_test_procedure_stub(requirement_text)

def dvp_predict(requirements: List[Dict[str, Any]], ha: List[Dict[str, Any]]):
    _load_model()
    rows: List[Dict[str, Any]] = []
    for r in requirements:
        rid = str(r.get("Requirement ID", "") or "")
        vid = str(r.get("Verification ID", "") or "")
        rtxt = r.get("Requirements", "") or ""
        if rtxt.strip().lower().endswith("requirements") and not vid:
            rows.append({
                "verification_id": vid, "requirement_id": rid, "requirements": rtxt,
                "verification_method": "NA", "sample_size": "NA",
                "test_procedure": "NA", "acceptance_criteria": "NA",
            })
            continue
        method = _get_verification_method(rtxt)
        sample = _get_sample_size(rid, ha or [])
        bullets = _gen_test_procedure(rtxt)
        ac = "TBD"
        for kw, spec in TEST_SPEC_LOOKUP.items():
            if kw in (rtxt or "").lower(): ac = spec; break

        rows.append({
            "verification_id": vid, "requirement_id": rid, "requirements": rtxt,
            "verification_method": method or "NA", "sample_size": sample or "NA",
            "test_procedure": bullets, "acceptance_criteria": ac,
        })
    return rows
