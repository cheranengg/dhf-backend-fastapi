# ======================
# app/models/tm_infer.py
# ======================
import os, json, re
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Defaulted to your path; override with TM_MODEL_DIR env var in prod
TM_MODEL_DIR = os.getenv(
    "TM_MODEL_DIR",
    "/content/drive/MyDrive/Colab Notebooks/Dissertation/mistral_finetuned_Trace_Matrix",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_model = None

_json_obj = re.compile(r"\{[\s\S]*?\}")

def _load_tm_model():
    global _tokenizer, _model
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(TM_MODEL_DIR)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _model = AutoModelForCausalLM.from_pretrained(
        TM_MODEL_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

def _extract_json(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    m = _json_obj.findall(text)
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

def _is_heading(requirement_text: str, verification_id: str) -> bool:
    if not verification_id:
        return True
    t = (requirement_text or "").strip().lower()
    return t in {
        "functional requirements", "performance requirements", "safety requirements",
        "usability requirements", "environmental requirements", "design inputs",
        "general requirements"
    }

def _join_unique(values: List[str]) -> str:
    vals = [str(v).strip() for v in values if str(v).strip() and str(v).strip().upper() != "NA"]
    seen = []
    for v in vals:
        if v not in seen:
            seen.append(v)
    return ", ".join(seen) if seen else "NA"

def tm_predict(requirements: List[Dict[str, Any]], ha: List[Dict[str, Any]], dvp: List[Dict[str, Any]]):
    """Use the fine-tuned Trace Matrix model to produce final rows.
    The model is expected to emit JSON directly. We pass a compact context per
    requirement (Requirement + HA slice + DVP slice), parse JSON, and fall back
    to a rules-based composition with 'TBD - Human / SME input' if needed.
    """
    _load_tm_model()
    rows: List[Dict[str, Any]] = []

    # Index HA by Requirement ID and DVP by Verification ID
    from collections import defaultdict
    ha_by_req: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in ha or []:
        rid = str(h.get("requirement_id")) or str(h.get("Requirement ID", ""))
        if rid:
            ha_by_req[rid].append(h)
    dvp_by_vid: Dict[str, Dict[str, Any]] = {}
    for d in dvp or []:
        vid = str(d.get("verification_id") or d.get("Verification ID") or "")
        if vid and vid not in dvp_by_vid:
            dvp_by_vid[vid] = d

    for r in requirements:
        rid = str(r.get("Requirement ID", ""))
        vid = str(r.get("Verification ID", ""))
        rtxt = r.get("Requirements", "") or ""

        # Headings -> NA
        if _is_heading(rtxt, vid):
            rows.append({
                "verification_id": vid or "NA",
                "requirement_id": rid,
                "requirements": rtxt,
                "risk_ids": "NA",
                "risks_to_health": "NA",
                "ha_risk_controls": "NA",
                "verification_method": "NA",
                "acceptance_criteria": "NA",
            })
            continue

        # Aggregate HA for this requirement
        ha_slice = ha_by_req.get(rid, [])
        risk_ids = _join_unique([h.get("risk_id") or h.get("Risk ID") or "" for h in ha_slice])
        risks_to_health = _join_unique([h.get("risk_to_health") or h.get("Risk to Health") or "" for h in ha_slice])
        risk_controls = _join_unique([h.get("risk_control") or h.get("HA Risk Control") or "" for h in ha_slice])

        # DVP slice
        drow = dvp_by_vid.get(vid, {})
        method = drow.get("verification_method") or drow.get("Verification Method") or "TBD - Human / SME input"
        criteria = drow.get("acceptance_criteria") or drow.get("Acceptance Criteria") or "TBD - Human / SME input"

        # Prompt the TM model (it should output JSON directly)
        instruction = (
            "You are generating a Traceability Matrix row for an infusion pump.\n"
            "Return ONLY a single JSON object with keys:\n"
            "{\"verification_id\",\"requirement_id\",\"requirements\","
            "\"risk_ids\",\"risks_to_health\",\"ha_risk_controls\","
            "\"verification_method\",\"acceptance_criteria\"}.\n"
            "Use comma-separated strings for fields that may contain multiple values."
        )
        context = {
            "requirement": {"Requirement ID": rid, "Verification ID": vid, "Requirements": rtxt},
            "ha": ha_slice,
            "dvp": drow,
        }
        prompt = instruction + "\nINPUT:\n" + json.dumps(context, ensure_ascii=False)

        try:
            inputs = _tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = _model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.9,
                )
            decoded = _tokenizer.decode(outputs[0], skip_special_tokens=True)
            parsed = _extract_json(decoded)
        except Exception:
            parsed = None

        if not parsed:
            # Fallback
            parsed = {
                "verification_id": vid,
                "requirement_id": rid,
                "requirements": rtxt,
                "risk_ids": risk_ids if risk_ids != "NA" else "TBD - Human / SME input",
                "risks_to_health": risks_to_health if risks_to_health != "NA" else "TBD - Human / SME input",
                "ha_risk_controls": risk_controls if risk_controls != "NA" else "TBD - Human / SME input",
                "verification_method": method,
                "acceptance_criteria": criteria,
            }
        else:
            # Fill any gaps and normalize
            parsed.setdefault("verification_id", vid)
            parsed.setdefault("requirement_id", rid)
            parsed.setdefault("requirements", rtxt)
            parsed.setdefault("risk_ids", risk_ids or "TBD - Human / SME input")
            parsed.setdefault("risks_to_health", risks_to_health or "TBD - Human / SME input")
            parsed.setdefault("ha_risk_controls", risk_controls or "TBD - Human / SME input")
            parsed.setdefault("verification_method", method)
            parsed.setdefault("acceptance_criteria", criteria)

            for k in ["risk_ids","risks_to_health","ha_risk_controls","verification_method","acceptance_criteria"]:
                if not str(parsed.get(k, "")).strip():
                    parsed[k] = "TBD - Human / SME input"

        rows.append(parsed)

    return rows
