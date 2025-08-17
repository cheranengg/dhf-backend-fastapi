# app/models/ha_infer.py
from __future__ import annotations

import os
import re
import gc
import json
import random
from typing import List, Dict, Any, Optional, Tuple

import torch

# ---------------------------
# Environment / toggles
# ---------------------------
USE_HA_ADAPTER: bool = os.getenv("USE_HA_ADAPTER", "1") == "1"
BASE_MODEL_ID: str = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HA_ADAPTER_REPO: str = os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter")

# RAG: generic JSONL (synthetic HA or MAUDE distilled)
HA_RAG_PATH: str = os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl")

# Local MAUDE
MAUDE_LOCAL_PATH: str = os.getenv("MAUDE_LOCAL_JSONL", "app/rag_sources/maude_sigma_spectrum.jsonl")
MAUDE_LOCAL_ONLY: bool = os.getenv("MAUDE_LOCAL_ONLY", "1") == "1"
MAUDE_FRACTION: float = float(os.getenv("MAUDE_FRACTION", "0.70"))

# Generation controls
HA_MAX_NEW_TOKENS: int = int(os.getenv("HA_MAX_NEW_TOKENS", "192"))
NUM_BEAMS: int = int(os.getenv("NUM_BEAMS", "1"))
DO_SAMPLE: bool = os.getenv("do_sample", "1") == "1"
TOP_P: float = float(os.getenv("HA_TOP_P", "0.90"))
TEMPERATURE: float = float(os.getenv("HA_TEMPERATURE", "0.35"))
REPETITION_PENALTY: float = float(os.getenv("HA_REPETITION_PENALTY", "1.05"))

# Safety / speed
FORCE_CPU: bool = os.getenv("FORCE_CPU", "0") == "1"
OFFLOAD_DIR: str = os.getenv("OFFLOAD_DIR", "/tmp/offload")
CACHE_DIR: str = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"  # fixed

# Input length cap
HA_INPUT_MAX_TOKENS: int = int(os.getenv("HA_INPUT_MAX_TOKENS", "512"))

# Paraphrase RAG
PARAPHRASE_FROM_RAG: bool = os.getenv("PARAPHRASE_FROM_RAG", "1") == "1"
PARAPHRASE_MAX_WORDS: int = int(os.getenv("PARAPHRASE_MAX_WORDS", "22"))

# Cap rows
ROW_LIMIT: int = int(os.getenv("HA_ROW_LIMIT", "50"))

# Debug
DEBUG_HA: bool = os.getenv("DEBUG_HA", "1") == "1"
DEBUG_PEEK_CHARS: int = int(os.getenv("DEBUG_PEEK_CHARS", "320"))

# ---------------------------
# Globals
# ---------------------------
_tokenizer = None
_model = None
_RAG_DB: List[Dict[str, Any]] = []
_MAUDE_ROWS: List[Dict[str, Any]] = []
_logged_model_banner = False

# ---------------------------
# Controlled vocab
# ---------------------------
RISK_TO_HEALTH_CHOICES = [
    "Air Embolism", "Allergic response", "Delay of therapy", "Environmental Hazard",
    "Incorrect Therapy", "Infection", "Overdose", "Particulate", "Trauma", "Underdose",
]

HARM_BY_RTH = {
    "Air Embolism": ["Pulmonary Embolism", "Stroke", "Shortness of breath", "Severe Injury", "Death"],
    "Allergic response": ["Allergic reaction (Systemic / Localized)", "Toxic effects", "Severe Injury"],
    "Delay of therapy": ["Disease Progression", "Severe Injury", "Death"],
    "Environmental Hazard": ["Toxic effects", "Chemical burns", "Severe Injury"],
    "Incorrect Therapy": ["Hypertension", "Hypotension", "Cardiac Arrhythmia", "Tachycardia", "Bradycardia", "Seizure", "Organ damage"],
    "Infection": ["Sepsis", "Cellulitis", "Severe Septic Shock"],
    "Overdose": ["Organ Failure", "Cardiac Arrhythmia", "Toxic effects"],
    "Underdose": ["Progression of untreated condition", "Severe Injury"],
    "Particulate": ["Embolism", "Organ damage", "Severe Injury"],
    "Trauma": ["Severe Injury", "Organ damage", "Bradycardia"],
}

# Requirement → (hazard, hazardous_situation, risk_to_health)
REQ_TO_HA_PATTERNS: List[Tuple[List[str], Tuple[str, str, str]]] = [
    (["air-in-line", "air in line", "bubble", "air detection"],
     ("Air-in-line not detected", "Patient receives air", "Air Embolism")),
    (["occlusion", "blockage", "line occlusion"],
     ("Line occlusion", "Flow restricted during therapy", "Delay of therapy")),
    (["flow", "accuracy", "rate"],
     ("Inaccurate flow rate", "Incorrect volume delivered", "Incorrect Therapy")),
    (["leakage current", "patient leakage"],
     ("Electrical leakage", "Patient contacted by leakage current", "Trauma")),
    (["dielectric", "hi-pot", "hipot"],
     ("Insulation breakdown", "Breakdown under high potential", "Trauma")),
    (["insulation resistance", "insulation"],
     ("Insulation degradation", "Compromised insulation", "Trauma")),
    (["protective earth", "earth continuity"],
     ("Protective earth failure", "Accessible parts not bonded", "Trauma")),
    (["alarm"],
     ("Alarm failure", "Alarm not triggered or inaudible", "Delay of therapy")),
    (["emc", "immunity", "emission", "esd", "radiated", "conducted"],
     ("Electromagnetic interference", "Device behavior affected by EM field", "Incorrect Therapy")),
    (["ip", "ingress", "water", "drip"],
     ("Liquid ingress", "Moisture enters enclosure", "Incorrect Therapy")),
    (["drop", "shock", "impact", "vibration"],
     ("Mechanical shock/vibration", "Component/connection damage", "Trauma")),
    (["battery", "power", "shutdown"],
     ("Battery failure", "Unexpected shutdown", "Delay of therapy")),
    (["usability", "human factors", "ui", "use error", "lockout"],
     ("Use error", "User action leads to incorrect setup", "Incorrect Therapy")),
    (["label", "marking", "symbol", "udi"],
     ("Labeling error", "User misinterprets label/IFU", "Incorrect Therapy")),
    (["luer", "connector", "80369"],
     ("Misconnection", "Wrong small-bore connection", "Incorrect Therapy")),
    (["temperature rise", "clause 11", "overheating"],
     ("Overheating", "Accessible parts exceed safe temp", "Trauma")),
]

# ---------------------------
# Utils
# ---------------------------
def _jsonl_load(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _maybe_truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

def _paraphrase_sentence(text: str) -> str:
    return re.sub(r"\b(device|system|pump)\b", "infusion system", text, flags=re.I)

def _gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _log_once_banner():
    global _logged_model_banner
    if not _logged_model_banner:
        print(f"[ha_infer] Using base={BASE_MODEL_ID}, adapter={HA_ADAPTER_REPO if USE_HA_ADAPTER else 'None'}, cache={CACHE_DIR}")
        _logged_model_banner = True

def _get_req_id(item: Any) -> str:
    if isinstance(item, dict):
        for k in ("Requirement ID", "requirement_id", "req_id", "RequirementID"):
            v = item.get(k)
            if v:
                return str(v).strip()
    return ""

def _get_req_text(item: Any) -> str:
    if isinstance(item, dict):
        for k in ("Requirements", "requirements", "requirement", "Requirement"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # fallback: stringify the dict (but shorten)
        return _maybe_truncate_words(json.dumps(item, ensure_ascii=False), 40)
    # string input
    return str(item).strip()

# ---------------------------
# Model loader
# ---------------------------
def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
    _tokenizer.pad_token = _tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=CACHE_DIR,
        offload_folder=OFFLOAD_DIR,
    )

    if USE_HA_ADAPTER:
        model = PeftModel.from_pretrained(model, HA_ADAPTER_REPO, cache_dir=CACHE_DIR)

    _model = model
    _log_once_banner()
    return _tokenizer, _model

# ---------------------------
# RAG / MAUDE
# ---------------------------
def _load_rag_once():
    global _RAG_DB
    if not _RAG_DB:
        _RAG_DB = _jsonl_load(HA_RAG_PATH)

def _load_maude_once():
    global _MAUDE_ROWS
    if not _MAUDE_ROWS and os.path.exists(MAUDE_LOCAL_PATH):
        _MAUDE_ROWS = _jsonl_load(MAUDE_LOCAL_PATH)

def _pick_rag_seed() -> Optional[Dict[str, Any]]:
    _load_rag_once()
    if not _RAG_DB:
        return None
    return random.choice(_RAG_DB)

def _maude_snippets(n: int = 3) -> List[str]:
    _load_maude_once()
    if not _MAUDE_ROWS:
        return []
    # very short snippets just to inject variety
    texts: List[str] = []
    for r in random.sample(_MAUDE_ROWS, min(n * 4, len(_MAUDE_ROWS))):
        for key in ("event_description", "device_problem_text", "event_text", "text", "mdr_text"):
            v = r.get(key)
            if isinstance(v, str) and len(v.strip()) > 30:
                texts.append(_maybe_truncate_words(v.strip(), 30))
    random.shuffle(texts)
    return texts[:n]

# ---------------------------
# Prompt + Generation
# ---------------------------
def _build_prompt(requirement_text: str, rag_seed: Optional[Dict[str, Any]]) -> str:
    context = ""
    if rag_seed:
        seeds = []
        for k in ("hazard", "hazardous_situation", "risk_to_health", "text"):
            v = rag_seed.get(k)
            if isinstance(v, str) and v.strip():
                s = _paraphrase_sentence(v) if PARAPHRASE_FROM_RAG else v
                seeds.append(s)
        if seeds:
            context = "\nContext: " + " | ".join(seeds[:3])

    if not MAUDE_LOCAL_ONLY:
        m = _maude_snippets(2)
        if m:
            context += "\nMAUDE Evidence: " + " | ".join(m)

    return (
        f"### Instruction:\n"
        f"Generate a concise hazard analysis JSON for an infusion pump requirement.\n"
        f"Requirement: {requirement_text}\n"
        f"{context}\n\n"
        f"### Response:\n"
    )

def _decode_json_from_text(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {}

def _generate_json(prompt: str) -> Dict[str, Any]:
    tok, model = _load_model()
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=HA_INPUT_MAX_TOKENS).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=HA_MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tok.eos_token_id,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    if DEBUG_HA:
        print("[DEBUG decoded]", decoded[:DEBUG_PEEK_CHARS])
    return _decode_json_from_text(decoded)

# ---------------------------
# Mapping helpers
# ---------------------------
def choose_harm(risk_to_health: str) -> str:
    harms = HARM_BY_RTH.get(risk_to_health, [])
    if harms:
        return random.choice(harms)
    return "Severe Injury"

def suggest_control(hazard: str, requirement_text: str) -> str:
    hz = hazard.lower()
    rt = requirement_text.lower()
    if "flow" in rt or "accuracy" in rt:
        return "Flow calibration; verification per IEC 60601-2-24"
    if "air" in rt:
        return "Ultrasonic air-in-line detector with alarm; purge guidance"
    if "occlusion" in rt or "blockage" in rt:
        return "Pressure sensing with occlusion alarm; tubing management"
    if "leakage current" in rt:
        return "Leakage current limits; type testing per IEC 60601-1"
    if "dielectric" in rt or "hipot" in rt:
        return "Dielectric withstand test; insulation design margins"
    if "insulation" in rt:
        return "Creepage/clearance compliance; insulation resistance checks"
    if "protective earth" in rt:
        return "Bonding resistance test; protective earth integrity"
    if "alarm" in rt:
        return "Alarm verification per IEC 60601-1-8; audibility tests"
    if "emc" in rt or "immunity" in rt or "esd" in rt:
        return "EMC testing (IEC 60601-1-2); functional performance criteria"
    if "ingress" in rt or "ip" in rt or "drip" in rt:
        return "IP testing per IEC 60529; gasket and seam protection"
    if "vibration" in rt or "shock" in rt or "drop" in rt or "impact" in rt:
        return "Mechanical robustness tests; secure connectors"
    if "battery" in rt or "power" in rt:
        return "Battery monitoring; backup and safe-shutdown behavior"
    if "label" in rt or "marking" in rt or "udi" in rt:
        return "Label content verification; IFU validation and symbols per ISO 15223-1"
    if "luer" in rt or "connector" in rt or "80369" in rt:
        return "ISO 80369-7 compliance; misconnection prevention"
    if "temperature" in rt or "overheating" in rt:
        return "Temperature rise tests (clause 11); thermal protection"
    # generic but still meaningful
    if "electrical" in hz:
        return "Insulation barriers and protective earthing; type testing"
    return "Risk controls via design/alarms/labeling; verification per relevant standards"

RISK_WORD_TO_LEVEL = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4, "Very High": 5}
LEVEL_TO_RISK_WORD = {v: k for k, v in RISK_WORD_TO_LEVEL.items()}

def _seq_templates(hazard: str, rth: str) -> list[str]:
    hz = hazard.lower()
    if "air-in-line" in hz or "air in line" in hz or "air" in hz:
        return [
            "Air enters set → detector fails/late → patient receives air",
            "Bubble bypasses sensor → venous access → embolic event",
            "Residual air after setup → inadequate purge → embolism risk",
        ]
    if "occlusion" in hz or "blockage" in hz:
        return [
            "Upstream blockage → pump continues/alarms late → restricted delivery",
            "Kinked tubing → pressure rises → therapy interrupted",
            "Downstream clamp left closed → occlusion develops → delivery stops",
        ]
    if "flow" in hz or "inaccurate" in hz:
        return [
            "Calibration drift → setpoint not met → incorrect volume delivered",
            "Misassembled set → flow error → under/over delivery",
            "Viscosity/height change → un-compensated → rate deviates",
        ]
    if "emc" in hz or "interference" in hz or "esd" in hz:
        return [
            "EM field couples into electronics → control loop perturbed → wrong output",
            "ESD hit → transient upset → alarm/therapy error",
            "Radiated immunity failure → spurious inputs → incorrect therapy",
        ]
    if "leakage" in hz or "earth" in hz or "insulation" in hz or "dielectric" in hz:
        return [
            "Insulation degradation → leakage path forms → patient contact current",
            "Protective earth open → fault current through patient",
            "Dielectric breakdown → exposed conductive part energized",
        ]
    if "ingress" in hz or "liquid" in hz or "ip" in hz:
        return [
            "Liquid drip enters enclosure → short/erratic behavior → wrong output",
            "Condensation in device → sensor error → incorrect therapy",
        ]
    if "label" in hz or "use error" in hz:
        return [
            "Ambiguous label/IFU → user sets wrong value → incorrect therapy",
            "Similar connector/marking → misconnection → wrong route/dose",
        ]
    if "shock" in hz or "impact" in hz or "vibration" in hz or "drop" in hz:
        return [
            "Mechanical shock → component shifts → intermittent failure",
            "Vibration loosens fitting → leak → therapy interruption",
        ]
    # generic set
    return [
        "Design or use issue evolves → hazardous device state → patient impact",
        "Fault undetected → device behavior drifts → hazardous situation",
        "Abnormal condition persists → unsafe output → clinical harm",
    ]


def _derive_severity_pxx(rth: str, hazard: str) -> tuple[str, str, str, str]:
    """
    Returns (severity_of_harm, p0, p1, poh).
    Severity is '1'..'5'; p0/p1/poh ∈ {Very Low, Low, Medium, High, Very High}.
    """
    # severity priors by Risk to Health
    sev_map = {
        "Air Embolism": (4, 5),
        "Overdose": (3, 5),
        "Incorrect Therapy": (2, 4),
        "Underdose": (2, 4),
        "Infection": (3, 4),
        "Trauma": (2, 4),
        "Particulate": (2, 4),
        "Delay of therapy": (2, 4),
        "Environmental Hazard": (2, 3),
        "Allergic response": (2, 3),
    }
    lo, hi = sev_map.get(rth, (2, 4))
    severity = str(random.randint(lo, hi))

    # p0 by hazard “type”
    hz = hazard.lower()
    if any(k in hz for k in ["air-in-line", "air ", "embol"]):
        p0 = "High"
    elif any(k in hz for k in ["occlusion", "blockage"]):
        p0 = "Medium"
    elif any(k in hz for k in ["flow", "inaccurate", "label", "use error", "emc", "interference"]):
        p0 = "Medium"
    elif any(k in hz for k in ["leakage", "earth", "insulation", "dielectric"]):
        p0 = "Low"
    else:
        p0 = random.choice(["Low", "Medium"])

    # Controls reduce p1 and poh by ~1 level (bounded to 1..5)
    def down1(word: str) -> str:
        lvl = max(1, RISK_WORD_TO_LEVEL[word] - 1)
        return LEVEL_TO_RISK_WORD[lvl]

    p1 = down1(p0)
    poh = down1(p1)

    return severity, p0, p1, poh


def _risk_index_from(severity: str, p0: str) -> str:
    # simple matrix on sev (1..5) × likelihood (1..5)
    s = max(1, min(5, int(severity) if severity.isdigit() else 3))
    l = RISK_WORD_TO_LEVEL.get(p0, 3)
    score = s * l  # 1..25
    if score >= 16:
        return "High"
    if score >= 9:
        return "Medium"
    return "Low"

def _ensure_fields(obj: Dict[str, Any], requirement_text: str, idx: int) -> Dict[str, Any]:
    risk_id = f"HA-{idx+1:03d}"

    # heuristic mapping from requirement text
    matched = None
    lt = requirement_text.lower()
    for keys, (haz, sit, rth) in REQ_TO_HA_PATTERNS:
        if any(k in lt for k in keys):
            matched = (haz, sit, rth)
            break

    hazard = matched[0] if matched else obj.get("hazard", "Device malfunction")
    situation = matched[1] if matched else obj.get("hazardous_situation", "Patient exposed to device fault")
    risk_to_health = matched[2] if matched else obj.get("risk_to_health", random.choice(RISK_TO_HEALTH_CHOICES))

    # extra diversity by requirement keywords when no direct pattern matched
    if not matched:
        if "flow" in lt:
            hazard, situation, risk_to_health = "Inaccurate flow rate", "Incorrect volume delivered", "Incorrect Therapy"
        elif "air" in lt:
            hazard, situation, risk_to_health = "Air-in-line not detected", "Patient receives air", "Air Embolism"
        elif "occlusion" in lt or "blockage" in lt:
            hazard, situation, risk_to_health = "Line occlusion", "Flow restricted during therapy", "Delay of therapy"

    # sequence of events: pick varied templates and shuffle wording lightly
    seq_options = _seq_templates(hazard, risk_to_health)
    sequence = random.choice(seq_options)
    sequence = sequence.replace(" → ", " \u2192 ")  # nice arrow
    if random.random() < 0.35:
        sequence = sequence.capitalize()

    # severity & likelihoods (p0, p1, poh) + derived risk index
    severity, p0, p1, poh = _derive_severity_pxx(risk_to_health, hazard)
    risk_index = _risk_index_from(severity, p0)

    # harm consistent with Risk to Health
    harm = choose_harm(risk_to_health)

    # allow model JSON to override if it provided reasonable fields
    seq_in = obj.get("sequence_of_events")
    if isinstance(seq_in, str) and len(seq_in.strip()) > 8:
        sequence = seq_in.strip()

    sev_in = obj.get("severity") or obj.get("severity_of_harm")
    try:
        sev_val = int(str(sev_in))
        if 1 <= sev_val <= 5:
            severity = str(sev_val)
            risk_index = _risk_index_from(severity, p0)
    except Exception:
        pass

    return {
        "risk_id": risk_id,
        "risk_to_health": risk_to_health,
        "hazard": hazard,
        "hazardous_situation": situation,
        "harm": harm,
        "sequence_of_events": sequence,
        "severity_of_harm": severity,
        "p0": p0,
        "p1": p1,
        "poh": poh,
        "risk_index": risk_index,
        "risk_control": suggest_control(hazard, requirement_text),
    }


# ---------------------------
# Public
# ---------------------------
def infer_from_requirement(item: Any, idx: int) -> Dict[str, Any]:
    req_text = _get_req_text(item)
    rag_seed = _pick_rag_seed()
    prompt = _build_prompt(req_text, rag_seed)
    raw = _generate_json(prompt)
    data = _ensure_fields(raw, req_text, idx)

    rid = _get_req_id(item)
    if rid:
        data["requirement_id"] = rid
    return data

def infer_ha(requirements: List[Any]) -> List[Dict[str, Any]]:
    results = []
    for idx, item in enumerate((requirements or [])[:ROW_LIMIT]):
        try:
            results.append(infer_from_requirement(item, idx))
        except Exception as e:
            if DEBUG_HA:
                print(f"[ha] row {idx} error: {e}")
            # Still produce a row so downstream never breaks
            req_text = _get_req_text(item)
            fallback = {
                "risk_id": f"HA-{idx+1:03d}",
                "risk_to_health": random.choice(RISK_TO_HEALTH_CHOICES),
                "hazard": "Device malfunction",
                "hazardous_situation": "Patient exposed to device fault",
                "harm": "Severe Injury",
                "sequence_of_events": "Design or use issue leads to hazardous condition",
                "severity_of_harm": "3",
                "p0": "Medium",
                "p1": "Medium",
                "poh": "Low",
                "risk_index": "Medium",
                "risk_control": suggest_control("Device malfunction", req_text),
            }
            rid = _get_req_id(item)
            if rid:
                fallback["requirement_id"] = rid
            results.append(fallback)
    _gc_cuda()
    return results
