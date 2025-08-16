# app/models/ha_infer.py
from __future__ import annotations
import os, json, time, random, re, hashlib
from typing import Any, Dict, List, Tuple

import requests

# -----------------------
# Env & runtime toggles
# -----------------------
BASE_MODEL_ID   = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
USE_HA_ADAPTER  = os.getenv("USE_HA_ADAPTER", "1") == "1"
HA_ADAPTER_REPO = os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter")

# RAG (synthetic HA) file
HA_RAG_PATH     = os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl")

# MAUDE dialing:
MAUDE_FETCH     = os.getenv("MAUDE_FETCH", "1") == "1"     # keep ON
MAUDE_DEVICE    = os.getenv("MAUDE_DEVICE", "SIGMA SPECTRUM")  # **single** device
MAUDE_LIMIT     = int(os.getenv("MAUDE_LIMIT", "20"))
MAUDE_TTL       = int(os.getenv("MAUDE_TTL", "86400"))
# NEW: only use MAUDE for 70% of rows (randomized per request-row)
MAUDE_SAMPLE_RATE = float(os.getenv("MAUDE_SAMPLE_RATE", "0.7"))

# Quick test limiter (main.py also enforces)
QUICK_LIMIT     = int(os.getenv("QUICK_LIMIT", "5"))

# Optional paraphrasing of exact RAG matches
PARA_ENABLE     = os.getenv("PARAPHRASE_FROM_RAG", "1") == "1"
PARA_MAX_WORDS  = int(os.getenv("PARAPHRASE_MAX_WORDS", "24"))
SIM_THRESHOLD   = float(os.getenv("SIM_THRESHOLD", "0.88"))

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
)

CACHE_DIR       = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"
MAUDE_CACHE_DIR = os.getenv("MAUDE_CACHE_DIR", "/tmp/maude_cache")

os.makedirs(MAUDE_CACHE_DIR, exist_ok=True)

# ------------------------------------------------
# Lightweight adapter load (same as earlier build)
# ------------------------------------------------
_tokenizer = None
_model     = None

def _load_model():
    """Load the base instruct model (+ adapter if present)."""
    global _tokenizer, _model
    if _model is not None:
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    kw = {"cache_dir": CACHE_DIR}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **kw)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
        **kw
    )

    # Attach PEFT adapter (LoRA) if requested
    if USE_HA_ADAPTER:
        try:
            from peft import PeftModel
            _model = PeftModel.from_pretrained(_model, HA_ADAPTER_REPO, **kw)
        except Exception as e:
            print({"ha_adapter_load_warning": str(e)})

    try:
        _model.config.pad_token_id = _tokenizer.pad_token_id
    except Exception:
        pass


# ----------------------
# RAG: load once
# ----------------------
_RAG_DB: List[Dict[str, Any]] = []

def _load_rag():
    global _RAG_DB
    if _RAG_DB:
        return
    try:
        with open(HA_RAG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    _RAG_DB.append(obj)
                except Exception:
                    pass
        print({"ha_rag": "loaded", "path": HA_RAG_PATH, "rows": len(_RAG_DB)})
    except FileNotFoundError:
        print({"ha_rag": "missing", "path": HA_RAG_PATH})
        _RAG_DB = []


# ----------------------
# MAUDE helpers (70%)
# ----------------------
def _cache_path(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return os.path.join(MAUDE_CACHE_DIR, f"{h}.json")

def _read_cache(key: str) -> Dict[str, Any] | None:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # TTL
        ts = obj.get("_ts", 0)
        if time.time() - ts > MAUDE_TTL:
            return None
        return obj
    except Exception:
        return None

def _write_cache(key: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["_ts"] = int(time.time())
    with open(_cache_path(key), "w", encoding="utf-8") as f:
        json.dump(payload, f)

def _fetch_maude_events(brand: str, limit: int) -> List[Dict[str, Any]]:
    """Fetch MAUDE event records (FDA) for brand name."""
    url = "https://api.fda.gov/device/event.json"
    params = {"search": f'device.brand_name:"{brand}"', "limit": limit}
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])

def _collect_maude_narratives(events: List[Dict[str, Any]]) -> List[str]:
    out = []
    for e in events or []:
        # two possible fields:
        tlist = e.get("mdr_text", []) or []
        for t in tlist:
            txt = t.get("text")
            if txt and isinstance(txt, str):
                # keep reasonably sized snippets
                txt = re.sub(r"\s+", " ", txt).strip()
                if 50 <= len(txt) <= 1200:
                    out.append(txt)
    return out


# ----------------------
# Tiny similarity & paraphrase
# ----------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _sim(a: str, b: str) -> float:
    """Very small Jaccard-ish sim for de-dup control."""
    A = set(_norm(a).split())
    B = set(_norm(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))

def _paraphrase(text: str, max_words: int = 24) -> str:
    """Cheap paraphrase using the model itself (short generation)."""
    if not text:
        return text
    try:
        _load_model()
        import torch
        prompt = (
            "Paraphrase the following in different wording but same meaning. "
            "Keep under {} words.\n\nText: {}\n\nParaphrase:".format(max_words, text)
        )
        inputs = _tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        dec = _tokenizer.decode(out[0], skip_special_tokens=True)
        # Keep the last line as the paraphrase
        lines = [ln.strip() for ln in dec.split("\n") if ln.strip()]
        para = lines[-1]
        # trim to ~max_words
        words = para.split()
        if len(words) > max_words:
            para = " ".join(words[:max_words])
        return para
    except Exception as e:
        print({"paraphrase_warning": str(e)})
        return text


# ----------------------
# Core HA row synthesis
# ----------------------
def _risks_for_requirement(req_text: str) -> List[str]:
    # fixed palette like before
    palette = [
        "Air Embolism", "Allergic response", "Infection", "Overdose",
        "Underdose", "Delay of therapy", "Environmental Hazard",
        "Incorrect Therapy", "Trauma", "Particulate",
    ]
    # simple select 1–2 based on hash
    h = int(hashlib.md5(_norm(req_text).encode("utf-8")).hexdigest(), 16)
    idx = h % len(palette)
    idx2 = (idx + 3) % len(palette)
    return [palette[idx], palette[idx2]]

def _row_from_sources(req_id: str, risk: str, narratives: List[str]) -> Dict[str, str]:
    """
    Compress multiple narratives into structured HA fields with short phrasing.
    """
    # defaults
    hazard = "Device malfunction"
    situation = "Patient exposed to device fault"
    harm = "Adverse physiological effects"
    soe = "Improper setup or device issue led to patient exposure"

    # If we have narratives, extract a few hints
    blob = " ".join(narratives[:3])  # limit to 3 chunks per risk
    blob_n = _norm(blob)

    # hazard-ish
    if "air" in blob_n and "line" in blob_n:
        hazard = "Air in Line"
        situation = "Patient exposed to intravenous air"
        harm = "Shortness of breath, Cardiac Arrhythmia"
        soe = "Set not primed before attached to pump & patient"
    elif "occlusion" in blob_n or "block" in blob_n:
        hazard = "Occlusion"
        situation = "Flow interrupted due to downstream blockage"
        harm = "Delay of therapy, Pain at insertion site"
        soe = "Tubing occluded; pressure rise during infusion"
    elif "leak" in blob_n:
        hazard = "Leakage"
        situation = "Fluid leakage around line/connector"
        harm = "Exposure to infusion fluid; Loss of intended therapy"
        soe = "Loose connector led to gradual leak during use"

    # guard + paraphrase-lite to avoid copy-paste
    if PARA_ENABLE:
        hazard = _paraphrase(hazard, 8) if hazard else hazard
        situation = _paraphrase(situation, 16) if situation else situation
        harm = _paraphrase(harm, 16) if harm else harm
        soe = _paraphrase(soe, 18) if soe else soe

    return {
        "risk_to_health": risk,
        "hazard": hazard,
        "hazardous_situation": situation,
        "harm": harm,
        "sequence_of_events": soe,
    }


# ----------------------
# Public API
# ----------------------
def infer_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _load_model()
    _load_rag()

    # QUICK limit (defensive)
    if QUICK_LIMIT and len(requirements) > QUICK_LIMIT:
        requirements = requirements[:QUICK_LIMIT]

    # 1) RAG small helper (closest hazard template by simple token overlap)
    def _rag_hint(risk: str) -> Dict[str, str] | None:
        best, best_s = None, 0.0
        for row in _RAG_DB or []:
            cand = str(row.get("Risk to Health") or row.get("risk_to_health") or "")
            s = _sim(cand, risk)
            if s > best_s:
                best_s = s
                best = row
        if best and best_s >= SIM_THRESHOLD:
            out = {
                "risk_to_health": risk,
                "hazard": best.get("Hazard") or "",
                "hazardous_situation": best.get("Hazardous situation") or "",
                "harm": best.get("Harm") or "",
                "sequence_of_events": best.get("Sequence of Events") or "",
            }
            # paraphrase to avoid verbatim
            if PARA_ENABLE:
                for k in ("hazard", "hazardous_situation", "harm", "sequence_of_events"):
                    out[k] = _paraphrase(out[k], 18)
            return out
        return None

    # 2) MAUDE fetch (single brand, randomized 70%)
    maude_brand = MAUDE_DEVICE.split(",")[0].strip() if MAUDE_DEVICE else "SIGMA SPECTRUM"
    use_maude = MAUDE_FETCH

    maude_events: List[Dict[str, Any]] = []
    maude_narratives: List[str] = []

    if use_maude:
        cache_key = f"maude::{maude_brand}::{MAUDE_LIMIT}"
        cached = _read_cache(cache_key)
        if cached:
            maude_events = cached.get("events", [])
            maude_narratives = cached.get("narratives", [])
        else:
            try:
                ev = _fetch_maude_events(maude_brand, MAUDE_LIMIT)
                maude_events = ev
                maude_narratives = _collect_maude_narratives(ev)
                _write_cache(cache_key, {"events": ev, "narratives": maude_narratives})
            except Exception as e:
                print({"maude_fetch_warning": str(e)})
                maude_events = []
                maude_narratives = []

        print({
            "maude_devices": [maude_brand],
            "maude_counts": {maude_brand: len(maude_events)},
            "maude_total": len(maude_events)
        })
        print({"maude_events_total": len(maude_events), "maude_narratives": len(maude_narratives)})

    rows: List[Dict[str, Any]] = []

    for r in requirements or []:
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "")
        rtxt = str(r.get("Requirements") or r.get("requirements") or "")
        risk_ids = _risks_for_requirement(rtxt)

        for risk in risk_ids:
            # Decide per-row if we use MAUDE based on rate (70%)
            use_row_maude = use_maude and (random.random() < MAUDE_SAMPLE_RATE) and bool(maude_narratives)

            if use_row_maude:
                # use narrative-backed
                srow = _row_from_sources(rid, risk, maude_narratives)
            else:
                # try RAG hint; if none, make short generic + paraphrase
                hint = _rag_hint(risk)
                if hint:
                    srow = hint
                else:
                    srow = _row_from_sources(rid, risk, [])

            row = {
                "requirement_id": rid or "REQ-000",
                "risk_id": f"HA-{random.randint(1000,9999)}",
                "risk_to_health": srow.get("risk_to_health", risk),
                "hazard": srow.get("hazard", "Device malfunction"),
                "hazardous_situation": srow.get("hazardous_situation", "Patient exposed to device fault"),
                "harm": srow.get("harm", "Adverse physiological effects"),
                "sequence_of_events": srow.get("sequence_of_events", "Device issue led to patient exposure"),
                # keep same severity table / p0/p1/poh
                "severity_of_harm": "3",
                "p0": "Medium",
                "p1": "Medium",
                "poh": "Low",
                "risk_index": "Medium",
                "risk_control": "The pump must maintain flow precision kept within ±5%.",
            }

            # micro-clean: avoid "string" placeholders
            for k in ("hazard", "hazardous_situation", "harm", "sequence_of_events"):
                if not row[k] or row[k].strip().lower() in {"string", "tbd"}:
                    row[k] = "TBD"

            rows.append(row)

        # stop if QUICK_LIMIT would be exceeded by many risks
        if QUICK_LIMIT and len(rows) >= QUICK_LIMIT:
            rows = rows[:QUICK_LIMIT]
            break

    return rows
