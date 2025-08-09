import os, json, re, requests
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# ---------- Config ----------
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
LORA_HA_DIR = os.getenv("LORA_HA_DIR", "/models/mistral_finetuned_Hazard_Analysis")
LOAD_WEB = os.getenv("LOAD_WEB_SOURCES", "0") == "1"  # optional heavy fetch

_manual_urls = [
    "https://adeptomed.com/wp-content/uploads/2020/05/Sigma-Spectrum-operators-manual.pdf",
    "https://www.baxter.com/sites/g/files/ebysai746/files/2019-01/ALARIS-system.pdf",
    "https://www.ardusmedical.com/wp-content/uploads/2013/09/Baxter-Colleague-3.pdf",
    "https://infusystem.com/images/catalog_manuals/English/Pole_Mounted_Pump_Manuals/Plum_360_15.2_System_Operators_Manual_02_2020-06.pdf",
    "https://adeptomed.com/wp-content/uploads/2020/05/smiths-medical-cadd-solis-operators-manual.pdf",
]
_standard_urls = [
    "https://en.wikipedia.org/wiki/IEC_60601",
    "https://en.wikipedia.org/wiki/ISO_80369",
    "https://en.wikipedia.org/wiki/ISO_10993",
    "https://en.wikipedia.org/wiki/ISO_11135",
    "https://en.wikipedia.org/wiki/ISO_15223",
    "https://en.wikipedia.org/wiki/IEEE_11073",
]

# ---------- Globals (lazy-loaded) ----------
_tokenizer = None
_model = None
_llm_pipe = None
_llm = None
_vectorstore = None
_vectorstore_reqs = None

# ---------- Helper: MAUDE query ----------
def _query_maude(device_name: str, limit: int = 3) -> List[Dict[str, Any]]:
    try:
        r = requests.get("https://api.fda.gov/device/event.json", params={"search": f"device.brand_name:{device_name}", "limit": limit}, timeout=20)
        data = r.json()
        res = []
        for item in data.get("results", []):
            res.append({
                "date": item.get("date_of_event"),
                "event": item.get("event_description", "No description"),
                "type": item.get("event_type", "Unknown"),
            })
        return res or [{"event": "No reports found"}]
    except Exception as e:
        return [{"error": str(e)}]

# ---------- Guardrails ----------
_severity_map = {"negligible":1, "minor":2, "moderate":3, "serious":4, "critical":5}

def _calculate_risk_fields(parsed: Dict[str, Any]):
    sev = str(parsed.get("Severity of Harm", "Moderate")).lower()
    severity = _severity_map.get(sev, 3)
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

# ---------- JSON extractor ----------
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

# ---------- Loader ----------

def load_ha_pipeline():
    global _tokenizer, _model, _llm_pipe, _vectorstore, _vectorstore_reqs
    if _model is not None:
        return

    # Load base + LoRA
    _tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
    base = AutoModelForCausalLM.from_pretrained(MISTRAL_MODEL, torch_dtype="auto", device_map="auto")
    _model = PeftModel.from_pretrained(base, LORA_HA_DIR)

    _llm_pipe = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        max_length=2048,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        truncation=True,
    )

    # Build background knowledge vectorstore (optional web fetch)
    docs: List[Document] = []
    if LOAD_WEB:
        from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
        # Manuals
        for url in _manual_urls:
            try:
                fn = url.split("/")[-1]
                fp = f"/tmp/{fn}"
                r = requests.get(url, timeout=60)
                with open(fp, "wb") as f:
                    f.write(r.content)
                ld = PyPDFLoader(fp)
                docs.extend(ld.load())
            except Exception:
                pass
        # Standards summaries (Wikipedia)
        try:
            wloader = WebBaseLoader(_standard_urls)
            docs.extend(wloader.load())
        except Exception:
            pass
        # MAUDE → docs
        from langchain.docstore.document import Document as LCDocument
        maude = _query_maude("SIGMA SPECTRUM", limit=5)
        for m in maude:
            if "error" in m: continue
            docs.append(LCDocument(page_content=f"Date: {m['date']}\nEvent: {m['event']}\nType: {m['type']}"))

    # Split + embed background docs
    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vectorstore = FAISS.from_documents(chunks, embeddings)

    # Placeholder reqs store: we build per request in ha_predict()

# ---------- Inference ----------

_default_risks = [
    "Air Embolism", "Allergic response", "Infection", "Overdose", "Underdose",
    "Delay of therapy", "Environmental Hazard", "Incorrect Therapy", "Trauma", "Particulate",
]

_prompt_template = (
    "Provide a hazard analysis entry for {risk} in infusion pumps.\n"
    "Return ONLY valid JSON in this format:\n"
    "{{\n"
    "  \"Hazard\": \"...\",\n"
    "  \"Hazardous Situation\": \"...\",\n"
    "  \"Harm\": \"...\",\n"
    "  \"Sequence of Events\": \"...\",\n"
    "  \"Severity of Harm\": \"Moderate|Serious|Critical|...\",\n"
    "  \"P0\": \"Very Low|Low|Medium|High|Very High\",\n"
    "  \"P1\": \"Very Low|Low|Medium|High|Very High\"\n"
    "}}"
)


def ha_predict(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate HA rows tied to each requirement.
    For each requirement we generate entries across a curated set of Risks to Health,
    apply guardrails, and pull a candidate Risk Control by retrieving the closest
    requirement text (k=1).
    """
    load_ha_pipeline()

    # Build a reqs retriever per request (so Risk Control can reference nearest req)
    req_docs = [Document(page_content=r.get("Requirements", ""), metadata={"Requirement ID": r.get("Requirement ID", "")}) for r in requirements]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    req_store = FAISS.from_documents(req_docs, embeddings) if req_docs else None

    qa = None
    if _vectorstore is not None:
        # Use a basic retrieval chain if background store exists
        qa = RetrievalQA.from_chain_type(
            llm=None,  # we'll call pipeline manually below; RetrievalQA used only for retriever
            retriever=_vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type="stuff",
        )

    out: List[Dict[str, Any]] = []
    for r in requirements:
        rid = r.get("Requirement ID") or ""
        rtext = r.get("Requirements") or ""
        for risk in _default_risks:
            prompt = _prompt_template.format(risk=risk)
            # Generate text via HF pipeline
            gen = _llm_pipe(prompt)[0]["generated_text"]
            parsed = _extract_json(gen) or {}

            # Guardrails
            severity, p0, p1, poh, risk_index = _calculate_risk_fields(parsed)

            # Risk Control via nearest requirement
            control = "Refer to IEC 60601, ISO 14971, MAUDE reports"
            if req_store is not None:
                hits = req_store.similarity_search((parsed.get("Hazardous Situation", "") + " " + parsed.get("Harm", "")).strip() or rtext, k=1)
                if hits:
                    control = f"{hits[0].page_content} (Ref: {hits[0].metadata.get('Requirement ID','')})"

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
