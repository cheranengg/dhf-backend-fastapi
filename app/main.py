from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os

# Import model wrappers (you've wired dvp_infer.py and tm_infer.py already)
from app.models import ha_infer, dvp_infer, tm_infer

# ---------------- Security ----------------
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "dev-token")

def require_auth(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    if token != BACKEND_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

# ---------------- App ----------------
app = FastAPI(title="DHF Backend", version="1.0")

# CORS: allow your Streamlit app origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Streamlit domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Schemas ----------------
class Requirement(BaseModel):
    **kwargs: Any

class HARsp(BaseModel):
    ha: List[Dict[str, Any]]

class DVPRsp(BaseModel):
    dvp: List[Dict[str, Any]]

class TMRsp(BaseModel):
    trace_matrix: List[Dict[str, Any]]

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/hazard-analysis", response_model=HARsp)
def hazard_analysis(payload: Dict[str, Any], _=Depends(require_auth)):
    reqs = payload.get("requirements", [])
    if not isinstance(reqs, list):
        raise HTTPException(400, "`requirements` must be a list")
    # Your HA predictor should accept list[dict] and return list[dict]
    ha_rows = ha_infer.ha_predict(reqs)
    return {"ha": ha_rows}

@app.post("/dvp", response_model=DVPRsp)
def dvp(payload: Dict[str, Any], _=Depends(require_auth)):
    reqs = payload.get("requirements", [])
    ha = payload.get("ha", [])
    if not isinstance(reqs, list):
        raise HTTPException(400, "`requirements` must be a list")
    dvp_rows = dvp_infer.dvp_predict(reqs, ha)
    return {"dvp": dvp_rows}

@app.post("/trace-matrix", response_model=TMRsp)
def trace_matrix(payload: Dict[str, Any], _=Depends(require_auth)):
    reqs = payload.get("requirements", [])
    ha = payload.get("ha", [])
    dvp_rows = payload.get("dvp", [])
    if not isinstance(reqs, list):
        raise HTTPException(400, "`requirements` must be a list")
    tm_rows = tm_infer.tm_predict(reqs, ha, dvp_rows)
    return {"trace_matrix": tm_rows}
