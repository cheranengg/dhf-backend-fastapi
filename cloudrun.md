# Deploy FastAPI (GPU) to Google Cloud Run

This guide deploys your DHF FastAPI backend (HA / DVP / TM) to **Google Cloud Run (GPU)**.

---

## 0) Prereqs

- Google Cloud project with **billing enabled**
- gcloud CLI: `gcloud auth login`
- Set project: `gcloud config set project <PROJECT_ID>`
- GPU quota in your region (e.g., `us-central1`)

Enable services:
```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  compute.googleapis.com secretmanager.googleapis.com storage.googleapis.com
