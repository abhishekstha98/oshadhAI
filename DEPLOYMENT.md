# Deploying Antigravity to Cloud

This guide explains how to deploy the Antigravity system as a web API so your colleagues can access it.

**Recommended Option**: **Hugging Face Spaces** (Free, Easy, designed for ML)

---

## Prerequisite: API Files Created
I have already created the necessary files in your repository:
1.  `src/api.py` (FastAPI Wrapper)
2.  `requirements.txt` (Dependencies)
3.  `Dockerfile` (Container instructions)

---

## Option 1: Hugging Face Spaces (FREE & FASTEST) 
**Best for**: Internal tools, demos, free hosting.

1.  **Create Account**: Go to [huggingface.co](https://huggingface.co) and sign up.
2.  **Create Space**:
    *   Click **New Space**.
    *   Name: `antigravity-api`.
    *   SDK: Select **Docker**.
    *   Privacy: **Public** or **Private** (Private requires a token to access).
    *   Create Space.
3.  **Upload Code**:
    *   Use the web interface or git to upload your entire repository context.
    *   CRITICAL: Ensure `data/herb_combinator.duckdb` and `checkpoints/` are uploaded (Git LFS might be needed for large files).
4.  **Wait for Build**:
    *   Hugging Face will automatically find the `Dockerfile` and build the container.
    *   Once "Running", your API is live at `https://huggingface.co/spaces/USERNAME/antigravity-api`.
5.  **Usage**:
    *   Colleague visits: `https://.../docs` to see the interactive API UI.
    *   Web App calls: `POST https://.../score`

---

## Option 2: Google Cloud Run (SCALABLE)
**Best for**: Production web apps, autoscaling to zero (pay only when used).

1.  **Install Google Cloud SDK**.
2.  **Build & Push**:
    ```bash
    # Set project
    gcloud config set project YOUR_PROJECT_ID
    
    # Build container
    gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/antigravity
    ```
3.  **Deploy**:
    ```bash
    gcloud run deploy antigravity \
      --image gcr.io/YOUR_PROJECT_ID/antigravity \
      --platform managed \
      --memory 2Gi \
      --allow-unauthenticated
    ```
4.  **Result**: You get a URL like `https://antigravity-xyz-uc.a.run.app`.

---

## Option 3: Render (ALTERNATIVE)
**Best for**: Simple setup if you prefer GitHub integration.

1.  Link your GitHub repo to Render.
2.  Select **Web Service**.
3.  Runtime: **Docker**.
4.  Plan: **Free** (Note: Free tier spins down after inactivity, might be slow to wake up).

---

## Testing the Deployed API

Once deployed, your colleague can test it via `curl`:

```bash
curl -X 'POST' \
  'https://YOUR-DEPLOYMENT-URL/score' \
  -H 'Content-Type: application/json' \
  -d '{
  "smiles_list": [
    "CCO",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
  ],
  "weights": ["LOW", "HIGH"]
}'
```
