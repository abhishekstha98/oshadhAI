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

## Option 1: Hugging Face Spaces (RECOMMENDED)
**Best for**: Free hosting, Zero-Setup, Docker-based.

1.  **Create Space**:
    -   Go to [huggingface.co/spaces](https://huggingface.co/spaces) -> **Create new Space**.
    -   **SDK**: Select **Docker**.
    -   Name: `antigravity-api`.

2.  **Push Your Code**:
    Since we enabled **Lightweight Inference**, you do **NOT** need to upload the 1.2GB database. The system will use the optimized `data/inference/*.json.gz` files.
    
    Simply run:
    ```bash
    git add Dockerfile requirements.txt .dockerignore src/api.py
    git commit -m "Configure Docker Deployment"
    git push origin main
    ```
    *(Or drag-and-drop these files via the web interface)*.

3.  **Automatic Build**:
    -   Hugging Face will detect the `Dockerfile` and build the container.
    -   It will install dependencies from `requirements.txt`.
    -   The API will launch automatically (Status: **Running**).

4.  **Usage**:
    -   **API Docs**: `https://huggingface.co/spaces/USERNAME/antigravity-api/docs`
    -   **Score Endpoint**: `POST /score`


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
