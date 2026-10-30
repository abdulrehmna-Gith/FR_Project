# Face Recognition with NVIDIA Triton, Qdrant and FastAPI

This project recognizes a person from a photo. It is made for beginners who want to learn how to deploy AI models on NVIDIA Triton Inference Server.

## What you will learn

- How to deploy models on Triton
- How an ensemble joins many models into one call
- How Qdrant stores and searches face embeddings
- How to run the project on Kubernetes
- How Argo CD deploys it from Git (GitOps)
- How GitHub Actions tracks every change in the repo

## Words you should know

| Word | Simple meaning |
|---|---|
| Inference | Using a trained model on a new photo to get a result |
| Embedding | A list of 512 numbers that describes a face |
| Ensemble | A pipeline in Triton that runs many models in one call |
| Vector database | A database that finds the closest embedding (Qdrant) |
| Kubernetes | A system that runs and manages containers |
| Argo CD | A tool that keeps the cluster the same as the YAML in Git |
| GitHub Actions | A tool on GitHub that runs a job every time you push |

## How it works

```
Photo -> FastAPI -> Triton -> embedding (512 numbers) -> Qdrant -> name
```

1. You send a photo to FastAPI.
2. FastAPI sends it to Triton. Triton finds the face, aligns it and makes the embedding.
3. Qdrant saves the embedding, or finds the closest saved one.

Inside Triton, one call runs four steps:

```
preprocess -> face detector -> postprocess (align + crop) -> ArcFace
```

## How the project is delivered

```
git push -> GitHub Actions (lists what changed) -> Argo CD (updates the cluster)
```

## What you need before you start

- Linux with **Docker** installed
- **Python 3** and **git** installed
- About **30 GB free disk space** (the Triton image alone is about 20 GB)
- You need **3 terminals** open: one for Triton, one for Qdrant, one for FastAPI
- For the Kubernetes part: a running cluster and **kubectl**

## Project structure

```
FR_Project/
├── app.py                  FastAPI app
├── face_utils.py           helper code for /detect
├── README.md
├── model_repository/
│   ├── arcface/            ArcFace model (you add model.onnx)
│   ├── retinaface/         face detector (you add model.onnx)
│   ├── preprocess/         Python step before the detector
│   ├── postprocess/        Python step after the detector (align + crop)
│   └── face_pipeline/      the ensemble that connects all steps
├── k8s/                    Kubernetes YAML (Deployments and Services)
├── argocd/                 Argo CD Application YAML
└── .github/workflows/      GitHub Actions (change tracking)
```

---

# Step by step

## Step 1: Download the project

```bash
git clone https://github.com/abdulrehmna-Gith/FR_Project.git
cd FR_Project
```

You should now be inside the `FR_Project` folder. Check with `ls`.

## Step 2: Create a Python environment

This keeps the packages for this project separate from your system.

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line.
Run `source venv/bin/activate` again every time you open a new terminal.

## Step 3: Install the Python packages

```bash
pip install fastapi uvicorn python-multipart pillow numpy qdrant-client "tritonclient[http]"
```

## Step 4: Download the face models

The models are large, so they are not stored in this repo. We download them from InsightFace.

Install the tool:

```bash
pip install insightface onnxruntime
```

Download the models (this can take a few minutes):

```bash
python3 -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')"
```

Check that the files are there:

```bash
ls ~/.insightface/models/buffalo_l/
```

You need two files from this list:

| File | What it is |
|---|---|
| `w600k_r50.onnx` | ArcFace (makes the embedding) |
| `det_10g.onnx` | Face detector |

## Step 5: Put the models in the Triton folders

Triton reads models from folders. The folder name is the model name, `1` is the version, and the file must be called `model.onnx`.

```bash
mkdir -p model_repository/arcface/1
mkdir -p model_repository/retinaface/1
mkdir -p model_repository/face_pipeline/1

cp ~/.insightface/models/buffalo_l/w600k_r50.onnx model_repository/arcface/1/model.onnx
cp ~/.insightface/models/buffalo_l/det_10g.onnx   model_repository/retinaface/1/model.onnx
```

Check the files:

```bash
ls -lh model_repository/arcface/1 model_repository/retinaface/1
```

You should see `model.onnx` in both folders (about 167 MB and 17 MB).

Your folders should now look like this:

```
model_repository/
├── arcface/
│   ├── config.pbtxt
│   └── 1/model.onnx
├── retinaface/
│   ├── config.pbtxt
│   └── 1/model.onnx
├── preprocess/
│   ├── config.pbtxt
│   └── 1/model.py
├── postprocess/
│   ├── config.pbtxt
│   └── 1/model.py
└── face_pipeline/
    ├── config.pbtxt
    └── 1/          (empty folder, this is normal)
```

## Step 6: Start Triton (Terminal 1)

```bash
docker run --rm --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${PWD}/model_repository":/models \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  tritonserver --model-repository=/models
```

The first run downloads a large image, so it takes time.

Wait until you see a table like this:

```
| arcface       | 1 | READY |
| face_pipeline | 1 | READY |
| postprocess   | 1 | READY |
| preprocess    | 1 | READY |
| retinaface    | 1 | READY |
```

If any line says `UNAVAILABLE`, go to **Common errors** at the bottom.

Leave this terminal open. Triton keeps running here.

Open a **new terminal** and check the pipeline:

```bash
curl -s localhost:8000/v2/models/face_pipeline/ready -o /dev/null -w "%{http_code}\n"
```

`200` means ready.

## Step 7: Start Qdrant (Terminal 2)

Qdrant is the database that stores the embeddings.

```bash
cd FR_Project
docker run --rm -p 6333:6333 -v "$PWD/qdrant_data":/qdrant/storage qdrant/qdrant
```

The `-v` part keeps your saved faces even after you stop Qdrant.

Check it in your browser: http://localhost:6333/dashboard

Leave this terminal open.

## Step 8: Start FastAPI (Terminal 3)

```bash
cd FR_Project
source venv/bin/activate
uvicorn app:app --reload --port 8080
```

You should see:

```
Uvicorn running on http://127.0.0.1:8080
Application startup complete.
```

Open the test page: http://localhost:8080/docs

You will see 3 blocks: `/detect`, `/embed` and `/recognize`.

## Step 9: Test the project

Use a clear, front-facing photo of yourself. Use only your own photos, or photos of people who agreed.

### 9.1 Detect a face

1. Click **POST /detect**, then **Try it out**.
2. Choose a photo and click **Execute**.
3. You should see your photo with a **green box** on the face.

### 9.2 Save a face

1. Click **POST /embed**, then **Try it out**.
2. Type a name, for example `abdi`.
3. Choose your photo and click **Execute**.
4. You should see `"saved_in_qdrant": true` and 512 numbers.

### 9.3 See the saved face in Qdrant

1. Open http://localhost:6333/dashboard
2. Click **Collections**, then **faces**, then **Points**.
3. You should see a point with `{"name": "abdi"}` and a vector of length 512.

### 9.4 Recognize a face

1. Click **POST /recognize**, then **Try it out**.
2. Choose a **different photo** of the same person and click **Execute**.
3. You should see the name and a score:

```json
{ "name": "abdi", "confidence": 0.62 }
```

Now try a photo of a different person. You should see:

```json
{ "name": "Unknown", "confidence": 0.04 }
```

## How to read the score

| Score | Meaning |
|---|---|
| About 1.0 | Same photo |
| 0.4 to 0.7 | Same person, different photo |
| About 0 | Different person |

The code says a score of 0.5 or more is a match. You can change this number in `app.py`.

## How to stop everything

Press `Ctrl + C` in each terminal. Stop FastAPI first, then Qdrant, then Triton.

## How to delete all saved faces

```bash
curl -X DELETE localhost:6333/collections/faces
```

The collection is created again the next time you use `/embed`.

## See the inference happening

After you call `/embed`, run:

```bash
curl -s localhost:8000/v2/models/arcface/stats
```

The `inference_count` goes up every time a photo goes through the model.

---

# Run on Kubernetes

The `k8s/` folder has a Deployment and a Service for each part of the project.

| Component | Service port | NodePort |
|---|---|---|
| `face-ui` | 80 | 30090 |
| `face-app` | 8080 | 30080 |
| `triton` | 8000, 8001 | 30000, 30001 |
| `qdrant` | 6333 | 30333 |

Deploy and check:

```bash
kubectl apply -f k8s/
kubectl get all
```

All 4 pods should show `1/1 Running`. Triton can take a few minutes to start.

Open the UI at `http://<node-ip>:30090`.

Useful commands:

```bash
kubectl logs deploy/face-app           # see app logs
kubectl describe pod <pod-name>        # see why a pod fails
kubectl rollout restart deploy/face-app
```

---

# Deploy with Argo CD

Argo CD runs inside the cluster, watches this GitHub repo, and keeps the cluster the same as the YAML in `k8s/`.

```
git push -> Argo CD sees the change -> Argo CD updates the cluster
```

In this project one Argo CD application, called **`face-recognition`**, manages all four components (face-ui, face-app, qdrant and triton). Auto sync is on.

## Install Argo CD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl get pods -n argocd
```

Install the CLI (download to `/tmp` to avoid file conflicts):

```bash
cd /tmp
curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
sudo install -m 555 argocd-linux-amd64 /usr/local/bin/argocd
rm argocd-linux-amd64
cd ~/Desktop/FR_Project
```

## Log in

Terminal 1 (leave it open):

```bash
kubectl port-forward svc/argocd-server -n argocd 8443:443
```

Terminal 2:

```bash
argocd admin initial-password -n argocd
argocd login localhost:8443 --username admin --insecure
```

The UI is at https://localhost:8443 (user `admin`).

## Create the app

The Application file lives in `argocd/`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: face-recognition
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/abdulrehmna-Gith/FR_Project.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

Apply and check:

```bash
kubectl apply -f argocd/
kubectl get applications -n argocd
argocd app get face-recognition
```

You should see `Synced` and `Healthy`.

## Main commands

| Task | Command |
|---|---|
| List apps | `argocd app list` |
| See status | `argocd app get face-recognition` |
| See differences | `argocd app diff face-recognition` |
| Deploy now | `argocd app sync face-recognition` |
| See history | `argocd app history face-recognition` |

In the UI, open the `face-recognition` app and click **History and Rollback** to see every sync.

Argo CD only deploys changes under its `path` (`k8s/`). A change to the README or the workflow keeps the app `Synced` and adds no new history row.

## Give a teammate access to only the app

Do not share your admin kubeconfig. Create an Argo CD account that can only see and sync this app:

```bash
kubectl edit configmap argocd-cm -n argocd
#   data:
#     accounts.teamlead: login

kubectl edit configmap argocd-rbac-cm -n argocd
#   data:
#     policy.csv: |
#       p, teamlead, applications, get, default/face-recognition, allow
#       p, teamlead, applications, sync, default/face-recognition, allow

argocd account update-password --account teamlead --new-password <password>
```

To limit a teammate to only `face-ui`, create a separate Argo CD application for `k8s/face-ui` and give the account access to that one.

---

# Track changes with GitHub Actions

A GitHub Action in `.github/workflows/changes.yaml` runs on every push to `main` and on every pull request. It shows who made the change, which files changed and how many lines changed.

```
git push -> GitHub Actions lists the changed files -> Argo CD syncs the cluster
```

See the results:

1. Open the repo on GitHub and click the **Actions** tab.
2. Click the latest `track-changes` run.
3. Open the `changes` job, or read the **Summary** at the top of the run.

| Letter | Meaning |
|---|---|
| `A` | File added |
| `M` | File modified |
| `D` | File deleted |
| `R` | File renamed |

For now this workflow only reports changes. It does not build images or touch the cluster. Argo CD deploys whatever changes under `k8s/`.

## Daily workflow

```bash
git add .
git commit -m "Describe your change"
git push origin main
```

1. GitHub Actions starts and lists the changed files (Actions tab).
2. If the change is under `k8s/`, Argo CD syncs it and adds a row in **History and Rollback**.
3. Check the result with `kubectl get all` or the Argo CD UI.

---

# Common errors and fixes

| Error | Reason | Fix |
|---|---|---|
| `model does not support batching` | Wrong batch setting | Set `max_batch_size: 0` in that model's `config.pbtxt` |
| `failed to stat file .../model.onnx` | Model file is missing | Repeat Step 5 |
| `Python model file not found ... model.py` | File has a wrong name, like `mode1.py` | Rename it to `model.py` |
| `Stub process is not healthy` | Not enough shared memory, or `model.py` is missing | Start Triton with `--shm-size=1g` and check the file name |
| `ModuleNotFoundError: face_utils` | You are not in the project folder | Run `cd FR_Project` |
| `NameError: np is not defined` | Missing import | Add `import numpy as np` at the top of `app.py` |
| `Method Not Allowed` in the browser | A browser address bar sends GET, but the endpoint needs POST | Use the `/docs` page |
| `Connection refused` | Triton or Qdrant is not running | Check Steps 6 and 7 |
| `/recognize` always says `Unknown` for the same person | Old saved faces from older code | Delete the faces (see above) and save again with `/embed` |
| Triton exits after `Ctrl + C` with a long wait | FastAPI still has a connection open | Stop FastAPI first |
| `argocd: command not found` | CLI is not installed | Install it (see Deploy with Argo CD) |
| `Argo CD server address unspecified` | You are not logged in | Run `argocd login` |
| `application not found` | Wrong app name, or the app was never created | Use `face-recognition`, or run `kubectl apply -f argocd/` |
| App is `Synced` but History is empty | No sync has run yet | Click **Sync** in the UI, or push a change under `k8s/` |
| `curl: (23) client returned ERROR on write` | curl cannot write the file | Download to `/tmp` |
| Pod in `ImagePullBackOff` or `CrashLoopBackOff` | Wrong image, or the app crashes | `kubectl describe pod <pod>` and `kubectl logs <pod>` |
| No run in the GitHub **Actions** tab | Workflow file is in the wrong place | It must be `.github/workflows/changes.yaml` on `main` |
| `git push` is rejected | GitHub has newer commits | `git pull --rebase origin main`, then push again |

---

## Notes

- This project runs on **CPU**. For GPU, remove `KIND_CPU` from the configs and use the NVIDIA Container Toolkit.
- Never push a kubeconfig, token or password to Git.
- Never type a token or password directly in a command. It is saved in your shell history.
