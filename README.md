# Face Recognition with NVIDIA Triton, Qdrant and FastAPI

This project recognizes a person from a photo. It is made for beginners who want to learn how to deploy AI models on NVIDIA Triton Inference Server.

## What you will learn

- How to deploy models on Triton
- How an ensemble joins many models into one call
- How Qdrant stores and searches face embeddings

## Words you should know

| Word | Simple meaning |
|---|---|
| Inference | Using a trained model on a new photo to get a result |
| Embedding | A list of 512 numbers that describes a face |
| Ensemble | A pipeline in Triton that runs many models in one call |
| Vector database | A database that finds the closest embedding (Qdrant) |

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

## What you need before you start

- Linux with **Docker** installed
- **Python 3** and **git** installed
- About **30 GB free disk space** (the Triton image alone is about 20 GB)
- You need **3 terminals** open: one for Triton, one for Qdrant, one for FastAPI

## Project structure

```
FR_Project/
├── app.py                  FastAPI app
├── face_utils.py           helper code for /detect
├── README.md
└── model_repository/
    ├── arcface/            ArcFace model (you add model.onnx)
    ├── retinaface/         face detector (you add model.onnx)
    ├── preprocess/         Python step before the detector
    ├── postprocess/        Python step after the detector (align + crop)
    └── face_pipeline/      the ensemble that connects all steps
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

---

## Notes

- This project runs on **CPU**. For GPU, remove `KIND_CPU` from the configs and use the NVIDIA Container Toolkit.
-
