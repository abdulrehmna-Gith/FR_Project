import io
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, Response
from PIL import Image
import tritonclient.http as httpclient
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from face_utils import detect_faces, draw_boxes

app = FastAPI()
triton = httpclient.InferenceServerClient("localhost:8000")
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION = "faces"


def get_embedding(img):
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))      # make the photo square
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    arr = np.asarray(canvas.resize((640, 640)), dtype=np.uint8)

    inp = httpclient.InferInput("IMAGE", list(arr.shape), "UINT8")
    inp.set_data_from_numpy(arr)
    res = triton.infer("face_pipeline", [inp])
    if float(res.as_numpy("FACE_SCORE")[0]) == 0:
        return None
    emb = res.as_numpy("EMBEDDING")[0]
    return emb / np.linalg.norm(emb)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    boxes = detect_faces(triton, img)
    return Response(content=draw_boxes(img, boxes), media_type="image/png",
                    headers={"X-Faces-Found": str(len(boxes))})


@app.post("/embed")
async def embed(name: str, file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    emb = get_embedding(img)
    if emb is None:
        return {"face_found": False}

    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            COLLECTION, vectors_config=VectorParams(size=512, distance=Distance.COSINE))

    qdrant.upsert(COLLECTION, points=[
        PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload={"name": name})])
    return {"face_found": True, "saved_in_qdrant": True, "name": name,
            "embedding_size": len(emb), "embedding": emb.tolist()}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        emb = get_embedding(img)
        if emb is None:
            return {"name": "No face found"}

        if not qdrant.collection_exists(COLLECTION):
            return {"error": "No faces saved yet. Use /embed with a name first."}

        hits = qdrant.query_points(COLLECTION, query=emb.tolist(), limit=1).points
        if not hits:
            return {"name": "Database is empty"}

        best = hits[0]
        if best.score >= 0.5:
            return {"name": best.payload["name"], "confidence": round(best.score, 3)}
        return {"name": "Unknown", "confidence": round(best.score, 3)}
    except Exception as e:
        return {"error": str(e)}
