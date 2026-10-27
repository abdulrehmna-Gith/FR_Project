import io
import numpy as np
import tritonclient.http as httpclient
from PIL import ImageDraw


def nms(boxes, scores, thr=0.4):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < thr]
    return keep


def detect_faces(triton, img):
    """Send image to Triton retinaface, return list of [x1, y1, x2, y2]."""
    W, H = img.size
    arr = np.asarray(img.resize((640, 640)), dtype=np.float32)
    arr = ((arr - 127.5) / 128.0).transpose(2, 0, 1)[None]

    inp = httpclient.InferInput("input.1", list(arr.shape), "FP32")
    inp.set_data_from_numpy(arr)
    res = triton.infer("retinaface", [inp])

    all_boxes, all_scores = [], []
    for stride, s_name, b_name in [(8, "448", "451"), (16, "471", "474"), (32, "494", "497")]:
        scores = res.as_numpy(s_name).reshape(-1)
        bbox = res.as_numpy(b_name) * stride
        n = 640 // stride
        xs, ys = np.meshgrid(np.arange(n), np.arange(n))
        c = np.repeat(np.stack([xs, ys], -1).reshape(-1, 2) * stride, 2, axis=0)
        boxes = np.stack([c[:, 0] - bbox[:, 0], c[:, 1] - bbox[:, 1],
                          c[:, 0] + bbox[:, 2], c[:, 1] + bbox[:, 3]], axis=1)
        mask = scores > 0.5
        all_boxes.append(boxes[mask])
        all_scores.append(scores[mask])

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    if len(boxes) == 0:
        return []
    scale = [W / 640, H / 640, W / 640, H / 640]
    return [(boxes[i] * scale).tolist() for i in nms(boxes, scores)]


def draw_boxes(img, boxes):
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle(b, outline="lime", width=4)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()
