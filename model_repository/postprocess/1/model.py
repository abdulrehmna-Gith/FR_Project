import numpy as np
import triton_python_backend_utils as pb_utils

# where ArcFace expects the 5 points in a 112x112 face
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def similarity_transform(src, dst):
    """Find scale, rotation, shift so that dst = scale * R @ src + t"""
    n = src.shape[0]
    sm, dm = src.mean(0), dst.mean(0)
    s, d = src - sm, dst - dm
    cov = d.T @ s / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.ones(2)
    if np.linalg.det(cov) < 0:
        D[-1] = -1
    R = U @ np.diag(D) @ Vt
    scale = (S * D).sum() / ((s ** 2).sum() / n)
    t = dm - scale * (R @ sm)
    return scale, R, t


def align_face(img, kps):
    scale, R, t = similarity_transform(kps.astype(np.float64), TEMPLATE.astype(np.float64))
    ys, xs = np.mgrid[0:112, 0:112]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    src = ((pts - t) @ R) / scale                      # where to read in the big image
    sx = np.clip(np.round(src[:, 0]).astype(int), 0, img.shape[1] - 1)
    sy = np.clip(np.round(src[:, 1]).astype(int), 0, img.shape[0] - 1)
    return img[sy, sx].reshape(112, 112, 3)


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            get = lambda n: pb_utils.get_input_tensor_by_name(request, n).as_numpy()
            img = get("IMAGE")

            all_scores, all_kps = [], []
            for stride in (8, 16, 32):
                scores = get(f"SCORE_{stride}").reshape(-1)
                kp = get(f"KPS_{stride}") * stride
                n = 640 // stride
                xs, ys = np.meshgrid(np.arange(n), np.arange(n))
                c = np.repeat(np.stack([xs, ys], -1).reshape(-1, 2) * stride, 2, axis=0)
                kps = np.stack([c[:, 0:1] + kp[:, 0::2], c[:, 1:2] + kp[:, 1::2]], axis=-1)
                keep = scores > 0.5
                all_scores.append(scores[keep])
                all_kps.append(kps[keep])
            scores = np.concatenate(all_scores)
            kps = np.concatenate(all_kps)

            face = np.zeros((1, 3, 112, 112), dtype=np.float32)
            best = 0.0
            if len(scores) > 0:
                i = int(scores.argmax())
                crop = align_face(img, kps[i])
                face = ((crop.astype(np.float32) - 127.5) / 127.5)
                face = face.transpose(2, 0, 1)[None].astype(np.float32)
                best = float(scores[i])

            out1 = pb_utils.Tensor("FACE", face)
            out2 = pb_utils.Tensor("FACE_SCORE", np.array([best], dtype=np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out1, out2]))
        return responses
