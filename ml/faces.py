"""Fast, low-cost frame sampling + face cropping for VeritasAI.

* Frames are sampled uniformly (chronological order kept).
* Faces are found with OpenCV YuNet (~230 KB ONNX, a few ms per frame on CPU).
  If the YuNet file is missing we fall back to the Haar cascade bundled with OpenCV.
* Detection runs on a down-scaled copy of the frame (cheap); the crop is taken
  from the full-resolution frame (keeps the high-frequency artefacts we need).
* A simple IoU tracker keeps the SAME face across frames.
"""
from pathlib import Path
import hashlib
import urllib.request

import cv2
import numpy as np

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = [
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "weights"


def download_yunet(target_dir=DEFAULT_MODEL_DIR):
    target_dir = Path(target_dir); target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / YUNET_FILE
    if path.exists() and path.stat().st_size > 100_000:
        return path
    for url in YUNET_URLS:
        try:
            urllib.request.urlretrieve(url, path)
            if hashlib.sha256(path.read_bytes()).hexdigest() == YUNET_SHA256:
                return path
        except Exception:
            pass
    raise RuntimeError("Could not download YuNet; Haar fallback will be used.")


def _iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0])); ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


class FaceCropper:
    def __init__(self, size=224, margin=1.3, det_width=480, score=0.6, model_path=None):
        self.size, self.margin, self.det_width = size, margin, det_width
        self.kind = "haar"
        path = Path(model_path) if model_path else DEFAULT_MODEL_DIR / YUNET_FILE
        if path.exists():
            try:
                self.det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score, 0.3, 20)
                self.kind = "yunet"
            except Exception:
                pass
        if self.kind == "haar":
            self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame):
        """Return list of (x, y, w, h) boxes in full-resolution coordinates."""
        h, w = frame.shape[:2]
        s = min(1.0, self.det_width / w)
        small = cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
        if self.kind == "yunet":
            self.det.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self.det.detect(small)
            boxes = [] if faces is None else [f[:4] for f in faces]
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            boxes = self.det.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
        return [tuple(float(v) / s for v in b) for b in boxes]

    def crop(self, frame, box):
        """Square crop around the face with margin; pads at borders. Returns RGB uint8."""
        x, y, w, h = box
        cx, cy, side = x + w / 2, y + h / 2, max(w, h) * self.margin
        x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x2, y2 = x1 + int(round(side)), y1 + int(round(side))
        H, W = frame.shape[:2]
        pad = max(0, -x1, -y1, x2 - W, y2 - H)
        if pad:
            frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REFLECT)
            x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
        face = cv2.resize(frame[y1:y2, x1:x2], (self.size, self.size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


def read_frames(video_path, num_frames=16):
    """Uniformly sample frames. Uses grab() (no colour conversion) between targets = fast."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); raise ValueError(f"no frames: {video_path}")
    targets = np.unique(np.linspace(0, total - 1, min(num_frames, total)).astype(int))
    frames, idx = [], []
    if total <= 900:                       # short clip: sequential grab is fastest
        want, i = set(targets.tolist()), 0
        while i <= targets[-1]:
            if not cap.grab():
                break
            if i in want:
                ok, f = cap.retrieve()
                if ok: frames.append(f); idx.append(i)
            i += 1
    else:                                  # long video: seek
        for t in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, f = cap.read()
            if ok: frames.append(f); idx.append(int(t))
    cap.release()
    if not frames:
        raise ValueError(f"no readable frames: {video_path}")
    return frames, idx


def faces_from_video(video_path, cropper, num_frames=16):
    """Returns (faces uint8 [T,S,S,3] RGB, face_ratio in [0,1], frame indices)."""
    frames, idx = read_frames(video_path, num_frames)
    boxes, prev, found = [], None, 0
    for f in frames:
        cand = cropper.detect(f)
        if cand:
            found += 1
            if prev is None:
                prev = max(cand, key=lambda b: b[2] * b[3])
            else:
                best = max(cand, key=lambda b: _iou(b, prev))
                prev = best if _iou(best, prev) > 0.1 else prev
        boxes.append(prev)
    if found == 0:                         # no face at all: centre crop
        h, w = frames[0].shape[:2]; s = min(h, w) / cropper.margin
        boxes = [((w - s) / 2, (h - s) / 2, s, s)] * len(frames)
    else:                                  # back-fill frames before first detection
        first = next(b for b in boxes if b is not None)
        boxes = [b if b is not None else first for b in boxes]
    faces = np.stack([cropper.crop(f, b) for f, b in zip(frames, boxes)])
    # always return exactly num_frames (repeat last frame for very short clips)
    if len(faces) < num_frames:
        faces = np.concatenate([faces, np.repeat(faces[-1:], num_frames - len(faces), 0)])
    return faces, found / len(frames), idx


def save_strip(faces, path, quality=95):
    """Store T faces as ONE horizontal JPEG strip (≈10x smaller than .npy)."""
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(list(faces), axis=1), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])


def jpeg_roundtrip(faces, quality=95):
    """Inference must see the same JPEG-q95 statistics the model was trained on (strips are JPEG)."""
    bgr = cv2.cvtColor(np.concatenate(list(faces), axis=1), cv2.COLOR_RGB2BGR)
    dec = cv2.imdecode(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)
    S = faces.shape[1]
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).reshape(S, -1, S, 3).transpose(1, 0, 2, 3)


def load_strip(path, size=224):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.reshape(size, -1, size, 3).transpose(1, 0, 2, 3)   # [T,S,S,3]


def freq_features(faces, bins=64):
    """Cheap frequency stream: azimuthally averaged log power spectrum per frame.
    GAN/face-swap up-sampling leaves periodic high-frequency traces that show up here."""
    T, S = faces.shape[0], faces.shape[1]
    gray = faces.astype(np.float32).mean(-1)
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray), axes=(-2, -1))))
    yy, xx = np.indices((S, S)); r = np.hypot(yy - S / 2, xx - S / 2)
    r = np.minimum((r / r.max() * bins).astype(int), bins - 1).ravel()
    counts = np.bincount(r, minlength=bins)
    out = np.stack([np.bincount(r, weights=s.ravel(), minlength=bins) / counts for s in spec])
    out = (out - out.mean(1, keepdims=True)) / (out.std(1, keepdims=True) + 1e-6)
    return out.astype(np.float32)                                   # [T,bins]
