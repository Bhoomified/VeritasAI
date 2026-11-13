from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

class PreprocessingError(Exception): pass
@dataclass
class SampledFrames:
    frames: torch.Tensor
    indices: list
    faces_found: int

def sample_indices(total_frames, count):
    if total_frames <= 0 or count <= 0:
        return []
    return np.linspace(0, total_frames - 1, min(total_frames, count), dtype=int).tolist()

def _detector():
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade)
    if detector.empty():
        raise PreprocessingError("Face detector could not be loaded.")
    return detector

def crop_prominent_face(frame, detector=None):
    detector = detector or _detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
    if len(faces) == 0:
        return frame, False
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad_x, pad_y = int(w * .15), int(h * .15)
    height, width = frame.shape[:2]
    crop = frame[max(0, y-pad_y):min(height, y+h+pad_y), max(0, x-pad_x):min(width, x+w+pad_x)]
    return crop, True

def preprocess_frame(frame, detector=None):
    crop, found = crop_prominent_face(frame, detector)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(cv2.resize(rgb, (224, 224)))
    tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor, found

def extract_frames(video_path, count=20):
    path = Path(video_path)
    if not path.exists():
        raise PreprocessingError("Video file does not exist.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PreprocessingError("Video could not be opened or is unsupported.")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_indices(total, count)
    if not indices:
        capture.release()
        raise PreprocessingError("Video contains no readable frames.")
    detector = _detector()
    frames, actual, found = [], [], 0
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok and frame is not None:
            tensor, has_face = preprocess_frame(frame, detector)
            frames.append(tensor); actual.append(index); found += int(has_face)
    capture.release()
    if not frames:
        raise PreprocessingError("No readable frames were extracted.")
    return SampledFrames(torch.stack(frames), actual, found)

