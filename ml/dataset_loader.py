from pathlib import Path
import csv
import torch
from torch.utils.data import Dataset
from .preprocess import extract_frames

class VideoManifestDataset(Dataset):
    def __init__(self, manifest, num_frames=20):
        self.rows = []
        with open(manifest, newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("label") not in {"0", "1"}:
                    raise ValueError("Manifest labels must be 0 (REAL) or 1 (FAKE).")
                if not Path(row["video_path"]).exists():
                    raise FileNotFoundError(row["video_path"])
                self.rows.append(row)
        if not self.rows:
            raise ValueError("The manifest contains no labelled videos.")
        self.num_frames = num_frames
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        sampled = extract_frames(row["video_path"], self.num_frames)
        return sampled.frames, torch.tensor(float(row["label"]), dtype=torch.float32), row.get("group", "")

