"""Step 1 – build manifests (identity-safe splits) and extract face strips ONCE.

  python -m ml.prepare manifest --layout ffpp    --root /data/FF++    --out data/ffpp
  python -m ml.prepare manifest --layout celebdf --root /data/CelebDF --out data/celebdf --test-only
  python -m ml.prepare manifest --layout dfdc    --root /data/dfdc    --out data/dfdc
  python -m ml.prepare manifest --layout folders --root /data/mine    --out data/mine   (real/ & fake/ sub-folders)
  python -m ml.prepare faces --csv data/ffpp/train.csv data/ffpp/val.csv data/ffpp/test.csv --out faces/ffpp --frames 16

Output of `faces`: faces/<name>/<uid>.jpg (16 face crops in one JPEG strip) + index CSVs.
Everything after this step reads only the small strips – videos are never decoded again.
"""
import argparse, csv, hashlib, json, os, random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _videos(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXT)


def rows_ffpp(root, comp="c23", methods=None):
    root = Path(root); rows = []
    for p in _videos(root / "original_sequences" / "youtube" / comp / "videos"):
        rows.append((p, 0, p.stem, "original"))
    for mdir in sorted((root / "manipulated_sequences").glob("*")):
        if methods and mdir.name not in methods:
            continue
        for p in _videos(mdir / comp / "videos"):          # e.g. 000_003.mp4 -> target identity 000
            rows.append((p, 1, p.stem.split("_")[0], mdir.name))
    return rows


def rows_celebdf(root):
    root = Path(root); rows = []
    for sub, lab in (("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)):
        for p in _videos(root / sub):
            grp = ("yt_" + p.stem) if sub == "YouTube-real" else p.stem.split("_")[0]
            rows.append((p, lab, grp, sub))
    test_list = root / "List_of_testing_videos.txt"
    official = None
    if test_list.exists():
        official = {line.split()[1] for line in test_list.read_text().splitlines() if line.strip()}
    return rows, official


def rows_dfdc(root):
    rows = []
    for meta in Path(root).rglob("metadata.json"):
        info = json.loads(meta.read_text())
        for name, m in info.items():
            p = meta.parent / name
            if p.exists():
                lab = int(m["label"] == "FAKE")
                rows.append((p, lab, (m.get("original") or name).rsplit(".", 1)[0], "dfdc"))
    return rows


def rows_folders(root):
    rows = []
    for sub, lab in (("real", 0), ("fake", 1)):
        for p in _videos(Path(root) / sub):
            rows.append((p, lab, p.stem.split("_")[0], sub))
    return rows


def group_split(rows, val=0.15, test=0.15, seed=42):
    """Split by identity/group so the same person never appears in two splits (no leakage)."""
    groups = sorted({r[2] for r in rows}); random.Random(seed).shuffle(groups)
    n = len(groups); nt, nv = int(n * test), int(n * val)
    split_of = {g: "test" for g in groups[:nt]}
    split_of.update({g: "val" for g in groups[nt:nt + nv]})
    return {s: [r for r in rows if split_of.get(r[2], "train") == s] for s in ("train", "val", "test")}


def balance(rows, ratio, seed=42):
    """Keep at most `ratio` fake videos per real video (FF++ is 1 real : 4+ fake)."""
    real = [r for r in rows if r[1] == 0]; fake = [r for r in rows if r[1] == 1]
    random.Random(seed).shuffle(fake)
    return real + fake[: int(len(real) * ratio)]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_path", "label", "group", "method"])
        w.writerows([(str(p), l, g, m) for p, l, g, m in rows])
    n1 = sum(r[1] for r in rows)
    print(f"{path}: {len(rows)} videos  real={len(rows) - n1} fake={n1}")


def cmd_manifest(a):
    official = None
    if a.layout == "ffpp":
        rows = rows_ffpp(a.root, a.compression, a.methods)
    elif a.layout == "celebdf":
        rows, official = rows_celebdf(a.root)
    elif a.layout == "dfdc":
        rows = rows_dfdc(a.root)
    else:
        rows = rows_folders(a.root)
    if not rows:
        raise SystemExit("No videos found – check --root and --layout.")
    out = Path(a.out)
    if a.test_only:                                  # cross-dataset evaluation set
        if official is not None:
            rows = [r for r in rows if f"{Path(r[0]).parent.name}/{Path(r[0]).name}" in official]
        rows = balance(rows, a.fake_ratio, a.seed) if a.fake_ratio else rows
        if a.limit:
            random.Random(a.seed).shuffle(rows); rows = rows[: a.limit]
        write_csv(out / "test.csv", rows); return
    splits = group_split(rows, a.val, a.test, a.seed)
    for name, rs in splits.items():
        if a.fake_ratio:
            rs = balance(rs, a.fake_ratio, a.seed)
        if a.limit and name == "train":
            random.Random(a.seed).shuffle(rs); rs = rs[: a.limit]
        write_csv(out / f"{name}.csv", rs)


# ---------------------------------------------------------------- face extraction
_CROPPER = None


def _init(size, yunet):
    global _CROPPER
    import cv2
    cv2.setNumThreads(1)
    from .faces import FaceCropper
    _CROPPER = FaceCropper(size=size, model_path=yunet)


def _work(job):
    from .faces import faces_from_video, save_strip
    path, out, frames = job
    side = Path(out).with_suffix(".ratio")
    if Path(out).exists() and side.exists():            # resume support: already done
        return path, float(side.read_text()), True
    try:
        faces, ratio, _ = faces_from_video(path, _CROPPER, frames)
        save_strip(faces, out); side.write_text(f"{ratio:.3f}"); return path, ratio, True
    except Exception as e:
        print("skip", path, e); return path, 0.0, False


def cmd_faces(a):
    from .faces import download_yunet
    try:
        yunet = str(download_yunet())
    except Exception as e:
        yunet = None; print("WARNING:", e)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for csv_path in a.csv:
        rows = list(csv.DictReader(open(csv_path)))
        for r in rows:
            r["uid"] = hashlib.sha1(r["video_path"].encode()).hexdigest()[:16]
        jobs = [(r["video_path"], str(out / f"{r['uid']}.jpg"), a.frames) for r in rows]
        with Pool(a.workers, _init, (a.size, yunet)) as pool:
            res = {p: (ratio, ok) for p, ratio, ok in pool.imap_unordered(_work, jobs, chunksize=4)}
        idx = out / f"{Path(csv_path).parent.name}_{Path(csv_path).stem}.csv"
        with open(idx, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["uid", "label", "group", "method", "face_ratio", "video_path"])
            kept = 0
            for r in rows:
                ratio, ok = res[r["video_path"]]
                if ok and ratio >= a.min_face_ratio:
                    w.writerow([r["uid"], r["label"], r["group"], r.get("method", ""), f"{ratio:.2f}", r["video_path"]])
                    kept += 1
        print(f"{idx}: kept {kept}/{len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--layout", choices=["ffpp", "celebdf", "dfdc", "folders"], required=True)
    m.add_argument("--root", required=True); m.add_argument("--out", required=True)
    m.add_argument("--compression", default="c23")
    m.add_argument("--methods", nargs="*", help="FF++ methods e.g. Deepfakes Face2Face FaceSwap NeuralTextures")
    m.add_argument("--val", type=float, default=0.15); m.add_argument("--test", type=float, default=0.15)
    m.add_argument("--fake-ratio", type=float, default=1.0, help="max fake per real (0 = keep all)")
    m.add_argument("--limit", type=int, default=0, help="cap #train videos (quick experiments)")
    m.add_argument("--test-only", action="store_true"); m.add_argument("--seed", type=int, default=42)
    f = sub.add_parser("faces")
    f.add_argument("--csv", nargs="+", required=True); f.add_argument("--out", required=True)
    f.add_argument("--frames", type=int, default=16); f.add_argument("--size", type=int, default=224)
    f.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    f.add_argument("--min-face-ratio", type=float, default=0.5)
    a = ap.parse_args()
    cmd_manifest(a) if a.cmd == "manifest" else cmd_faces(a)
