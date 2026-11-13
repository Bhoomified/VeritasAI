import hashlib
import time
from django.conf import settings
from ml.inference import InferenceError, predict_video

def sha256_file(uploaded):
    digest = hashlib.sha256()
    for chunk in uploaded.chunks():
        digest.update(chunk)
    uploaded.seek(0)
    return digest.hexdigest()

def run_analysis(record):
    started = time.perf_counter()
    record.status = "processing"
    record.save(update_fields=["status", "updated_at"])
    try:
        result = predict_video(
            record.video.path, checkpoint_path=settings.MODEL_CHECKPOINT_PATH or None,
            num_frames=settings.SAMPLED_FRAMES, use_imagenet_weights=settings.USE_IMAGENET_WEIGHTS,
        )
        record.status = "completed"
        record.prediction = result.prediction
        record.confidence = result.confidence
        record.frame_count = result.frame_count
        record.model_version = result.model_version
        record.processing_duration = time.perf_counter() - started
        record.save()
    except InferenceError as exc:
        record.status = "failed"
        record.error_details = str(exc)
        record.processing_duration = time.perf_counter() - started
        record.save()

