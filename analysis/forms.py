from django import forms
from django.conf import settings
from .models import AnalysisRecord

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = AnalysisRecord
        fields = ("video",)
    def clean_video(self):
        video = self.cleaned_data["video"]
        allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        name = video.name.lower()
        if not any(name.endswith(ext) for ext in allowed):
            raise forms.ValidationError("Unsupported video extension. Use MP4, MOV, AVI, MKV or WebM.")
        if video.size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise forms.ValidationError(f"Video must be smaller than {settings.MAX_UPLOAD_SIZE_MB} MB.")
        content_type = getattr(video, "content_type", "")
        if content_type and not content_type.startswith("video/") and content_type not in {"application/octet-stream"}:
            raise forms.ValidationError("The uploaded file is not recognised as a video.")
        return video

