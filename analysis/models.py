from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

class AnalysisRecord(models.Model):
    STATUS_CHOICES = [("queued", "Queued"), ("processing", "Processing"), ("completed", "Completed"), ("failed", "Failed")]
    PREDICTION_CHOICES = [("REAL", "Real"), ("FAKE", "Fake")]
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="analyses")
    video = models.FileField(upload_to="uploads/%Y/%m/%d/")
    original_filename = models.CharField(max_length=255)
    file_hash = models.CharField(max_length=64, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")
    prediction = models.CharField(max_length=4, choices=PREDICTION_CHOICES, blank=True)
    confidence = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])
    frame_count = models.PositiveIntegerField(default=0)
    model_version = models.CharField(max_length=100, blank=True)
    processing_duration = models.FloatField(null=True, blank=True)
    error_details = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    class Meta:
        ordering = ["-created_at"]
    def __str__(self):
        return f"{self.original_filename} ({self.status})"

