from django.contrib import admin
from .models import AnalysisRecord
@admin.register(AnalysisRecord)
class AnalysisRecordAdmin(admin.ModelAdmin):
    list_display = ("original_filename", "owner", "status", "prediction", "created_at")
    list_filter = ("status", "prediction")

