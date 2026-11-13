from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.db.models import QuerySet
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from .forms import VideoUploadForm
from .models import AnalysisRecord
from .services import run_analysis, sha256_file

def landing(request):
    return render(request, "landing.html")

@login_required
def dashboard(request):
    return render(request, "analysis/dashboard.html", {"recent": AnalysisRecord.objects.filter(owner=request.user)[:5]})

@login_required
def upload(request):
    form = VideoUploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        record = form.save(commit=False)
        record.owner = request.user
        record.original_filename = record.video.name
        record.file_hash = sha256_file(record.video)
        record.save()
        run_analysis(record)
        return redirect("analysis:detail", pk=record.pk)
    return render(request, "analysis/upload.html", {"form": form, "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB})

@login_required
def history(request):
    return render(request, "analysis/history.html", {"records": AnalysisRecord.objects.filter(owner=request.user)})

@login_required
def detail(request, pk):
    record = get_object_or_404(AnalysisRecord, pk=pk, owner=request.user)
    return render(request, "analysis/detail.html", {"record": record})

@login_required
def loading(request, pk):
    record = get_object_or_404(AnalysisRecord, pk=pk, owner=request.user)
    return render(request, "analysis/loading.html", {"record": record})

@login_required
def delete_record(request, pk):
    record = get_object_or_404(AnalysisRecord, pk=pk, owner=request.user)
    if request.method == "POST":
        record.video.delete(save=False)
        record.delete()
        messages.success(request, "Analysis deleted.")
        return redirect("analysis:history")
    raise Http404
