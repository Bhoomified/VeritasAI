from django.urls import path
from . import views
app_name = "analysis"
urlpatterns = [
    path("", views.landing, name="landing"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("upload/", views.upload, name="upload"),
    path("history/", views.history, name="history"),
    path("analysis/<int:pk>/loading/", views.loading, name="loading"),
    path("analysis/<int:pk>/", views.detail, name="detail"),
    path("analysis/<int:pk>/delete/", views.delete_record, name="delete"),
]
