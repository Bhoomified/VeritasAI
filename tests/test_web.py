from io import BytesIO
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from analysis.forms import VideoUploadForm
from analysis.models import AnalysisRecord

class WebTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", "alice@example.com", "safe-password-123")
        self.other = User.objects.create_user("bob", "bob@example.com", "safe-password-123")

    def test_upload_extension_validation(self):
        upload = SimpleUploadedFile("clip.txt", b"not a video", content_type="text/plain")
        form = VideoUploadForm(files={"video": upload})
        self.assertFalse(form.is_valid())

    def test_authentication_and_user_isolation(self):
        self.client.login(username="alice", password="safe-password-123")
        record = AnalysisRecord.objects.create(owner=self.other, video="uploads/other.mp4",
            original_filename="other.mp4", file_hash="a" * 64)
        response = self.client.get(f"/analysis/{record.pk}/")
        self.assertEqual(response.status_code, 404)

    def test_dashboard_requires_login(self):
        self.assertEqual(self.client.get("/dashboard/").status_code, 302)

