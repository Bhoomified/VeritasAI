import tempfile
from pathlib import Path
import torch
from django.test import SimpleTestCase
from ml.device import select_device
from ml.model import LSTMClassifier, ModelConfig
from ml.preprocess import sample_indices
from ml.inference import InferenceError, load_checkpoint

class MLTests(SimpleTestCase):
    def test_even_sampling_is_ordered_and_bounded(self):
        indexes = sample_indices(100, 20)
        self.assertEqual(len(indexes), 20)
        self.assertEqual(indexes[0], 0)
        self.assertEqual(indexes[-1], 99)
        self.assertEqual(indexes, sorted(indexes))

    def test_lstm_shape(self):
        model = LSTMClassifier(ModelConfig(input_size=2048, hidden_size=8, num_layers=1))
        output = model(torch.randn(2, 20, 2048))
        self.assertEqual(output.shape, (2,))
        self.assertTrue(torch.all((torch.sigmoid(output) >= 0) & (torch.sigmoid(output) <= 1)))

    def test_cpu_device_is_available(self):
        self.assertEqual(str(select_device("cpu")), "cpu")

    def test_missing_checkpoint_is_explicit(self):
        with self.assertRaises(InferenceError):
            load_checkpoint("/definitely/missing/checkpoint.pt")

    def test_invalid_checkpoint_is_explicit(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            checkpoint.write(b"not a checkpoint")
            checkpoint.flush()
            with self.assertRaises(InferenceError):
                load_checkpoint(checkpoint.name)
