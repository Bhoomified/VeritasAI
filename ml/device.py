import torch
def select_device(preferred=None):
    preferred = preferred or "auto"
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

