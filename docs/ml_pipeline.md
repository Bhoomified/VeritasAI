# ML pipeline

Frames are sampled with `numpy.linspace` from the complete readable frame range, preserving chronology. Each frame is passed through OpenCV's bundled Haar cascade; the largest detected face is padded slightly, resized to 224x224, converted to RGB, and normalized with ImageNet channel statistics. If no face is found, the full frame is used and the count is tracked.

ResNeXt50_32x4d has its final classifier removed, yielding 2048 features per frame. The ordered tensor is shaped `[batch, sequence_length, 2048]` and passed to a configurable LSTM. Its final timestep is classified with a linear head and sigmoid. Checkpoints contain both state dictionaries and model configuration.

Inference errors (unreadable video, no frames, missing checkpoint) are surfaced as failed analyses. The system never maps an error to REAL or FAKE.

