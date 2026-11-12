"""Pick ONNX Runtime execution providers (CPU/GPU backends) for this machine."""
import onnxruntime as ort

# our short device names -> ONNX Runtime provider names
PROVIDERS = {
    "cuda": "CUDAExecutionProvider",          # NVIDIA
    "directml": "DmlExecutionProvider",       # any GPU on Windows
    "coreml": "CoreMLExecutionProvider",      # Apple
    "openvino": "OpenVINOExecutionProvider",  # Intel
    "cpu": "CPUExecutionProvider",
}
AUTO_ORDER = ["cuda", "directml", "coreml", "openvino", "cpu"]
CPU = PROVIDERS["cpu"]


def available_devices() -> list[str]:
    """Short names of the devices this onnxruntime build supports, in AUTO_ORDER."""
    present = set(ort.get_available_providers())
    return [name for name in AUTO_ORDER if PROVIDERS[name] in present]


def choose_providers(preference: str, available: list[str] | None = None) -> list[str]:
    """Provider list for ort.InferenceSession, best first, always ending with CPU.

    preference is "auto" or one of PROVIDERS. If the preferred device is not available we
    fall back to the "auto" choice instead of failing. `available` can be injected in tests.
    """
    if available is None:
        available = available_devices()
    preference = (preference or "auto").lower()

    if preference in PROVIDERS and preference in available:
        best = preference
    else:  # "auto", an unknown name, or a device that is not installed
        best = next((name for name in AUTO_ORDER if name in available), "cpu")

    if best == "cpu":
        return [CPU]
    return [PROVIDERS[best], CPU]
