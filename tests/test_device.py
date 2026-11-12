from idcard.device import available_devices, choose_providers

CPU = "CPUExecutionProvider"


def test_cpu_only_machine():
    assert choose_providers("auto", ["cpu"]) == [CPU]
    assert choose_providers("cuda", ["cpu"]) == [CPU]  # not installed -> falls back


def test_auto_order():
    everything = ["cpu", "openvino", "coreml", "directml", "cuda"]
    assert choose_providers("auto", everything) == ["CUDAExecutionProvider", CPU]
    assert choose_providers("auto", ["cpu", "openvino", "directml"])[0] == "DmlExecutionProvider"
    assert choose_providers("auto", ["openvino", "coreml", "cpu"])[0] == "CoreMLExecutionProvider"
    assert choose_providers("auto", ["openvino", "cpu"]) == ["OpenVINOExecutionProvider", CPU]


def test_explicit_preference_wins_when_available():
    assert choose_providers("openvino", ["cuda", "openvino", "cpu"]) == [
        "OpenVINOExecutionProvider", CPU]
    assert choose_providers("cpu", ["cuda", "cpu"]) == [CPU]


def test_unavailable_or_unknown_preference_falls_back_to_auto():
    assert choose_providers("coreml", ["cuda", "cpu"]) == ["CUDAExecutionProvider", CPU]
    assert choose_providers("banana", ["openvino", "cpu"])[0] == "OpenVINOExecutionProvider"
    assert choose_providers("", ["cpu"]) == [CPU]
    assert choose_providers("CUDA", ["cuda", "cpu"])[0] == "CUDAExecutionProvider"


def test_always_ends_with_cpu():
    for pref in ["auto", "cuda", "directml", "openvino", "coreml", "cpu"]:
        assert choose_providers(pref, ["cuda", "directml", "openvino", "coreml"])[-1] == CPU


def test_real_machine_has_cpu():
    assert "cpu" in available_devices()
    assert choose_providers("auto")[-1] == CPU
