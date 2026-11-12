"""Export an Ultralytics model to ONNX into models/.

The project venv has no torch on purpose, so run this in a throwaway environment with a
CPU-only torch wheel (small download, no CUDA):

    uv run --no-project --python 3.12 \
        --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        --with ultralytics --with onnx --with onnxslim \
        python train/export.py                       # default: yolo11n-pose.pt, 640

    ... python train/export.py runs/train/weights/best.pt --imgsz 640   # a trained model later
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", nargs="?", default="yolo11n-pose.pt",
                        help="a .pt file, or an Ultralytics model name to download")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--keep-pt", action="store_true", help="do not delete a downloaded .pt")
    args = parser.parse_args()

    from ultralytics import YOLO  # imported here so --help works without ultralytics

    MODELS.mkdir(exist_ok=True)
    weights = Path(args.weights)
    downloaded = not weights.exists()
    model = YOLO(str(weights))  # downloads official weights into the cwd if needed
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                                  simplify=True, dynamic=False, half=False))
    target = MODELS / onnx_path.name
    if onnx_path.resolve() != target.resolve():
        shutil.move(str(onnx_path), target)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")

    if downloaded and not args.keep_pt:
        pt = Path(weights.name)
        if pt.exists():
            pt.unlink()
            print(f"deleted downloaded {pt}")


if __name__ == "__main__":
    main()
