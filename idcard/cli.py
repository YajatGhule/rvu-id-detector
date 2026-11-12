"""Command line: `idcard serve` and `idcard doctor`."""

import argparse
import time
from pathlib import Path

import numpy as np

HOST = "127.0.0.1"  # never listen on other interfaces


def serve(args):
    import uvicorn

    from idcard.server import create_app

    print(f"Open http://{HOST}:{args.port}")
    uvicorn.run(create_app(source=args.source), host=HOST, port=args.port)


def doctor(args):
    import onnxruntime

    from idcard import config, device

    settings = config.load(Path(__file__).resolve().parent.parent / "settings.json")
    print("onnxruntime providers:", onnxruntime.get_available_providers())
    print("devices:", device.available_devices())
    print(f"setting '{settings.device}' ->", device.choose_providers(settings.device))

    from idcard.pipeline import Pipeline

    pipeline = Pipeline(settings)
    print("backend loaded:", pipeline.backend_name)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pipeline.process(frame, 0.0)  # warm-up
    start = time.perf_counter()
    for i in range(30):
        pipeline.process(frame, i / 30)
    print(f"FPS over 30 synthetic frames: {30 / (time.perf_counter() - start):.1f}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="idcard")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve", help="start the website on 127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--source", help="camera index or video file (overrides settings)")
    p.set_defaults(func=serve)
    sub.add_parser("doctor", help="print backend info and speed").set_defaults(func=doctor)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
