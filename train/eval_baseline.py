"""Score the current pipeline on the labelled VOC images (ID = wearing, No_ID = not wearing).

Each image is shown to a fresh pipeline a few times so the smoothing can settle, then the
biggest person in the picture is compared with the label.

Run: uv run python train/eval_baseline.py --color "#67a5b8"
"""
import argparse
import collections
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from idcard.config import Settings
from idcard.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
VOC = ROOT / "data/external/aravinth-id-card/Tensorflow/workspace/images"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", default="#67a5b8")
    parser.add_argument("--tolerance", type=float, default=25.0)
    args = parser.parse_args()

    settings = Settings(band_color=args.color, tolerance=args.tolerance)
    pipeline = Pipeline(settings, ROOT / "models")
    table = collections.Counter()
    for xml_path in sorted(VOC.glob("*/*.xml")):
        label = ET.parse(xml_path).getroot().find("object/name").text
        frame = cv2.imread(str(xml_path.with_suffix(".jpg")))
        pipeline.reset()
        people = []
        for step in range(4):
            people = pipeline.process(frame, step * 0.2)
        if not people:
            table[(label, "NO_PERSON")] += 1
            continue
        biggest = max(people, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
        table[(label, biggest.state)] += 1

    for label in ("ID", "No_ID"):
        row = {state: n for (lab, state), n in table.items() if lab == label}
        print(label, dict(sorted(row.items())))
    right = table[("ID", "WEARING")] + table[("No_ID", "NOT_WEARING")]
    print(f"correct: {right}/{sum(table.values())}")


if __name__ == "__main__":
    main()
