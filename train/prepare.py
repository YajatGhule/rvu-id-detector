"""Build the YOLO training set for the card/lanyard detector under data/dataset/.

Source (under data/, which is not committed):
  data/external/kaggle-idcard12   Kaggle "shriharisrini/idcard12", a Roboflow export with
                                  rotated boxes: 8 corner numbers, class, difficulty per box.
                                  Class "0" is the ID card, class "1" is the lanyard strap.

Output: data/dataset/images/{train,val}, labels/{train,val}, data.yaml
  class 0 = id_card, class 1 = lanyard. Rotated boxes become upright boxes around them.
  The Kaggle "valid" and "test" folders together become our val split.

The 200 GitHub photos (ID / No_ID per image) are NOT mixed in here: they stay separate as an
honest wearing / not wearing check, see train/eval_baseline.py.

Run: uv run python train/prepare.py
"""
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = ROOT / "data/external/kaggle-idcard12"
OUT = ROOT / "data/dataset"
CLASSES = {"0": 0, "1": 1}          # Kaggle name -> our class id; anything else is dropped


def read_rotated_boxes(path: Path) -> list[tuple[int, list[float]]]:
    """Return (class id, 8 corner numbers) per box.

    Some files have two boxes glued together without a line break, so the "difficulty" digit
    of one box is stuck to the first number of the next ("0574.34"). We split those apart.
    """
    tokens: list[str] = []
    for token in path.read_text().split():
        glued = (len(tokens) % 10 == 9 and "." in token and len(token) > 1
                 and token[0] in "01" and token[1].isdigit())
        tokens += [token[0], token[1:]] if glued else [token]
    boxes = []
    for i in range(0, len(tokens) - 9, 10):
        name = tokens[i + 8]
        if name in CLASSES:
            boxes.append((CLASSES[name], [float(v) for v in tokens[i:i + 8]]))
    return boxes


def to_yolo_line(class_id: int, corners: list[float], width: int, height: int) -> str | None:
    xs = [min(max(x, 0), width) for x in corners[0::2]]
    ys = [min(max(y, 0), height) for y in corners[1::2]]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w < 2 or h < 2:
        return None
    return (f"{class_id} {(min(xs) + max(xs)) / 2 / width:.6f} {(min(ys) + max(ys)) / 2 / height:.6f} "
            f"{w / width:.6f} {h / height:.6f}")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split, folders in (("train", ["train"]), ("val", ["valid", "test"])):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        images = boxes = 0
        per_class = [0, 0]
        for folder in folders:
            for image in sorted((KAGGLE / folder / "images").iterdir()):
                label = KAGGLE / folder / "labelTxt" / (image.stem + ".txt")
                picture = cv2.imread(str(image))
                if picture is None or not label.exists():
                    continue
                height, width = picture.shape[:2]
                lines = []
                for class_id, corners in read_rotated_boxes(label):
                    line = to_yolo_line(class_id, corners, width, height)
                    if line:
                        lines.append(line)
                        per_class[class_id] += 1
                shutil.copy(image, OUT / "images" / split / image.name)
                (OUT / "labels" / split / (image.stem + ".txt")).write_text("\n".join(lines))
                images += 1
                boxes += len(lines)
        print(f"{split}: {images} images, {boxes} boxes "
              f"(id_card {per_class[0]}, lanyard {per_class[1]})")
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: id_card\n  1: lanyard\n")


if __name__ == "__main__":
    main()
