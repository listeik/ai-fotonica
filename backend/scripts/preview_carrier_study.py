"""Render fixed examples with identical grayscale levels across all models."""

import argparse
from io import BytesIO
from pathlib import Path

import torch
from evaluate_carrier_study import (
    SIZE,
    FresnelPropagator,
    checkpoint_arg,
    file_hash,
    load_model,
    load_test,
    measure,
)
from PIL import Image, ImageDraw


def example_arg(value):
    split, separator, index = value.rpartition(":")
    if not separator or not index.isdecimal():
        raise argparse.ArgumentTypeError("expected SPLIT:INDEX")
    return split, int(index)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", type=checkpoint_arg, required=True)
    parser.add_argument("--example", action="append", type=example_arg,
                        help="fixed examples; default first quartet from every test split")
    parser.add_argument("--scale", type=int, default=1)
    args = parser.parse_args()
    if args.scale < 1 or len(dict(args.model)) != len(args.model):
        parser.error("scale must be positive and model names unique")
    if args.output.resolve() in {args.test.resolve(), *(path.resolve() for _, path in args.model)}:
        parser.error("output must differ from inputs")
    torch.set_num_threads(2)
    splits, ids, _ = load_test(args.test)
    examples = args.example or [(name, 0) for name in splits]
    for split, index in examples:
        if split not in splits or not 0 <= index < len(splits[split]):
            parser.error(f"invalid example {split}:{index}")
    models = [(name, load_model(path, "cpu")[0]) for name, path in args.model]
    cell = SIZE * args.scale
    left, gap, heading = 200, 14, 22
    row_height = heading + cell + 8
    width = left + 4 * (cell + gap)
    height = 42 + len(examples) * (len(models) + 1) * row_height
    sheet = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), "Fixed DOE examples | same 0..255 grayscale | nearest-neighbor pixels", fill="black")
    optics = FresnelPropagator()
    y = 42
    for split, index in examples:
        raw = splits[split][index]
        target = raw.float().div(255)
        rows = [("Targets", [Image.fromarray(plane.numpy(), "L") for plane in raw])]
        for name, model in models:
            phase = model(target[None])[0, 0]
            _, images, _ = measure(phase, target, optics)
            rows.append((name, [Image.open(BytesIO(data)).convert("L") for data in images]))
        for name, images in rows:
            draw.text((12, y + 5), name, fill="black")
            draw.text((12, y + 22), ids[split][index][:30], fill="#555555")
            for plane, image in enumerate(images):
                x = left + plane * (cell + gap)
                draw.text((x, y), f"z{plane + 1}", fill="black")
                sheet.paste(image.resize((cell, cell), Image.Resampling.NEAREST), (x, y + heading))
            y += row_height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Saved fixed preview {args.output}; test SHA256={file_hash(args.test)}")


if __name__ == "__main__":
    main()
