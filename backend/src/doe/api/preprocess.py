"""Convert common user images into strict challenge targets."""

import warnings
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from doe.physics import SIZE, png_to_gray

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_TARGET_BYTES = 64 * 1024


def convert_image(data: bytes) -> bytes:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data), formats=("JPEG", "PNG", "WEBP")) as source:
                if source.width * source.height > 24_000_000:
                    raise ValueError("Image exceeds 24 megapixels")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Choose a single-frame image")
                if (
                    len(data) <= MAX_TARGET_BYTES
                    and source.format == "PNG"
                    and source.size == (SIZE, SIZE)
                    and source.mode == "L"
                    and "transparency" not in source.info
                    and source.getexif().get(274, 1) == 1
                    and data[24:26] == b"\x08\x00"
                ):
                    png_to_gray(data)
                    return data
                image = ImageOps.exif_transpose(source)
                if image.mode.startswith("I;16"):
                    values = np.array(image, dtype=np.uint32)
                    values[values == image.info.get("transparency", -1)] = 0
                    gray = Image.fromarray(((values + 128) // 257).astype(np.uint8))
                elif "A" in image.getbands() or "transparency" in image.info:
                    background = Image.new("RGBA", image.size, (0, 0, 0, 255))
                    background.alpha_composite(image.convert("RGBA"))
                    gray = ImageOps.grayscale(background.convert("RGB"))
                else:
                    gray = image.convert("L")
                gray = ImageOps.pad(gray, (SIZE, SIZE), Image.Resampling.LANCZOS, color=0)
                output = BytesIO()
                gray.save(output, format="PNG")
                return output.getvalue()
    except (
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Cannot decode image. Choose a valid JPEG, PNG or WebP") from exc
