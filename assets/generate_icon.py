#!/usr/bin/env python3
"""
Generate SASA app icons (.icns for macOS, .ico for Windows, .png for Linux/web).

Uses matplotlib to render a spectral-analysis-themed icon programmatically.
No external image assets required.

Usage:
    python assets/generate_icon.py
"""

import struct
import io
import os
import sys
from pathlib import Path

# Add parent to path so we can optionally import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Requires: pip install numpy pillow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# The icon is the in-app mark, at icon scale: the same seven-bar spectrum as
# #i-brand in ui/renderer/index.html, in the brand's blue, on the light
# theme's surface.
#
# Light background on purpose: the application defaults to the light theme
# because its output is printed, and an icon that contradicted the window it
# opens would be the only dark thing in the product.
#
# The blue ramp is the mark's own colour and is deliberately NOT one of the
# series colours from tokens.css — those six are reserved for identifying
# traces in a plot, where a colour has to mean a specific weighting or role.
# The surface, border and ink below ARE tokens and must not drift from them.
# ---------------------------------------------------------------------------

# Bright at the top of the mark, deep at the bottom.
BRAND_GRADIENT = [
    (0.00, (60, 150, 255)),
    (1.00, (30, 100, 220)),
]

BG_SURFACE = (0xFF, 0xFF, 0xFF, 255)   # --bg-surface, light
BORDER = (0xC8, 0xD1, 0xDE, 255)       # --border,     light
INK = (0x0F, 0x14, 0x1A, 255)          # --text,       light

# x, y, width, height in the 32-unit box of #i-brand, with its bar opacity.
BRAND_BARS = [
    (2, 14, 3, 8, 0.55),
    (6, 10, 3, 16, 0.70),
    (10, 6, 3, 24, 0.85),
    (14, 2, 3, 28, 1.00),
    (18, 4, 3, 24, 1.00),
    (22, 8, 3, 16, 0.85),
    (26, 12, 3, 8, 0.70),
]


def _gradient_colour(t: float) -> tuple:
    """Sample BRAND_GRADIENT at t in [0, 1], linearly between stops."""
    t = min(1.0, max(0.0, t))
    for i in range(len(BRAND_GRADIENT) - 1):
        t0, c0 = BRAND_GRADIENT[i]
        t1, c1 = BRAND_GRADIENT[i + 1]
        if t <= t1:
            span = t1 - t0
            f = 0.0 if span == 0 else (t - t0) / span
            return tuple(int(round(a + (b - a) * f)) for a, b in zip(c0, c1))
    return BRAND_GRADIENT[-1][1]


def create_icon_image(size: int = 1024) -> Image.Image:
    """The SASA app icon: the spectrum mark, in the brand blue, on paper."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = int(size * 0.03)
    radius = int(size * 0.18)
    box = [margin, margin, size - margin, size - margin]

    draw.rounded_rectangle(box, radius=radius, fill=BG_SURFACE)
    draw.rounded_rectangle(box, radius=radius, outline=BORDER,
                           width=max(1, size // 192))

    # One vertical ramp across the whole mark, rendered full-bleed and then
    # masked by the bars, so every bar samples the same ramp at its own height
    # rather than repeating the ramp inside itself.
    inner = size - 2 * margin
    # The 32-unit box of #i-brand. Its bars only occupy y = 2..30 and
    # x = 2..29 of that box, so the box is oversized relative to the ink and
    # the placement below is chosen from where the BARS land, not where the
    # box does: they run from 34 % to 88 % of the icon's height, under a
    # wordmark that sits in the top sixth.
    mark = int(inner * 0.68)
    origin_x = (size - mark) // 2
    origin_y = int(size * 0.29)
    unit = mark / 32.0

    ramp = Image.new('RGBA', (1, mark))
    for y in range(mark):
        ramp.putpixel((0, y), _gradient_colour(y / max(1, mark - 1)) + (255,))
    ramp = ramp.resize((mark, mark), Image.Resampling.BILINEAR)

    mask = Image.new('L', (mark, mark), 0)
    mask_draw = ImageDraw.Draw(mask)
    for x, y, w, h, opacity in BRAND_BARS:
        left = x * unit
        top = y * unit
        mask_draw.rounded_rectangle(
            [left, top, left + w * unit, top + h * unit],
            radius=max(1, int(unit * 0.5)),
            fill=int(round(255 * opacity)),
        )

    img.paste(ramp, (origin_x, origin_y), mask)

    # Wordmark. In the light theme's ink, so it reads on the white ground.
    font = None
    font_size = int(size * 0.135)
    for font_name in ['SF Pro Display', 'HelveticaNeue', 'Helvetica Neue',
                      'Arial Bold', 'Arial', 'Segoe UI']:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    text = 'SASA'
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((size - (bbox[2] - bbox[0])) // 2 - bbox[0], int(size * 0.115) - bbox[1]),
        text, fill=INK, font=font,
    )

    return img


ICO_SIZES = [16, 32, 48, 64, 128, 256]


def save_ico(img: Image.Image, path: Path):
    """
    Save as .ico with every size Windows asks for.

    The base image passed to Pillow must be the LARGEST, not the smallest.
    Pillow's ICO writer drops any requested size bigger than the image it was
    handed, so saving from the 16x16 frame -- which is what
    `icons[0].save(..., append_images=icons[1:])` did, since the list was built
    ascending -- silently wrote a 660-byte file containing 16x16 alone. The
    Windows executable then carried a 16-pixel icon scaled up to the taskbar.

    The pre-resized frames are still passed so each entry is a LANCZOS
    downsample of the 1024px master rather than a chain of rescales.
    """
    frames = {s: img.resize((s, s), Image.Resampling.LANCZOS) for s in ICO_SIZES}
    largest = frames[max(ICO_SIZES)]
    largest.save(
        str(path),
        format='ICO',
        sizes=[(s, s) for s in ICO_SIZES],
        append_images=[frames[s] for s in ICO_SIZES if s != max(ICO_SIZES)],
    )

    # Trust the file, not the call. A silently-truncated icon is exactly the
    # failure this function just had, and it is invisible until someone looks
    # at the packaged .exe on a Windows desktop.
    with Image.open(str(path)) as written:
        got = sorted(written.info.get('sizes', []))
    want = [(s, s) for s in sorted(ICO_SIZES)]
    if got != want:
        raise SystemExit(
            f'generate_icon.py: {path.name} was written with sizes {got}, expected {want}'
        )


def save_icns(img: Image.Image, path: Path):
    """Save as .icns for macOS using iconutil."""
    import subprocess
    import tempfile

    # Create iconset directory
    with tempfile.TemporaryDirectory() as tmpdir:
        iconset_dir = Path(tmpdir) / 'sasa.iconset'
        iconset_dir.mkdir()

        # Required sizes for .icns
        icon_sizes = [
            (16, 'icon_16x16.png'),
            (32, 'icon_16x16@2x.png'),
            (32, 'icon_32x32.png'),
            (64, 'icon_32x32@2x.png'),
            (128, 'icon_128x128.png'),
            (256, 'icon_128x128@2x.png'),
            (256, 'icon_256x256.png'),
            (512, 'icon_256x256@2x.png'),
            (512, 'icon_512x512.png'),
            (1024, 'icon_512x512@2x.png'),
        ]

        for size, name in icon_sizes:
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(str(iconset_dir / name), format='PNG')

        # Use iconutil to create .icns
        try:
            subprocess.run(
                ['iconutil', '-c', 'icns', str(iconset_dir), '-o', str(path)],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: just save as PNG if iconutil not available
            img.resize((512, 512), Image.Resampling.LANCZOS).save(
                str(path.with_suffix('.png')), format='PNG'
            )
            print(f'  iconutil not available, saved PNG instead: {path.with_suffix(".png")}')
            return


def main():
    assets_dir = Path(__file__).resolve().parent
    print('Generating SASA app icons...')

    img = create_icon_image(1024)

    # Save PNG (universal)
    png_path = assets_dir / 'sasa_icon.png'
    img.save(str(png_path), format='PNG')
    print(f'  PNG: {png_path}')

    # Save ICO (Windows)
    ico_path = assets_dir / 'sasa.ico'
    try:
        save_ico(img, ico_path)
        print(f'  ICO: {ico_path}')
    except Exception as e:
        print(f'  ICO failed: {e}')

    # Save ICNS (macOS)
    icns_path = assets_dir / 'sasa.icns'
    try:
        save_icns(img, icns_path)
        print(f'  ICNS: {icns_path}')
    except Exception as e:
        print(f'  ICNS failed: {e}')

    print('Done.')


if __name__ == '__main__':
    main()
