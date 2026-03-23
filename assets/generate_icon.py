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


def create_icon_image(size: int = 1024) -> Image.Image:
    """Create a SASA-themed app icon."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background: rounded rectangle with dark gradient
    margin = int(size * 0.03)
    radius = int(size * 0.18)

    # Dark background
    bg_color = (12, 12, 20, 255)
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        fill=bg_color,
    )

    # Subtle border
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        outline=(30, 30, 50, 200),
        width=max(1, size // 256),
    )

    # Draw spectral bars (like the SVG logo in the web UI)
    bar_data = [
        (0.10, 0.44, 0.25, 0.40),   # x_center_frac, y_top_frac, height_frac, opacity
        (0.22, 0.31, 0.50, 0.55),
        (0.34, 0.19, 0.75, 0.70),
        (0.46, 0.06, 0.88, 0.90),
        (0.58, 0.13, 0.75, 0.85),
        (0.70, 0.25, 0.50, 0.70),
        (0.82, 0.38, 0.25, 0.50),
    ]

    bar_width = size * 0.085
    base_y = size * 0.85  # Bottom of bars

    for x_frac, y_frac, h_frac, opacity in bar_data:
        cx = size * x_frac
        bar_height = size * h_frac * 0.7
        top = base_y - bar_height
        left = cx - bar_width / 2
        right = cx + bar_width / 2

        # Gradient from blue at bottom to bright blue/white at top
        n_segments = max(1, int(bar_height / 2))
        for i in range(n_segments):
            frac = i / max(1, n_segments - 1)
            y0 = base_y - (i / n_segments) * bar_height
            y1 = base_y - ((i + 1) / n_segments) * bar_height

            # Color: electric blue gradient
            r = int(30 + frac * 30)
            g = int(100 + frac * 50)
            b = int(220 + frac * 35)
            a = int(255 * opacity * (0.6 + 0.4 * frac))

            bar_radius = max(1, int(bar_width * 0.15))
            draw.rounded_rectangle(
                [left, y1, right, y0],
                radius=bar_radius,
                fill=(r, g, b, a),
            )

    # Draw "SASA" text at top
    try:
        # Try system fonts
        font_size = int(size * 0.11)
        for font_name in ['SF Pro Display', 'Helvetica Neue', 'Arial', 'Segoe UI']:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except (OSError, IOError):
                continue
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # SASA text
    text = "SASA"
    text_color = (232, 232, 239, 230)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (size - text_width) // 2
    text_y = int(size * 0.03)
    draw.text((text_x, text_y), text, fill=text_color, font=font)

    return img


def save_ico(img: Image.Image, path: Path):
    """Save as .ico with multiple sizes."""
    sizes = [16, 32, 48, 64, 128, 256]
    icons = []
    for s in sizes:
        resized = img.resize((s, s), Image.Resampling.LANCZOS)
        icons.append(resized)
    icons[0].save(str(path), format='ICO', sizes=[(s, s) for s in sizes], append_images=icons[1:])


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
