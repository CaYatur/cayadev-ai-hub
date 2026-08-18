"""Draw the application icon.

The mark is a thick angular "C" — CaYaDev's wordmark uses cut corners rather
than curves, so the arc terminates in flat diagonal cuts instead of round caps —
with a bright node sitting in its opening. The C is the brand; the node is what
the app does, which is route work between models.

Drawn at 4x and downsampled, because the shapes are masks and Pillow does not
antialias those.

Run: python scripts/make_icon.py
Then: npm run tauri icon -- assets/icon.png
"""
import math
import pathlib

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
SS = 4  # supersampling factor
S = SIZE * SS

# Straight from the site's stylesheets.
RED = (220, 38, 38)
RED_LIGHT = (239, 68, 68)
INK_DEEP = (7, 9, 15)
INK = (11, 16, 32)
INK_LIFT = (18, 26, 48)
TEXT = (245, 247, 251)


def rounded_mask(size, radius):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius, fill=255)
    return mask


def background(size):
    """Dark plate with a faint glow behind where the mark sits."""
    base = Image.new("RGB", (size, size), INK)
    glow = Image.new("RGB", (size, size), INK)
    draw = ImageDraw.Draw(glow)

    # A soft pool of lift slightly above centre, so the plate is not flat.
    cx, cy, r = size * 0.5, size * 0.42, size * 0.42
    for step in range(28, 0, -1):
        t = step / 28
        radius = r * t
        shade = tuple(
            int(INK[i] + (INK_LIFT[i] - INK[i]) * (1 - t) ** 1.7) for i in range(3)
        )
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=shade)

    glow = glow.filter(ImageFilter.GaussianBlur(size * 0.06))
    base.paste(glow, (0, 0))

    # Vignette so the corners settle into the dark.
    vignette = Image.new("L", (size, size), 0)
    vd = ImageDraw.Draw(vignette)
    inset = size * 0.06
    vd.ellipse([inset, inset, size - inset, size - inset], fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(size * 0.11))
    return Image.composite(base, Image.new("RGB", (size, size), INK_DEEP), vignette)


def angular_c_mask(size):
    """A thick arc whose ends are cut flat on a diagonal rather than rounded."""
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)

    cx = cy = size / 2
    outer = size * 0.335
    inner = size * 0.205

    d.ellipse([cx - outer, cy - outer, cx + outer, cy + outer], fill=255)
    d.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], fill=0)

    # Remove the wedge that opens the C, on the right, with straight edges.
    reach = size
    opening_half = 31  # degrees either side of horizontal
    wedge = [(cx, cy)]
    for angle in (-opening_half, opening_half):
        rad = math.radians(angle)
        wedge.append((cx + math.cos(rad) * reach, cy + math.sin(rad) * reach))
    d.polygon(wedge, fill=0)

    return mask


def draw_mark(size):
    """The C in brand red, lit from the upper left."""
    mark = Image.new("RGB", (size, size), RED)
    lit = Image.new("RGB", (size, size), RED_LIGHT)

    # Diagonal gradient: brighter top-left, deeper bottom-right.
    ramp = Image.new("L", (size, size), 0)
    rd = ImageDraw.Draw(ramp)
    for i in range(size):
        rd.line([(0, i), (i, 0)], fill=max(0, 235 - int(i * 235 / (size * 0.95))))
    mark = Image.composite(lit, mark, ramp)

    return mark


def node(layer, size):
    """The bright point in the C's opening — the hub."""
    d = ImageDraw.Draw(layer)
    cx = size * 0.5 + size * 0.263
    cy = size * 0.5
    r = size * 0.062

    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse(
        [cx - r * 2.6, cy - r * 2.6, cx + r * 2.6, cy + r * 2.6],
        fill=RED + (110,),
    )
    layer.alpha_composite(glow.filter(ImageFilter.GaussianBlur(size * 0.02)))

    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED_LIGHT + (255,))
    d.ellipse(
        [cx - r * 0.42, cy - r * 0.42, cx + r * 0.42, cy + r * 0.42],
        fill=TEXT + (255,),
    )


def build():
    plate = background(S).convert("RGBA")

    mark = draw_mark(S).convert("RGBA")
    mark.putalpha(angular_c_mask(S))

    # A soft red bloom under the mark, so it sits in the plate rather than on it.
    bloom = mark.filter(ImageFilter.GaussianBlur(S * 0.035))
    plate.alpha_composite(Image.blend(Image.new("RGBA", (S, S), (0, 0, 0, 0)), bloom, 0.5))
    plate.alpha_composite(mark)

    node(plate, S)

    icon = plate.resize((SIZE, SIZE), Image.LANCZOS)
    icon.putalpha(rounded_mask(SIZE, int(SIZE * 0.22)).resize((SIZE, SIZE), Image.LANCZOS))
    return icon


if __name__ == "__main__":
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)
    icon = build()

    target = out_dir / "icon.png"
    icon.save(target)
    print(f"wrote {target}")

    # A quick sanity sheet: the mark has to survive being 32 pixels wide.
    sheet = Image.new("RGBA", (SIZE, 220), (0, 0, 0, 0))
    x = 8
    for px in (16, 32, 64, 128, 256):
        sheet.alpha_composite(icon.resize((px, px), Image.LANCZOS), (x, 8))
        x += px + 16
    sheet.save(out_dir / "icon-sizes.png")
    print(f"wrote {out_dir / 'icon-sizes.png'}")
