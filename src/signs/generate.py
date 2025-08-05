"""Utility to generate speed limit sign images on the fly.

This module used to pre-generate a folder full of PNG files which were then
loaded at runtime.  To keep the repository size small and to avoid shipping a
large number of almost identical assets, the game now generates the sign images
on demand.  The :func:`generate_speed_limit_sign` function returns a Pillow
``Image`` with transparent background that can be uploaded to the GPU as a
texture.

When run as a script it will still output all standard sign PNGs for debugging
purposes.
"""

from PIL import Image, ImageDraw, ImageFont

# Sign dimensions and styles
WIDTH, HEIGHT = 300, 400  # Official ratio close to 3:4
BORDER = 12
RADIUS = 28  # Rounded corner radius
PADDING = int(HEIGHT * 0.07)
BG = (255, 255, 255)
FG = (0, 0, 0)

SPEED_LIMIT_FONT_SIZE = int(HEIGHT * 0.13)
NUMERAL_FONT_SIZE = int(HEIGHT * 0.48)

try:
    SL_FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", SPEED_LIMIT_FONT_SIZE)
    NUM_FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", NUMERAL_FONT_SIZE)
except Exception:
    SL_FONT = ImageFont.load_default()
    NUM_FONT = ImageFont.load_default()


def generate_speed_limit_sign(speed: int) -> Image.Image:
    """Return a Pillow image containing a speed limit sign for *speed* mph."""

    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded rectangle mask (anti-aliased corners)
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, WIDTH - 1, HEIGHT - 1], RADIUS, fill=255)
    img.paste(BG, mask=mask)

    # Border (draw on top)
    draw.rounded_rectangle(
        [BORDER // 2, BORDER // 2, WIDTH - BORDER // 2, HEIGHT - BORDER // 2],
        RADIUS,
        outline=FG,
        width=BORDER,
    )

    # "SPEED LIMIT" - stack and center in upper half
    y_offset = PADDING
    bbox1 = draw.textbbox((0, 0), "SPEED", font=SL_FONT)
    w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    bbox2 = draw.textbbox((0, 0), "LIMIT", font=SL_FONT)
    w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    label_block_height = h1 + h2 + int(0.10 * h1)
    label_y = y_offset
    draw.text(((WIDTH - w1) // 2, label_y), "SPEED", font=SL_FONT, fill=FG)
    draw.text(
        ((WIDTH - w2) // 2, label_y + h1 + int(0.10 * h1)),
        "LIMIT",
        font=SL_FONT,
        fill=FG,
    )

    # Numeral - centered in remaining space below
    num_str = str(speed)
    bbox_num = draw.textbbox((0, 0), num_str, font=NUM_FONT)
    nw, nh = bbox_num[2] - bbox_num[0], bbox_num[3] - bbox_num[1]
    numeral_y = label_y + label_block_height + int(h1 * 0.5)
    draw.text(((WIDTH - nw) // 2, numeral_y), num_str, font=NUM_FONT, fill=FG)

    return img


if __name__ == "__main__":
    # Generate the standard set of signs for manual inspection / debugging
    for sp in range(5, 91, 5):
        generate_speed_limit_sign(sp).save(f"{sp}.png")
        print(f"Saved {sp}.png")
