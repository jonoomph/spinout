from PIL import Image, ImageDraw, ImageFont

# Sign dimensions and styles
width, height = 300, 400  # Official ratio close to 3:4
border = 12
radius = 28  # Rounded corner radius
padding = int(height * 0.07)
bg = (255,255,255)
fg = (0,0,0)

SPEED_LIMIT_FONT_SIZE = int(height * 0.13)
NUMERAL_FONT_SIZE = int(height * 0.48)

try:
    sl_font = ImageFont.truetype("DejaVuSans-Bold.ttf", SPEED_LIMIT_FONT_SIZE)
    num_font = ImageFont.truetype("DejaVuSans-Bold.ttf", NUMERAL_FONT_SIZE)
except Exception:
    sl_font = ImageFont.load_default()
    num_font = ImageFont.load_default()

# All standard increments
speeds = list(range(5, 91, 5))

for sp in speeds:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded rectangle mask (anti-aliased corners)
    mask = Image.new('L', (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, width - 1, height - 1], radius, fill=255)
    img.paste(bg, mask=mask)

    # Border (draw on top)
    draw.rounded_rectangle(
        [border // 2, border // 2, width - border // 2, height - border // 2],
        radius, outline=fg, width=border
    )

    # "SPEED LIMIT" - stack and center in upper half
    y_offset = padding
    bbox1 = draw.textbbox((0, 0), "SPEED", font=sl_font)
    w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    bbox2 = draw.textbbox((0, 0), "LIMIT", font=sl_font)
    w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    label_block_height = h1 + h2 + int(0.10*h1)
    label_y = y_offset
    draw.text(((width-w1)//2, label_y), "SPEED", font=sl_font, fill=fg)
    draw.text(((width-w2)//2, label_y + h1 + int(0.10*h1)), "LIMIT", font=sl_font, fill=fg)

    # Numeral - centered in remaining space below
    num_str = str(sp)
    bbox_num = draw.textbbox((0, 0), num_str, font=num_font)
    nw, nh = bbox_num[2] - bbox_num[0], bbox_num[3] - bbox_num[1]
    numeral_y = label_y + label_block_height + int(h1 * 0.5)
    draw.text(((width-nw)//2, numeral_y), num_str, font=num_font, fill=fg)

    # Save to PNG
    img.save(f"{sp}.png")
    print(f"Saved {sp}.png")
