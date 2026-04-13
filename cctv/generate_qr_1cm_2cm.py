#!/usr/bin/env python3
"""
Generate 1cm and 2cm QR codes - sharp at 600 DPI
"""

import os
import qrcode
from PIL import Image, ImageDraw, ImageFont

def generate():
    skus = ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005"]

    # 600 DPI - 1cm = 236px, 2cm = 472px
    dpi = 600
    cm_to_px = dpi / 2.54  # pixels per cm

    sizes = [
        (int(2 * cm_to_px), "2cm"),
        (int(1 * cm_to_px), "1cm"),
    ]

    # A4 at 600 DPI
    a4_width = int(8.27 * dpi)
    a4_height = int(11.69 * dpi)

    sheet = Image.new('RGB', (a4_width, a4_height), 'white')
    draw = ImageDraw.Draw(sheet)

    try:
        title_font = ImageFont.truetype("arial.ttf", 80)
        label_font = ImageFont.truetype("arial.ttf", 50)
        small_font = ImageFont.truetype("arial.ttf", 35)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
        small_font = title_font

    # Title
    draw.text((100, 80), "QR Size Test: 2cm vs 1cm (600 DPI)", fill='black', font=title_font)
    draw.text((100, 180), "Print at 100% scale - NO fit-to-page!", fill='red', font=label_font)

    margin = 200
    y_pos = 350

    for size_px, size_label in sizes:
        # Row header
        draw.text((margin, y_pos + size_px//2 - 30), f"{size_label}:", fill='black', font=label_font)

        x_pos = margin + 300

        for i, sku in enumerate(skus):
            # Generate sharp QR
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=30,  # Large for sharpness
                border=2,
            )
            qr.add_data(f"{sku}")
            qr.make(fit=True)

            qr_img = qr.make_image(fill_color="black", back_color="white")
            if qr_img.mode != 'RGB':
                qr_img = qr_img.convert('RGB')

            # Resize with NEAREST for sharp pixels
            qr_img = qr_img.resize((size_px, size_px), Image.NEAREST)

            # Paste QR
            sheet.paste(qr_img, (x_pos, y_pos))

            # Label
            draw.text((x_pos, y_pos + size_px + 20), sku, fill='black', font=small_font)

            x_pos += size_px + 150

        y_pos += size_px + 250

    # Ruler guide
    ruler_y = y_pos + 100
    draw.text((margin, ruler_y), "Reference ruler (measure to verify print scale):", fill='gray', font=small_font)

    ruler_y += 60
    # 1cm marks
    for i in range(11):
        x = margin + int(i * cm_to_px)
        draw.line([(x, ruler_y), (x, ruler_y + 50)], fill='black', width=3)
        if i % 2 == 0:
            draw.text((x - 15, ruler_y + 55), f"{i}", fill='black', font=small_font)

    draw.line([(margin, ruler_y + 25), (margin + int(10 * cm_to_px), ruler_y + 25)], fill='black', width=2)
    draw.text((margin + int(10 * cm_to_px) + 20, ruler_y + 10), "cm", fill='black', font=small_font)

    # Save
    output_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(output_dir, "qr_1cm_2cm.pdf")
    png_path = os.path.join(output_dir, "qr_1cm_2cm.png")

    sheet.save(pdf_path, "PDF", resolution=dpi)
    sheet.save(png_path, "PNG", dpi=(dpi, dpi))

    print(f"\nGenerated QR codes:")
    print(f"  2cm row: {int(2 * cm_to_px)}px ({2}cm)")
    print(f"  1cm row: {int(1 * cm_to_px)}px ({1}cm)")
    print(f"\nSaved:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")
    print(f"\n*** Print at 100% - use ruler to verify! ***")


if __name__ == "__main__":
    print("=" * 50)
    print("  1cm vs 2cm QR TEST (600 DPI)")
    print("=" * 50)
    generate()
