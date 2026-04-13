#!/usr/bin/env python3
"""
Generate SMALL QR codes for testing minimum readable size
"""

import os
import qrcode
from PIL import Image, ImageDraw, ImageFont

def generate_size_test_sheet():
    """Generate QR codes at different sizes to test minimum readable size."""

    skus = [
        "SKU-001",
        "SKU-002",
        "SKU-003",
        "SKU-004",
        "SKU-005",
        "SKU-006",
    ]

    # Different sizes to test (in pixels at 300 DPI)
    # 300 DPI means: 118 pixels = 1cm
    sizes = [
        (354, "3cm"),   # Large - should always work
        (236, "2cm"),   # Medium - should work
        (177, "1.5cm"), # Small - might work
        (118, "1cm"),   # Very small - challenging
        (89, "0.75cm"), # Tiny - probably won't work
        (59, "0.5cm"),  # Super tiny - unlikely to work
    ]

    # A4 at 300 DPI
    a4_width = 2480
    a4_height = 3508

    sheet = Image.new('RGB', (a4_width, a4_height), 'white')
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
        small_font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
        small_font = font

    # Title
    draw.text((100, 50), "QR Size Test - Find minimum readable size", fill='black', font=font)
    draw.text((100, 100), "Print at 100% scale on A4", fill='gray', font=small_font)

    # Generate grid
    y_pos = 200
    margin = 100

    for row, (size, size_label) in enumerate(sizes):
        # Row label
        draw.text((margin, y_pos + size//2 - 15), f"{size_label}:", fill='black', font=small_font)

        x_pos = margin + 150

        for col, sku in enumerate(skus):
            # Generate QR
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=1,
            )
            qr.add_data(f"{sku}-{size_label}")
            qr.make(fit=True)

            qr_img = qr.make_image(fill_color="black", back_color="white")
            qr_img = qr_img.resize((size, size), Image.LANCZOS)

            # Convert to RGB if needed
            if qr_img.mode != 'RGB':
                qr_img = qr_img.convert('RGB')

            # Paste
            sheet.paste(qr_img, (x_pos, y_pos))

            x_pos += size + 50

        y_pos += size + 80

    # Footer
    draw.text((100, a4_height - 100),
              "Smallest size that scans = your minimum QR size for this camera distance",
              fill='gray', font=small_font)

    # Save
    output_dir = os.path.dirname(__file__)
    png_path = os.path.join(output_dir, "qr_size_test.png")
    pdf_path = os.path.join(output_dir, "qr_size_test.pdf")

    sheet.save(png_path, "PNG", dpi=(300, 300))
    sheet.save(pdf_path, "PDF", resolution=300)

    print(f"\nGenerated size test sheet!")
    print(f"\nSizes to test:")
    for size, label in sizes:
        print(f"  {label} - {size}px")

    print(f"\nSaved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"\nPrint at 100% and test which row scans!")


if __name__ == "__main__":
    print("=" * 50)
    print("  QR SIZE TEST GENERATOR")
    print("=" * 50)
    generate_size_test_sheet()
