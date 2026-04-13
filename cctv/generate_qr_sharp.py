#!/usr/bin/env python3
"""
Generate SHARP high-resolution QR codes for printing
"""

import os
import qrcode
from PIL import Image, ImageDraw, ImageFont

def generate_sharp_qr():
    """Generate crisp, high-res QR codes."""

    skus = [
        "SKU-001",
        "SKU-002",
        "SKU-003",
        "SKU-004",
        "SKU-005",
        "SKU-006",
        "SKU-007",
        "SKU-008",
        "SKU-009",
        "SKU-010",
    ]

    # A4 at 600 DPI (higher = sharper print)
    dpi = 600
    a4_width = int(8.27 * dpi)   # 210mm
    a4_height = int(11.69 * dpi)  # 297mm

    sheet = Image.new('RGB', (a4_width, a4_height), 'white')
    draw = ImageDraw.Draw(sheet)

    # QR settings - use HIGH box_size for sharp edges
    # box_size=20 means each QR "pixel" is 20x20 actual pixels
    qr_box_size = 25  # Bigger = sharper
    qr_border = 4

    # Layout: 2 columns, 5 rows
    cols = 2
    rows = 5
    margin = int(0.5 * dpi)  # 0.5 inch margin

    # Calculate QR display size (about 3cm = ~70px at 600dpi * 3)
    qr_display_size = int(1.2 * dpi)  # ~3cm

    x_spacing = (a4_width - 2 * margin) // cols
    y_spacing = (a4_height - 2 * margin) // rows

    try:
        title_font = ImageFont.truetype("arial.ttf", int(dpi * 0.4))
        label_font = ImageFont.truetype("arial.ttf", int(dpi * 0.25))
    except:
        title_font = ImageFont.load_default()
        label_font = title_font

    # Title
    draw.text((margin, int(margin * 0.3)), "SKU QR Codes - Sharp Print (600 DPI)",
              fill='black', font=title_font)

    for i, sku in enumerate(skus):
        row = i // cols
        col = i % cols

        # Position
        x = margin + col * x_spacing + (x_spacing - qr_display_size) // 2
        y = margin + int(dpi * 0.5) + row * y_spacing

        # Generate QR at HIGH resolution
        qr = qrcode.QRCode(
            version=2,  # Slightly larger version for more data capacity
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # 30% error correction
            box_size=qr_box_size,
            border=qr_border,
        )
        qr.add_data(sku)
        qr.make(fit=True)

        # Create QR image - DO NOT resize (keep native resolution)
        qr_img = qr.make_image(fill_color="black", back_color="white")

        # Convert to RGB
        if qr_img.mode != 'RGB':
            qr_img = qr_img.convert('RGB')

        # Resize using NEAREST neighbor (keeps sharp edges!)
        qr_img = qr_img.resize((qr_display_size, qr_display_size), Image.NEAREST)

        # Paste
        sheet.paste(qr_img, (x, y))

        # Label below
        label = sku
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (qr_display_size - text_width) // 2
        text_y = y + qr_display_size + int(dpi * 0.1)
        draw.text((text_x, text_y), label, fill='black', font=label_font)

        # Number
        draw.text((x + 10, y + 10), f"#{i+1}", fill='gray', font=label_font)

    # Save at high DPI
    output_dir = os.path.dirname(__file__)

    png_path = os.path.join(output_dir, "qr_sharp.png")
    pdf_path = os.path.join(output_dir, "qr_sharp.pdf")

    sheet.save(png_path, "PNG", dpi=(dpi, dpi))
    sheet.save(pdf_path, "PDF", resolution=dpi)

    print(f"\nGenerated SHARP QR codes at {dpi} DPI")
    print(f"\nSKUs:")
    for i, sku in enumerate(skus, 1):
        print(f"  {i}. {sku}")

    print(f"\nSaved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"\n*** IMPORTANT: Print at 100% scale, no fit-to-page! ***")


def generate_individual_sharp():
    """Generate individual large QR images for manual printing/cutting."""

    skus = ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005",
            "SKU-006", "SKU-007", "SKU-008", "SKU-009", "SKU-010"]

    output_dir = os.path.join(os.path.dirname(__file__), "qr_individual")
    os.makedirs(output_dir, exist_ok=True)

    for i, sku in enumerate(skus, 1):
        # Very high resolution QR
        qr = qrcode.QRCode(
            version=2,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=50,  # Very large for sharp print
            border=4,
        )
        qr.add_data(sku)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save individual file
        filename = f"{i:02d}_{sku}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, "PNG", dpi=(600, 600))

    print(f"\nIndividual QR codes saved to: {output_dir}")
    print("Each QR is ~1500x1500 pixels - very sharp when printed!")


if __name__ == "__main__":
    print("=" * 50)
    print("  SHARP QR CODE GENERATOR (600 DPI)")
    print("=" * 50)

    generate_sharp_qr()
    generate_individual_sharp()

    print("\n" + "=" * 50)
    print("  PRINTING TIPS:")
    print("=" * 50)
    print("""
1. Open qr_sharp.pdf
2. Print settings:
   - Scale: 100% (NOT fit-to-page!)
   - Quality: Best/High
   - Paper: Plain white A4
3. Use a LASER printer if possible (sharper than inkjet)
4. Let ink dry before cutting
""")
