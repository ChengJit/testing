#!/usr/bin/env python3
"""
Generate QR codes for SKU testing - A4 printable sheet
"""

import os

# Install qrcode if needed
try:
    import qrcode
except ImportError:
    print("Installing qrcode...")
    os.system("pip install qrcode[pil]")
    import qrcode

from PIL import Image, ImageDraw, ImageFont

def generate_qr_sheet():
    """Generate 10 QR codes on A4 sheet."""

    # Sample SKUs - modify these to your actual SKUs
    skus = [
        "SKU-001-BOXLARGE",
        "SKU-002-BOXMED",
        "SKU-003-BOXSMALL",
        "D1V1450000001",
        "D1V1450000002",
        "D1V1450000003",
        "ITEM-A1-RACK01",
        "ITEM-B2-RACK02",
        "PRODUCT-12345",
        "PRODUCT-67890",
    ]

    # A4 size at 300 DPI
    a4_width = 2480   # 210mm
    a4_height = 3508  # 297mm

    # Create white A4 canvas
    sheet = Image.new('RGB', (a4_width, a4_height), 'white')
    draw = ImageDraw.Draw(sheet)

    # QR code settings
    qr_size = 400  # Size of each QR code
    margin = 100
    cols = 2
    rows = 5

    # Calculate spacing
    x_spacing = (a4_width - 2 * margin) // cols
    y_spacing = (a4_height - 2 * margin) // rows

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 36)
        small_font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        small_font = font

    # Title
    draw.text((a4_width//2 - 200, 30), "SKU QR Codes - Test Sheet", fill='black', font=font)

    # Generate QR codes
    for i, sku in enumerate(skus):
        row = i // cols
        col = i % cols

        # Position
        x = margin + col * x_spacing + (x_spacing - qr_size) // 2
        y = margin + 50 + row * y_spacing

        # Generate QR
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
            box_size=10,
            border=2,
        )
        qr.add_data(sku)
        qr.make(fit=True)

        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img = qr_img.resize((qr_size, qr_size), Image.LANCZOS)

        # Paste QR
        sheet.paste(qr_img, (x, y))

        # Add SKU text below QR
        text_x = x + qr_size // 2
        text_y = y + qr_size + 10

        # Center text
        bbox = draw.textbbox((0, 0), sku, font=small_font)
        text_width = bbox[2] - bbox[0]
        draw.text((text_x - text_width//2, text_y), sku, fill='black', font=small_font)

        # Add number
        draw.text((x + 5, y + 5), f"#{i+1}", fill='gray', font=small_font)

    # Save
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "qr_sheet.png")
    sheet.save(output_path, "PNG", dpi=(300, 300))

    # Also save as PDF for easier printing
    pdf_path = os.path.join(output_dir, "qr_sheet.pdf")
    sheet.save(pdf_path, "PDF", resolution=300)

    print(f"\nGenerated QR codes for {len(skus)} SKUs:")
    for i, sku in enumerate(skus, 1):
        print(f"  {i}. {sku}")

    print(f"\nSaved to:")
    print(f"  PNG: {output_path}")
    print(f"  PDF: {pdf_path}")
    print(f"\nPrint at 100% scale on A4 paper!")

    return output_path


def generate_individual_qrs():
    """Also generate individual QR images."""

    skus = [
        "SKU-001-BOXLARGE",
        "SKU-002-BOXMED",
        "SKU-003-BOXSMALL",
        "D1V1450000001",
        "D1V1450000002",
        "D1V1450000003",
        "ITEM-A1-RACK01",
        "ITEM-B2-RACK02",
        "PRODUCT-12345",
        "PRODUCT-67890",
    ]

    output_dir = os.path.join(os.path.dirname(__file__), "qr_codes")
    os.makedirs(output_dir, exist_ok=True)

    for i, sku in enumerate(skus, 1):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=20,
            border=4,
        )
        qr.add_data(sku)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Add text label
        img_with_label = Image.new('RGB', (img.size[0], img.size[1] + 60), 'white')
        img_with_label.paste(img, (0, 0))

        draw = ImageDraw.Draw(img_with_label)
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), sku, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(((img.size[0] - text_width) // 2, img.size[1] + 10), sku, fill='black', font=font)

        filename = f"qr_{i:02d}_{sku.replace('-', '_')}.png"
        img_with_label.save(os.path.join(output_dir, filename))

    print(f"\nIndividual QR codes saved to: {output_dir}")


if __name__ == "__main__":
    print("=" * 50)
    print("  QR CODE GENERATOR FOR SKU TESTING")
    print("=" * 50)

    generate_qr_sheet()
    generate_individual_qrs()

    print("\n" + "=" * 50)
    print("  DONE! Print qr_sheet.pdf on A4")
    print("=" * 50)
