from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "training-pipeline.png"

width, height = 1200, 700
bg_color = "#111827"
box_color = "#1f2937"
text_color = "#f9fafb"
accent_color = "#22c55e"
line_color = "#6b7280"
font_path = None

try:
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 32)
    small_font = ImageFont.truetype(str(font_path), 24)
except Exception:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

img = Image.new("RGB", (width, height), color=bg_color)
draw = ImageDraw.Draw(img)

boxes = [
    "Llama 3.1-8B / Qwen 2.5-7B",
    "SFT \u2192 Alpaca52K",
    "DPO \u2192 PKU-SafeRLHF",
    "AdvBench Safety",
    "MMLU Knowledge",
    "GSM8K Reasoning",
]

box_width = 760
box_height = 90
box_x = (width - box_width) // 2
start_y = 80
spacing = 100
arrow_x = width // 2

for idx, label in enumerate(boxes):
    y = start_y + idx * spacing
    draw.rounded_rectangle([box_x, y, box_x + box_width, y + box_height], radius=20, fill=box_color)
    draw.text(
        (box_x + 30, y + 24),
        label,
        font=font,
        fill=text_color,
    )
    if idx < len(boxes) - 1:
        arrow_y = y + box_height + 10
        draw.line([arrow_x, arrow_y, arrow_x, arrow_y + spacing - 20], fill=line_color, width=7)
        draw.polygon([
            (arrow_x - 15, arrow_y + spacing - 20),
            (arrow_x + 15, arrow_y + spacing - 20),
            (arrow_x, arrow_y + spacing - 2),
        ], fill=accent_color)

# Header text
header = "LLM Safety Alignment Pipeline"
subheader = "Fine-tuning into DPO, then safety + capability evaluation"
draw.text((60, 12), header, font=ImageFont.truetype(str(font_path), 42) if font_path else font, fill=accent_color)
draw.text((60, 60), subheader, font=small_font, fill="#d1d5db")

img.save(output_path)
print(f"Created: {output_path}")
