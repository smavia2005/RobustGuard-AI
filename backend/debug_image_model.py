
from pathlib import Path
import sys
import torch
import transformers

from PIL import Image, ImageOps
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor
)

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

torch.set_num_threads(2)

print("\n=== ROBUSTGUARD IMAGE DIAGNOSTIC ===")
print("Transformers version:", transformers.__version__)
print("PyTorch version:", torch.__version__)

print("\nLoading model and processor...")

processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(MODEL_ID)

model.to("cpu")
model.eval()

print("\n=== MODEL CONFIG ===")
print("Number of labels:", model.config.num_labels)
print("Attention heads:", model.config.num_attention_heads)
print("Intermediate size:", model.config.intermediate_size)

print("\n=== PREPROCESSING CONFIG ===")
print("Resize enabled:", processor.do_resize)
print("Resize size:", processor.size)
print("Center crop:", processor.do_center_crop)
print("Crop size:", processor.crop_size)
print("Normalise:", processor.do_normalize)
print("Mean:", processor.image_mean)
print("Std:", processor.image_std)

if not processor.do_center_crop:
    print("WARNING: Center crop is disabled!")

if not processor.do_resize:
    print("WARNING: Resize is disabled!")

if len(sys.argv) < 2:
    print("\nUsage:")
    print('python debug_image_model.py "C:\\path\\image.png"')
    sys.exit(1)

for filename in sys.argv[1:]:
    path = Path(filename)

    if not path.is_file():
        print("\nFile not found:", path)
        continue

    print("\n" + "=" * 45)
    print("IMAGE:", path.name)

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    print("Original resolution:", image.size)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixels = inputs["pixel_values"]

    print("Tensor shape:", tuple(pixels.shape))
    print("Pixel minimum:", round(pixels.min().item(), 5))
    print("Pixel maximum:", round(pixels.max().item(), 5))
    print("Pixel mean:", round(pixels.mean().item(), 5))
    print("Pixel std:", round(pixels.std().item(), 5))

    with torch.inference_mode():
        output = model(**inputs)

    logits = output.logits

    print("Logit shape:", tuple(logits.shape))
    print("Raw logit:", logits.flatten().tolist())

    if logits.numel() != 1:
        print("ERROR: Expected exactly one output logit.")
        continue

    logit = logits.flatten()[0]

    fake_probability = torch.sigmoid(logit).item()
    real_probability = 1.0 - fake_probability

    prediction = (
        "FAKE" if fake_probability >= 0.5 else "REAL"
    )

    print("\n=== ACTUAL AI RESULT ===")
    print("Fake probability:", f"{fake_probability * 100:.4f}%")
    print("Real probability:", f"{real_probability * 100:.4f}%")
    print("Prediction:", prediction)

    if fake_probability < 0.5:
        print("Reason: Fake probability is below 50%.")
    else:
        print("Reason: Fake probability is at least 50%.")

print("\nDIAGNOSTIC COMPLETE")
