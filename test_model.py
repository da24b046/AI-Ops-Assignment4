from PIL import Image, ImageDraw
from model_utils import generate_caption, load_model

image = Image.new("RGB", (400, 300), "skyblue")
draw = ImageDraw.Draw(image)

draw.ellipse((130, 60, 270, 200), fill="yellow")
draw.rectangle((0, 230, 400, 300), fill="forestgreen")

image.save("test_images/sample.png")

processor, model = load_model()

print("Model device:", next(model.parameters()).device)

caption, duration = generate_caption(image)

print("\n========== TEST RESULT ==========")
print("Caption:", caption)
print(f"Inference time: {duration:.2f} seconds")
print("=================================")
