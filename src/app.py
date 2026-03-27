import torch
from torchvision import transforms, datasets
from PIL import Image
import timm
import tkinter as tk
from tkinter import filedialog
from tkinter import Label

# -----------------------------
# LOAD CLASSES
# -----------------------------
data = datasets.ImageFolder("../dataset/train")
classes = data.classes

# -----------------------------
# LOAD MODEL
# -----------------------------
model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load("../model.pth", map_location="cpu"))
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return classes[pred.item()], conf.item()

# -----------------------------
# UI FUNCTION
# -----------------------------
def open_image():
    file_path = filedialog.askopenfilename()

    if file_path:
        pred, conf = predict_image(file_path)

        result_label.config(
            text=f"Prediction: {pred}\nConfidence: {round(conf,2)}",
            fg="green"
        )

# -----------------------------
# UI SETUP
# -----------------------------
root = tk.Tk()
root.title("Hand Sign Recognition")
root.geometry("400x300")

btn = tk.Button(root, text="Upload Image", command=open_image, height=2, width=20)
btn.pack(pady=40)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()