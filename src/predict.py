import torch
from torchvision import transforms, datasets
from PIL import Image
import timm

# -----------------------------
# LOAD CLASSES
# -----------------------------
data = datasets.ImageFolder("../dataset/train")
classes = data.classes
print("Classes:", classes)

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
# INPUT IMAGE
# -----------------------------
path = input("Enter image path: ")

try:
    img = Image.open(path).convert("RGB")
except:
    print("❌ Image not found")
    exit()

img = transform(img).unsqueeze(0)

# -----------------------------
# PREDICT
# -----------------------------
with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)
    conf, pred = torch.max(probs, 1)

print("Predicted:", classes[pred.item()])
print("Confidence:", round(conf.item(), 3))