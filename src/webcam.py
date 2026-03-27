import torch
from torchvision import transforms, datasets
import timm
import cv2
from PIL import Image
import time
from collections import Counter

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
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# VARIABLES
# -----------------------------
word = ""
sentence = ""

last_added_time = time.time()
last_detection_time = time.time()

CONF_THRESHOLD = 0.85
COOLDOWN = 1.2

frame_count = 0
FRAME_SKIP = 3   # reduce lag

pred_buffer = []   # for stability

# -----------------------------
# LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ROI
    size = 300
    x1 = w//2 - size//2
    y1 = h//2 - size//2
    x2 = x1 + size
    y2 = y1 + size

    roi = frame[y1:y2, x1:x2]

    frame_count += 1

    # -----------------------------
    # SKIP FRAMES (FASTER)
    # -----------------------------
    if frame_count % FRAME_SKIP == 0:

        roi_blur = cv2.GaussianBlur(roi, (5,5), 0)
        img = Image.fromarray(cv2.cvtColor(roi_blur, cv2.COLOR_BGR2RGB))
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        confidence = conf.item()

        if confidence > CONF_THRESHOLD:
            letter = classes[pred.item()]
            pred_buffer.append(letter)
            last_detection_time = time.time()
        else:
            pred_buffer.append("")

        # keep buffer small
        if len(pred_buffer) > 10:
            pred_buffer.pop(0)

        # -----------------------------
        # MAJORITY VOTING (STABILITY)
        # -----------------------------
        most_common = Counter(pred_buffer).most_common(1)[0][0]

        # -----------------------------
        # ADD LETTER
        # -----------------------------
        if most_common != "" and pred_buffer.count(most_common) > 6:
            current_time = time.time()

            if current_time - last_added_time > COOLDOWN:
                word += most_common
                print("Added:", most_common)

                last_added_time = current_time
                pred_buffer.clear()

        # -----------------------------
        # WORD BREAK (REAL PAUSE)
        # -----------------------------
        if time.time() - last_detection_time > 2.5:
            if word != "":
                sentence += word + " "
                print("Word added:", word)

                word = ""
                pred_buffer.clear()

    # -----------------------------
    # DRAW UI
    # -----------------------------
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    display_letter = pred_buffer[-1] if pred_buffer else ""

    cv2.putText(frame, f"Letter: {display_letter}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Word: {word}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(frame, f"Sentence: {sentence}", (20,150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()