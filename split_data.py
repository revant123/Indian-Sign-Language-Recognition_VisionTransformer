import os
import random
import shutil

# SOURCE DATA
source_dir = "dataset_all"
base_dir = "dataset"

# CREATE FOLDERS
for split in ["train", "val", "test"]:
    for cls in os.listdir(source_dir):
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# SPLIT RATIO
train_ratio = 0.7
val_ratio = 0.2

# PROCESS EACH CLASS
for cls in os.listdir(source_dir):
    class_path = os.path.join(source_dir, cls)

    if not os.path.isdir(class_path):
        continue

    files = os.listdir(class_path)
    random.shuffle(files)

    total = len(files)

    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # COPY FILES
    for f in train_files:
        shutil.copy(os.path.join(class_path, f),
                    os.path.join(base_dir, "train", cls, f))

    for f in val_files:
        shutil.copy(os.path.join(class_path, f),
                    os.path.join(base_dir, "val", cls, f))

    for f in test_files:
        shutil.copy(os.path.join(class_path, f),
                    os.path.join(base_dir, "test", cls, f))

    print(f"{cls}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

print("✅ DATASET SPLIT DONE")