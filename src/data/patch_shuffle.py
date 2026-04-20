import numpy as np
from PIL import Image
import os
import json
from datetime import datetime



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "1/train")
FRONTAL_FILENAME = "view1_frontal.jpg"
PATCH_SIZE = 32
TARGET_SIZE = (224, 224)


def patch_shuffle(img_array: np.ndarray, patch_size: int = 32) -> np.ndarray:
    H, W = img_array.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image dims must be divisible by patch_size={patch_size}"

    n_h = H // patch_size
    n_w = W // patch_size
    n_patches = n_h * n_w

    patches = img_array.reshape(n_h, patch_size, n_w, patch_size)
    patches = patches.transpose(0, 2, 1, 3)
    patches = patches.reshape(n_patches, patch_size, patch_size)

    idx = np.random.permutation(n_patches)
    patches = patches[idx]

    patches = patches.reshape(n_h, n_w, patch_size, patch_size)
    patches = patches.transpose(0, 2, 1, 3)
    return patches.reshape(H, W)


errors = []
processed = 0

patient_dirs = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])

for patient in patient_dirs:
    patient_path = os.path.join(TRAIN_DIR, patient)

    study_dirs = [
        d for d in os.listdir(patient_path)
        if os.path.isdir(os.path.join(patient_path, d))
    ]

    if not study_dirs:
        errors.append({"patient": patient, "error": "no study folders found"})
        continue

    for study in sorted(study_dirs):
        study_path = os.path.join(patient_path, study)
        input_path = os.path.join(study_path, FRONTAL_FILENAME)

        if not os.path.exists(input_path):
            errors.append({"patient": patient, "study": study, "error": "no frontal image found"})
            continue

        try:
            img = Image.open(input_path).convert("L").resize(TARGET_SIZE, Image.LANCZOS)
            img_array = np.array(img)
            shuffled = patch_shuffle(img_array, patch_size=PATCH_SIZE)

            output_filename = FRONTAL_FILENAME.replace(".jpg", "_PS.jpg")
            output_path = os.path.join(study_path, output_filename)
            Image.fromarray(shuffled).save(output_path)
            processed += 1
            if processed % 100 == 0:
                print(f"Progress: {processed} images processed...")
        except Exception as e:
            errors.append({"patient": patient, "study": study, "error": str(e)})

# Save error log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
error_log_path = os.path.join(BASE_DIR, f"patch_shuffle_errors_{timestamp}.json")
with open(error_log_path, "w") as f:
    json.dump(errors, f, indent=2)

print(f"Done. Processed: {processed} | Errors: {len(errors)}")
print(f"Error log saved to: {error_log_path}")