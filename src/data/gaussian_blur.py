import numpy as np
from PIL import Image, ImageFilter
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "1/train")
FRONTAL_FILENAME = "view1_frontal.jpg"
BLUR_RADIUS = 2
TARGET_SIZE = (224, 224)


def gaussian_blur(img: Image.Image, radius: int = 2) -> np.ndarray:
    return np.array(img.filter(ImageFilter.GaussianBlur(radius=radius)))


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
            blurred = gaussian_blur(img, radius=BLUR_RADIUS)

            output_filename = FRONTAL_FILENAME.replace(".jpg", "_GB.jpg")
            output_path = os.path.join(study_path, output_filename)
            Image.fromarray(blurred).save(output_path)
            processed += 1
            if processed % 100 == 0:
                print(f"Progress: {processed} images processed...")

        except Exception as e:
            errors.append({"patient": patient, "study": study, "error": str(e)})

# Save error log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
error_log_path = os.path.join(BASE_DIR, f"gaussian_blur_errors_{timestamp}.json")
with open(error_log_path, "w") as f:
    json.dump(errors, f, indent=2)

print(f"Done. Processed: {processed} | Errors: {len(errors)}")
print(f"Error log saved to: {error_log_path}")