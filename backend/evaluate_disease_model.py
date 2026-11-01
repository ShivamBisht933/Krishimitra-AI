
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# =====================================
# 1. PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml_models"

MODEL_PATH = MODEL_DIR / "blackgram_disease_model.keras"
MANIFEST_PATH = MODEL_DIR / "test_manifest.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# =====================================
# 2. LOAD MODEL AND TEST MANIFEST
# =====================================

if not MODEL_PATH.exists():
    raise FileNotFoundError("Trained model not found.")

if not MANIFEST_PATH.exists():
    raise FileNotFoundError("Test manifest not found.")

model = tf.keras.models.load_model(MODEL_PATH)

with open(MANIFEST_PATH, "r") as file:
    manifest = json.load(file)

class_names = manifest["class_names"]
test_images = manifest["images"]

print("Loaded model successfully.")
print("Number of test images:", len(test_images))
print("Classes:", class_names)

# =====================================
# 3. LOAD TEST IMAGES
# =====================================

def load_test_image(image_path):
    image = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)

    return image


X_test = []
y_true = []

for item in test_images:
    path = BASE_DIR / item["path"]

    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")

    X_test.append(load_test_image(path).numpy())
    y_true.append(item["label"])

X_test = np.array(X_test, dtype=np.float32)
y_true = np.array(y_true)

# =====================================
# 4. MODEL PREDICTIONS
# =====================================

predictions = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)
# =====================================
# 4A. CONFIDENCE AND RELIABILITY CHECK
# =====================================

confidence_scores = np.max(predictions, axis=1)

print("\nCONFIDENCE RELIABILITY ANALYSIS")

for threshold in [0.50, 0.60, 0.70, 0.80]:

    accepted = confidence_scores >= threshold

    accepted_count = np.sum(accepted)
    coverage = accepted_count / len(y_true) * 100

    if accepted_count > 0:
        accepted_accuracy = accuracy_score(
            y_true[accepted],
            y_pred[accepted]
        ) * 100
    else:
        accepted_accuracy = 0

    print(f"\nThreshold: {threshold * 100:.0f}%")
    print(f"Accepted predictions: {accepted_count}/{len(y_true)}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Accuracy on accepted predictions: {accepted_accuracy:.2f}%")
# =====================================
# 5. ACCURACY
# =====================================

accuracy = accuracy_score(y_true, y_pred)

print("\nINDEPENDENT TEST ACCURACY")
print(f"{accuracy * 100:.2f}%")

# =====================================
# 6. PRECISION, RECALL AND F1
# =====================================

print("\nCLASSIFICATION REPORT")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
)

print(report)

# =====================================
# 7. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 8))

display.plot(
    ax=ax,
    cmap="Greens",
    xticks_rotation=45,
    values_format="d"
)

plt.title("Black Gram Disease Detection - Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "confusion_matrix.png",
    dpi=300
)

plt.show()

# =====================================
# 8. SAVE TEST RESULTS
# =====================================

results = {
    "test_images": len(test_images),
    "test_accuracy": round(float(accuracy) * 100, 2),
    "class_names": class_names,
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
}

with open(MODEL_DIR / "evaluation_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nEvaluation completed.")
print("Saved confusion matrix and evaluation results.")
