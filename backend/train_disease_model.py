
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR / "datasets" / "blackgram_disease" / "BPLD Dataset"
)

MODEL_DIR = BASE_DIR / "ml_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# ==========================================
# 2. COLLECT IMAGE PATHS AND LABELS
# ==========================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

class_names = sorted(
    folder.name for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
)

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

image_paths = []
labels = []

for label, class_name in enumerate(class_names):
    class_folder = DATASET_DIR / class_name

    for image_path in class_folder.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            image_paths.append(str(image_path))
            labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("Classes:", class_names)
print("Total images:", len(image_paths))

# ==========================================
# 3. STRATIFIED 70/15/15 DATA SPLIT
# ==========================================

train_paths, remaining_paths, train_labels, remaining_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels
    )
)

validation_paths, test_paths, validation_labels, test_labels = (
    train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=remaining_labels
    )
)

print("\nDataset split:")
print("Training:", len(train_paths))
print("Validation:", len(validation_paths))
print("Testing:", len(test_paths))

# ==========================================
# 4. IMAGE LOADING PIPELINE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)

    return image, label


def create_dataset(paths, targets, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, targets)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


train_ds = create_dataset(
    train_paths, train_labels, training=True
)

validation_ds = create_dataset(
    validation_paths, validation_labels
)

# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10)
])

# ==========================================
# 6. MOBILE NET V2 BASE MODEL
# ==========================================

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ==========================================
# 7. BUILD CLASSIFIER
# ==========================================

inputs = keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)

x = keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

# ==========================================
# 8. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 9. TRAINING CALLBACKS
# ==========================================

checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_DIR / "blackgram_disease_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# ==========================================
# 10. TRAIN MODEL
# ==========================================

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

# ==========================================
# 11. SAVE CLASS NAMES
# ==========================================

with open(MODEL_DIR / "disease_classes.json", "w") as file:
    json.dump(class_names, file, indent=4)

# ==========================================
# 12. SAVE UNSEEN TEST IMAGE LIST
# ==========================================

test_manifest = {
    "class_names": class_names,
    "images": [
        {
            "path": str(Path(path).relative_to(BASE_DIR)),
            "label": int(label)
        }
        for path, label in zip(test_paths, test_labels)
    ]
}

with open(MODEL_DIR / "test_manifest.json", "w") as file:
    json.dump(test_manifest, file, indent=4)

print("\nTraining completed.")
print("Best model saved.")
print("Independent test images reserved:", len(test_paths))
print("Test manifest saved.")

print("\nDo not use the test images for further training.")
