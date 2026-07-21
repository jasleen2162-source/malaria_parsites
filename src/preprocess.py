import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.config import *
from src.gpu_setup import *

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
)

# Data augmentation (applied only to training images)
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)


def create_dataset():
    files = []
    labels = []

    for cls in CLASS_NAMES:
        folder = Path(DATASET_PATH) / cls

        for image in folder.iterdir():
            if image.suffix.lower() not in VALID_EXTENSIONS:
                continue

            files.append(str(image))
            labels.append(
                0 if cls == "Parasitized" else 1
            )

    return files, labels


def split_dataset():
    files, labels = create_dataset()

    X_train, X_temp, y_train, y_temp = train_test_split(
        files,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=SEED,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=SEED,
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


def load_image(path, label, training=False):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    # Resize
    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    # Normalize to [0,1]
    image = tf.cast(
        image,
        tf.float32,
    ) / 255.0

    # Apply augmentation only during training
    if training:
        image = data_augmentation(image)

    return image, label


def create_tf_dataset(
    x,
    y,
    training=False,
):
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    ds = ds.map(
        lambda path, label: load_image(
            path,
            label,
            training,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds