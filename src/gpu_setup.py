import tensorflow as tf


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print("GPU Enabled")
        print(f"GPU Device: {gpus[0].name}")
    else:
        print("CPU Mode")


def show_devices():
    print("Available Devices:")

    for device in tf.config.list_physical_devices():
        print(device)