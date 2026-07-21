# src/robustness.py

import cv2

import numpy as np

import matplotlib.pyplot as plt


def add_noise(image):

    noise = np.random.normal(

        0,

        0.05,

        image.shape

    )

    noisy_image = image + noise

    noisy_image = np.clip(

        noisy_image,

        0,

        1

    )

    return noisy_image


def add_blur(image):

    blurred_image = cv2.GaussianBlur(

        image,

        (5,5),

        0

    )

    return blurred_image


def save_robustness_report(image):

    noisy = add_noise(image)

    blurred = add_blur(image)

    fig,ax = plt.subplots(

        1,

        3,

        figsize=(15,5)

    )

    ax[0].imshow(image)

    ax[0].set_title("Original")

    ax[0].axis("off")

    ax[1].imshow(noisy)

    ax[1].set_title("Noise")

    ax[1].axis("off")

    ax[2].imshow(blurred)

    ax[2].set_title("Blur")

    ax[2].axis("off")

    plt.tight_layout()

    plt.savefig(

        "reports/robustness_report.png"

    )

    plt.close()