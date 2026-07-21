# main.py

from src.gpu_setup import *

from src.visualize import *

from src.train import *

from src.evaluate import *

from src.config import *


def main():

    setup_gpu()

    show_devices()

    display_samples(

        DATASET_PATH

    )

    train()

    evaluate_model()


if __name__ == "__main__":

    main()