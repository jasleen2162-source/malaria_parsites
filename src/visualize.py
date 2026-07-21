import os

import matplotlib.pyplot as plt

from PIL import Image


def display_samples(path):

    parasitized=os.listdir(

      f"{path}/Parasitized"

    )

    uninfected=os.listdir(

      f"{path}/Uninfected"

    )

    fig,ax=plt.subplots(2,5)

    for i in range(5):

        img=Image.open(

         f"{path}/Parasitized/{parasitized[i]}"

        )

        ax[0,i].imshow(img)

        ax[0,i].axis("off")

        img=Image.open(

         f"{path}/Uninfected/{uninfected[i]}"

        )

        ax[1,i].imshow(img)

        ax[1,i].axis("off")

    plt.close()