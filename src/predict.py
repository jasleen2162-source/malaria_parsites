from PIL import Image

import numpy as np

from tensorflow.keras.models import load_model

from src.config import *


def predict(path):

    model=load_model(MODEL_PATH)

    img=Image.open(path)

    img=img.resize(

      IMAGE_SIZE

    )

    img=np.array(img)

    img=img/255

    img=np.expand_dims(

      img,

      axis=0

    )

    pred=model.predict(img)

    if pred>0.5:

        print("Uninfected")

    else:

        print("Parasitized")