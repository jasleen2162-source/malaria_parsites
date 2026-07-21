from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model


def build_transfer_model():

    base=MobileNetV2(

      weights="imagenet",

      include_top=False,

      input_shape=(128,128,3)

    )

    base.trainable=False

    x=GlobalAveragePooling2D()(

      base.output

    )

    x=Dense(

      128,

      activation="relu"

    )(x)

    output=Dense(

      1,

      activation="sigmoid"

    )(x)

    model=Model(

      base.input,

      output

    )

    model.compile(

      optimizer="adam",

      loss="binary_crossentropy",

      metrics=["accuracy"]

    )

    return model