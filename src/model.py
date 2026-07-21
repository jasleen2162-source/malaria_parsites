# src/model.py

from tensorflow.keras import Sequential

from tensorflow.keras.layers import (

    Input,

    Conv2D,

    MaxPooling2D,

    Flatten,

    Dense,

    Dropout

)


def build_model():

    model = Sequential([

        Input(

            shape=(128,128,3)

        ),

        Conv2D(

            32,

            (3,3),

            activation="relu"

        ),

        MaxPooling2D(),

        Conv2D(

            64,

            (3,3),

            activation="relu"

        ),

        MaxPooling2D(),

        Conv2D(

            128,

            (3,3),

            activation="relu",

            name="last_conv"

        ),

        MaxPooling2D(),

        Flatten(),

        Dense(

            128,

            activation="relu"

        ),

        Dropout(

            0.5

        ),

        Dense(

            1,

            activation="sigmoid"

        )

    ])

    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    return model