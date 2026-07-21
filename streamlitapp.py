import streamlit as st

from PIL import Image

import numpy as np

from tensorflow.keras.models import load_model

import sys

sys.path.append("src")

from config import *


st.set_page_config(

    page_title="Malaria Detector",

    layout="wide"

)

st.title(

    "🩸 Malaria Parasite Detector"

)

model = load_model(

    MODEL_PATH

)

uploaded_file = st.file_uploader(

    "Upload Blood Smear Image",

    type=["png","jpg","jpeg"]

)


if uploaded_file:

    image = Image.open(

        uploaded_file

    )

    st.image(

        image,

        width=300

    )

    img = image.resize(

        IMAGE_SIZE

    )

    img = np.array(img)

    img = img/255.0

    img = np.expand_dims(

        img,

        axis=0

    )

    probability = model.predict(

        img

    )[0][0]

    if probability > 0.5:

        prediction = "Uninfected"

        confidence = probability

    else:

        prediction = "Parasitized"

        confidence = 1-probability

    st.success(

        f"Prediction : {prediction}"

    )

    st.write(

        f"Confidence : {confidence*100:.2f}%"

    )