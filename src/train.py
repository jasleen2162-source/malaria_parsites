from src.preprocess import *

from src.model import *

from src.config import *


def train():

    (

    X_train,

    y_train,

    X_val,

    y_val,

    _,

    _

    )=split_dataset()

    train_ds=create_tf_dataset(

      X_train,

      y_train,

      True

    )

    val_ds=create_tf_dataset(

      X_val,

      y_val

    )

    model=build_model()

    history=model.fit(

       train_ds,

       validation_data=val_ds,

       epochs=EPOCHS

    )

    model.save(

      MODEL_PATH

    )

    return history