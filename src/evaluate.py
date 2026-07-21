# src/evaluate.py

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix,

    roc_curve

)

from tensorflow.keras.models import load_model

from src.preprocess import (

    split_dataset,

    create_tf_dataset

)

from src.config import MODEL_PATH


def evaluate_model():

    _,_,_,_,X_test,y_test = split_dataset()

    test_ds = create_tf_dataset(

        X_test,

        y_test

    )

    model = load_model(

        MODEL_PATH

    )

    y_prob = model.predict(

        test_ds

    ).flatten()

    y_pred = (

        y_prob > 0.5

    ).astype(int)

    accuracy = accuracy_score(

        y_test,

        y_pred

    )

    precision = precision_score(

        y_test,

        y_pred,

        zero_division=0

    )

    recall = recall_score(

        y_test,

        y_pred,

        zero_division=0

    )

    f1 = f1_score(

        y_test,

        y_pred

    )

    roc_auc = roc_auc_score(

        y_test,

        y_prob

    )

    cm = confusion_matrix(

        y_test,

        y_pred

    )

    tn,fp,fn,tp = cm.ravel()

    specificity = tn/(tn+fp)

    print()

    print(f"Accuracy     : {accuracy:.4f}")

    print(f"Precision    : {precision:.4f}")

    print(f"Recall       : {recall:.4f}")

    print(f"Specificity  : {specificity:.4f}")

    print(f"F1 Score     : {f1:.4f}")

    print(f"ROC AUC      : {roc_auc:.4f}")

    print()

    print(cm)

    plot_confusion_matrix(cm)

    plot_roc_curve(

       y_test,

       y_prob

    )


def plot_confusion_matrix(cm):

    plt.figure(

      figsize=(6,6)

    )

    sns.heatmap(

      cm,

      annot=True,

      fmt="d"

    )

    plt.xlabel(

      "Predicted"

    )

    plt.ylabel(

      "Actual"

    )

    plt.savefig(

      "reports/confusion_matrix.png"

    )

    plt.close()


def plot_roc_curve(

    y_test,

    y_prob

):

    fpr,tpr,_ = roc_curve(

        y_test,

        y_prob

    )

    plt.figure(

      figsize=(6,6)

    )

    plt.plot(

      fpr,

      tpr,

      label="ROC"

    )

    plt.plot(

      [0,1],

      [0,1],

      "--"

    )

    plt.xlabel(

      "False Positive Rate"

    )

    plt.ylabel(

      "True Positive Rate"

    )

    plt.title(

      "ROC Curve"

    )

    plt.legend()

    plt.savefig(

      "reports/roc_curve.png"

    )

    plt.close()


if __name__=="__main__":

    evaluate_model()