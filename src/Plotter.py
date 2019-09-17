import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

warnings.simplefilter("ignore")


class Plotter:
    def __init__(self, reader):
        self.reader = reader

    def heatmap(self):
        sns.heatmap(self.reader.get_full_data().corr(), annot=True)
        plt.show()

    def kdeplot(self):
        full_data = self.reader.get_full_data()
        # sns.kdeplot(full_data['fixed acidity'], label="fixed acidity")
        # sns.kdeplot(full_data['citric acid'], label="citric acid")
        # sns.kdeplot(full_data['chlorides'], label="chlorides")
        # sns.kdeplot(full_data['free sulfur dioxide'], label="free sulfur dioxide")
        # sns.kdeplot(full_data['total sulfur dioxide'], label="total sulfur dioxide")
        # sns.kdeplot(full_data['pH'], label="pH")
        # sns.kdeplot(full_data['sulphates'], label="sulphates")
        sns.kdeplot(full_data['volatile acidity'], label="volatile acidity")
        sns.kdeplot(full_data['residual sugar'], label="residual sugar")
        sns.kdeplot(full_data['density'], label="density")
        sns.kdeplot(full_data['alcohol'], label="alcohol")
        plt.legend()
        plt.show()

    def pairplot(self):
        features = ['volatile acidity', 'residual sugar', 'density', 'alcohol']
        sns.pairplot(self.reader.get_full_data(), vars=features,
                     hue='quality')
        plt.show()

    def confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        classes = [0, 1, 2]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title="Confusion matrix",
               ylabel="True label",
               xlabel="Predicted label")
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                print(f"cm[{i}][{j}] = {cm[i][j]}")
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.subplots_adjust(hspace=.001)
        plt.show()

    def classification_report(self, y_true, y_pred, labels=[0, 1, 2]):
        print(classification_report(y_true, y_pred, labels))
