import csv
import math
import os

import click
import numpy as np
from PIL import Image
from keras.src.applications.mobilenet_v2 import preprocess_input
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

k_logistic_regression = 'logistic_regression'
k_nearest_neighbors = 'knn'
k_decision_tree = 'decision_tree'


def convert_label_to_int(label: str) -> int:
    return 1 if label == 'human' else 0


def convert_int_to_label(value: int) -> str:
    return 'human' if value == 1 else 'animal'


image_names = []


def load_data(image_folder: str, label_file: str):
    """ Loads images and labels from the specified folder and file."""
    # load labels file
    labels = []

    with open(label_file, 'r') as file:
        reader = csv.reader(file, delimiter="|")
        next(reader)
        for row in reader:
            image_names.append(row[0])
            labels.append(convert_label_to_int(row[3]))

    # load corresponding images
    images = []

    for name in image_names:
        image = imread(os.path.join(image_folder, name))
        resized_image = resize(image, (224, 224))
        grayscale_image = rgb2gray(resized_image)
        images.append(grayscale_image)

    return np.array(images), np.array(labels)


def vectorize_images(images: list) -> np.ndarray:
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

    processed_images = []

    for img_array in images:
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)

        img = Image.fromarray((img_array * 255).astype('uint8')).convert('RGB')
        img = img.resize((224, 224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = base_model.predict(x)
        processed_images.append(features.flatten())

    X = np.stack(processed_images, axis=0)
    return X


def validation_split(X: np.ndarray, y: np.ndarray, test_size: float=0.2):
    """ Splits data into train and test."""
    return train_test_split(X, y, test_size=test_size, random_state=42)


def create_model(model_name: str):
    if model_name == k_logistic_regression:
        return LogisticRegression(
            max_iter=5000,
            C=0.1,
            random_state=42,
            class_weight='balanced'
        )
    elif model_name == k_nearest_neighbors:
        return KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='euclidean'
        )
    elif model_name == k_decision_tree:
        return DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
        )


def train(X_train, y_train, X_test, y_test, model, strategy):
    y_pred = None
    scores = []
    if strategy == 'default':
        X_train_sub, X_val, y_train_sub, y_val = validation_split(X_train, y_train)
        model.fit(X_train_sub, y_train_sub)
        y_pred = model.predict(X_test)
    elif strategy == 'kfold':
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        for train_index, val_index in kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_val_fold, y_val_fold)
            scores.append(score)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif strategy == 'stratified_kfold':
        k = 5
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        for train_index, val_index in kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_val_fold, y_val_fold)
            scores.append(score)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred


def error_analysis(X_test, y_test, y_pred, images, model_name, validation_strategy):
    misclassified_indices = np.where(y_test != y_pred)[0]
    num_misclassified = len(misclassified_indices)
    print(f"Number of misclassified labels: {num_misclassified}")

    cols = 5
    rows = math.ceil(num_misclassified / cols)

    plt.figure(figsize=(cols * 3, rows * 3))

    for i, idx in enumerate(misclassified_indices):
        plt.subplot(rows, cols, i + 1)
        image = images[idx].reshape(224, 224)
        plt.imshow(image, cmap='gray')
        true_label = convert_int_to_label(y_test[idx])
        predicted_label = convert_int_to_label(y_pred[idx])
        plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"part2/{model_name}_{validation_strategy}_analysis.png")
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['human', 'animal'])
    disp.plot()
    plt.savefig(f"part2/{model_name}_{validation_strategy}_confusion.png")
    plt.close()


@click.command()
@click.option("--image_folder", type=str, help="Path to the folder containing images")
@click.option("--label_file", type=str, help="Path to the file containing labels")
@click.option("--model_name", type=str, help="Name of the model to use")
@click.option("--validation_strategy", type=str, default='default', help="Validation strategy to use")
@click.option("--test_size", type=float, default=0.2, help="Size of the test split")
def main(image_folder: str, label_file: str, model_name: str, test_size: float, validation_strategy: str):
    # Create dataset of image <-> label pairs
    images, labels = load_data(image_folder, label_file)

    # preprocess images and labels
    X = vectorize_images(images)
    y = labels

    # split data into train and test
    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        X, y, images, test_size=test_size, random_state=42, stratify=y
    )

    # create model
    model = create_model(model_name)

    # Train model using different validation strategies (refere to https://scikit-learn.org/stable/modules/cross_validation.html)
    # 1. Train, validation, test splits: so you need to split train into train and validation
    # 2. K-fold cross-validation: apply K-fold cross-validation on train data
    # 3. Leave-one-out cross-validation: apply Leave-one-out cross-validation on train data

    # Make a prediction on test data
    accuracy, y_pred = train(X_train, y_train, X_test, y_test, model, strategy=validation_strategy)

    # Calculate accuracy
    print(f"Accuracy: {accuracy:.2f}")

    # Make error analysis
    # 1. Plot the first 10 test images, and on each image plot the corresponding prediction
    # 2. Plot the confusion matrix
    error_analysis(X_test, y_test, y_pred, images_test, model_name, validation_strategy)


if __name__ == "__main__":
    main()
