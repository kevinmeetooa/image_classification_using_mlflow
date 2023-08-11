"""Train module for the binary image classification task.
We use MLFlow in order to perform hyperparameter optimization as well as experiment, model and artifacts tracking.
"""
import argparse
import logging
import urllib
import zipfile

import tensorflow as tf
from hyperopt import STATUS_OK, fmin, hp, tpe
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import mlflow
import mlflow.sklearn
import mlflow.tensorflow

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

mlflow.tensorflow.autolog()


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int, default=30, required=False,
                        help='Maximum number of epochs for each training run')
    parser.add_argument('--max_evals',
                        type=int, default=50, required=False,
                        help='Maximum number of training runs')
    parser.set_defaults(func=main)
    return parser


def create_model(space: dict) -> tf.keras.models.Sequential:
    """Create a simple CNN for the binary image classification task."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=(300, 200, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(space["dropout"]),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    return model


def train_fn(space: dict) -> dict:
    """Train the model with given hyperparameters."""
    train_path = "dataset/"

    # We perform data augmentation in order to prevent our model from
    # overfitting on the simple dataset
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        color_mode="rgb",
        target_size=(300, 200),
        batch_size=int(space["batch_size"]),
        shuffle=True,
        classes=['fields', 'roads'],
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_path,
        color_mode="rgb",
        target_size=(300, 200),
        batch_size=int(space["batch_size"]),
        classes=['fields', 'roads'],
        subset='validation')

    with mlflow.start_run():
        model = create_model(space)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=space["learning_rate"]),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ],
        )

        # We perform early stopping to avoid overfitting
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    restore_best_weights=True)

        # We use class weights to make up for the class imbalance.
        # We've chosen these weights becauses there are 2.4 more images in
        # the class '1' (roads) than in the class '0' (fields)
        history = model.fit_generator(train_generator, epochs=space["epochs"],
                                      callbacks=[callback],
                                      validation_data=validation_generator,
                                      class_weight={0: 2.4, 1: 1})

        best_loss = min(history.history['val_loss'])

        mlflow.log_param("learning_rate", space['learning_rate'])
        mlflow.log_param("dropout", space['dropout'])

        return {"loss": best_loss, "status": STATUS_OK}


def main():
    args, _ = build_parser().parse_known_args()

    url = 'https://drive.google.com/u/1/uc?id=1BEYh0sGZ8-DvRM2Je2XdRntsbJ_kJEv_&export=download'
    output_path = "dataset.zip"

    # Download the dataset and unzip it
    urllib.request.urlretrieve(url, output_path)

    zip_file = zipfile.ZipFile("dataset.zip")
    zip_file.extractall(".")

    # Hyperparameters
    space = {
        "batch_size": hp.quniform("batch_size", 2, 16, 2),
        "learning_rate": hp.loguniform("learning_rate", -5, 0),
        "dropout": hp.uniform('dropout', .25, .75),
        "epochs": args.epochs
    }

    # Run hyperparameter optimization with hyperopt
    best_hyperparam = fmin(fn=train_fn, space=space, algo=tpe.suggest,
                           max_evals=args.max_evals)
    print(f'Best hyperparameters: {best_hyperparam}')

    df = mlflow.search_runs()
    run_id = df.loc[df['metrics.loss'].idxmin()]['run_id']
    model = mlflow.tensorflow.load_model("runs:/" + run_id + "/model")
    model.save("model/")
    print('Best model saved to model/')


if __name__ == "__main__":
    main()
