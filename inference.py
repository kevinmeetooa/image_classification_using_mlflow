"""Inference module used to run predictions on unseen data."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        type=str, default="", required=True,
                        help='Test dataset path')
    parser.add_argument('--classes', nargs='+', help='Classes names',
                        required=False,
                        default=['test_images'])
    parser.add_argument('--model_path',
                        type=str, default="", required=True,
                        help='Model path')
    parser.set_defaults(func=main)
    return parser


def main():
    args, _ = build_parser().parse_known_args()

    model = tf.keras.models.load_model(args.model_path)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    test_generator = test_datagen.flow_from_directory(
        args.dataset_path,
        color_mode="rgb",
        target_size=(300, 200),
        batch_size=12,
        shuffle=False,
        classes=args.classes)

    predictions = model.predict(test_generator)
    class_labels = ['fields', 'roads']

    # Print images and corresponding predictions
    for i in range(len(test_generator.filenames)):
        image_path = os.path.join("dataset", test_generator.filenames[i])
        img = plt.imread(image_path)

        plt.imshow(img)
        plt.title(
            f"Prediction: {class_labels[int(np.argmax(predictions, axis=1)[i])]}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
