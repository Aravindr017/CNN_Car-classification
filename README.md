# Project: Algerian Used Car Classification

## Overview

This project aims to classify images of Algerian used cars into their respective makes or models. The goal is to build and evaluate various Convolutional Neural Network (CNN) models to accurately identify car types from a given image dataset.

## Dataset

The dataset used contains images of various Algerian used car models.

**Source:** [Kaggle: Algerian Used Cars](https://www.kaggle.com/datasets/boulahchichenadir/algerian-used-cars)

Each data point includes:

*   Images of cars from different angles and conditions.
*   Labels corresponding to 20 distinct car categories (makes/models) such as 'Golf', 'bmw serie 1', 'duster', 'hyundai i10', 'polo', 'toyota corolla', etc.

**Key Characteristics:**
*   **Image Data**: The primary feature for classification is image data. Images have varying quality and backgrounds, typical of real-world scenarios.
*   **Multi-class Classification**: The task involves classifying images into one of 20 distinct car categories.

## Preprocessing & Feature Engineering

The following preprocessing steps have been applied to the image data:

1.  **Image Loading and Resizing**: Images are loaded from their respective category directories, resized to a uniform `img_size` (e.g., 64x64 pixels), and converted to grayscale (`L` channel) to reduce dimensionality.

    ```python
    img_size = 64
    data = []
    labels = []
    # ... (loop through categories and load images)
    image = Image.open(image_path)
    image = image.resize((img_size, img_size)).convert('L')
    image_array = np.array(image)
    data.append(image_array)
    labels.append(categories.index(category))
    ```

2.  **Data Normalization**: Pixel values, which typically range from 0-255, are normalized to a 0.0-1.0 range by dividing by 255.0. This helps in faster and more stable training of neural networks.

    ```python
    data = data.astype('float32') / 255.0
    ```

3.  **Train-Test Split**: The dataset is split into training and testing sets, typically with an 80/20 ratio, to evaluate model performance on unseen data.

    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    ```

## Models Used

Several Convolutional Neural Network (CNN) models were experimented with:

### Base CNN Model (`model1`)

A sequential CNN model without explicit regularization (dropout) or padding.

**Architecture:**
*   `Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1))`
*   `MaxPooling2D((2, 2))`
*   `Conv2D(64, (3, 3), activation='relu')`
*   `MaxPooling2D((2, 2))`
*   `Conv2D(128, (3, 3), activation='relu')`
*   `MaxPooling2D((2, 2))`
*   `Flatten()`
*   `Dense(128, activation='relu')`
*   `Dense(len(categories), activation='softmax')`

### CNN Model with Dropout (`model2`)

This model adds `Dropout` layers to the base CNN architecture to combat overfitting.

**Architecture additions:**
*   `Dropout(0.25)` after each `MaxPooling2D` layer.
*   `Dropout(0.5)` before the final output `Dense` layer.

### CNN Model with 'same' Padding (`model3`)

This model modifies the base CNN by adding `padding='same'` to `Conv2D` layers to preserve spatial dimensions and prevent excessive reduction of feature map sizes.

**Architecture modification:**
*   `padding='same'` added to all `Conv2D` layers.

### CNN Model with 'same' Padding and Dropout (`model4`)

Combines both 'same' padding in convolutional layers and `Dropout` regularization for improved performance and generalization.

**Architecture combines features of `model2` and `model3`:**
*   `padding='same'` in `Conv2D` layers.
*   `Dropout(0.25)` after each `MaxPooling2D` layer.
*   `Dropout(0.5)` before the final output `Dense` layer.

**Training Details (for all models):**
*   **Optimizer**: Adam
*   **Loss Function**: `sparse_categorical_crossentropy` (since labels are integers)
*   **Metrics**: Accuracy
*   **Callbacks**: `EarlyStopping` (patience=10) was used to stop training when validation loss stopped improving, preventing further overfitting.

## Evaluation Metrics

The models were evaluated using the following metrics:

*   **Accuracy**: The proportion of correctly classified images.
*   **Loss**: The value of the loss function during training and validation.
*   **Training and Validation Loss/Accuracy Plots**: Visualizations showing the change in loss and accuracy over epochs for both training and validation sets.
*   **Prediction Visualization**: Displaying sample test images along with their true labels and the model's predicted labels, including prediction probabilities.

## Results and Analysis

### Key Observations:

*   **Base Model (model1)**: Showed signs of overfitting early, with training accuracy quickly approaching 1.0 while validation accuracy lagged and validation loss increased.
*   **Model with Dropout (model2)**: The introduction of dropout initially reduced the speed of overfitting and led to more stable training, but overall performance was still limited, with validation accuracy not improving significantly.
*   **Model with Padding (model3)**: Adding `padding='same'` also contributed to slightly better validation performance compared to the base model, likely by preserving more information in feature maps.
*   **Model with Padding and Dropout (model4)**: This combined model generally showed the best performance among the tested architectures, achieving higher validation accuracy and a more controlled training process. The accuracy on the test set for this model was approximately 63.5%.
*   **Overfitting**: Across all models, a common challenge was overfitting, indicating the models learned the training data too well but struggled to generalize to unseen validation/test data. This was evident from the divergence between training and validation loss/accuracy curves.

### Potential Improvements

1.  **Data Augmentation**: Implement techniques like rotation, shifting, zooming, and flipping on training images to increase the effective size of the training data and improve generalization.
2.  **Transfer Learning**: Utilize pre-trained CNN models (e.g., VGG16, ResNet, EfficientNet) on large image datasets (like ImageNet) and fine-tune them for this specific car classification task. This can significantly boost performance, especially with limited datasets.
3.  **Hyperparameter Tuning**: Conduct a more exhaustive search for optimal hyperparameters (e.g., learning rate, batch size, number of filters, kernel sizes, dropout rates) using techniques like GridSearchCV or RandomSearchCV.
4.  **More Complex Architectures**: Explore deeper or wider CNN architectures that might be better suited for capturing intricate features in the images.
5.  **Regularization**: Experiment with other regularization techniques beyond dropout, such as L1/L2 regularization or Batch Normalization, within the network architecture.
6.  **Image Quality and Preprocessing**: Further analyze the impact of image quality, noise, and different preprocessing techniques (e.g., histogram equalization, advanced color space conversions).

## How to Run the Project

### Prerequisites

*   Python 3.x
*   `tensorflow` (including Keras)
*   `numpy`
*   `matplotlib`
*   `scikit-learn`
*   `Pillow` (PIL) for image handling
*   `opencv-python` (cv2) for image operations

### Installation

Install the necessary libraries using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn Pillow opencv-python
```

### Code Structure (Conceptual)

The project typically follows these steps:

1.  **Import Libraries**: Essential libraries for deep learning, image processing, and data handling.

    ```python
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import cv2
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from PIL import Image
    ```

2.  **Mount Google Drive**: Access the dataset stored in Google Drive.

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3.  **Load Data**: Define directory paths and category names, then load and preprocess images.

    ```python
    directory = "/content/drive/MyDrive/AI ML/IRP/datasets/dataset/DATA"
    categories = ['Golf','bmw serie 1', ...]
    img_size = 64
    data = []
    labels = []
    # ... (image loading and processing loop)
    data = np.array(data)
    labels = np.array(labels)
    data = data.reshape(-1, img_size, img_size, 1) # Reshape for CNN input
    ```

4.  **Preprocessing**: Normalize pixel values and split into training and test sets.

    ```python
    data = data.astype('float32') / 255.0
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    ```

5.  **Model Building and Training**: Define, compile, and train the CNN models (e.g., `model4`).

    ```python
    model4 = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_size, img_size, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        # ... other layers
        Dense(len(categories), activation='softmax')
    ])
    model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model4.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    ```

6.  **Model Evaluation**: Evaluate the trained models on the test set and visualize training history.

    ```python
    test_loss, test_accuracy = model4.evaluate(X_test, y_test)
    print("Loss:", test_loss)
    print("Accuracy:", test_accuracy)

    # Plot training and validation curves
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.show()
    ```

Refer to the notebook for the full implementation details and specific code for each step.

## Contact

For any questions or suggestions, please open an issue in this repository.
