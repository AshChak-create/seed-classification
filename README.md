# Soybean Seed Classification Using Convolutional Neural Networks

**Image-based classification of soybean seeds using TensorFlow and Keras**

## Overview

Seed quality assessment often relies on visual inspection, where differences in surface texture, colour, and physical damage help distinguish healthy seeds from defective ones.

This project explores a deep learning approach to classifying soybean seed images into five categories based on their visible characteristics. A Convolutional Neural Network (CNN) was implemented using TensorFlow and Keras, covering the complete workflow from image preprocessing and augmentation to model training and validation.

Developed as part of my MSc coursework in Data Science and Big Data Analytics, the project provided an opportunity to work with image datasets and explore how convolutional networks learn visual features for multiclass classification.

## Dataset

The project report describes a dataset of **5,513 soybean seed images** distributed across five categories.

| Category | Description |
|---|---|
| Broken | Seeds with visible cracks or breakage |
| Intact | Fully developed seeds without visible damage |
| Skin Damaged | Seeds with damage to the outer surface |
| Spotted | Seeds showing spots or discolouration |
| Immature | Underdeveloped seeds |

The images are organized into separate training and validation directories, with individual folders for each class.

All images are resized to **227 × 227 pixels** before being passed to the model.

## Approach

### Image preprocessing and augmentation

Pixel intensities are normalized from the original 0–255 range to 0–1.

To introduce variation during training, the following augmentation techniques are applied:

- Random rotations of up to 20°
- Horizontal and vertical shifts of up to 20%
- Random zooming and shearing
- Horizontal flipping

Augmentation is applied only to the training images. Validation images are normalized without these additional transformations.

### CNN architecture

The model uses a Sequential architecture consisting of three convolutional blocks followed by fully connected layers.

| Layer | Configuration |
|---|---|
| Input | 227 × 227 × 3 RGB image |
| Conv2D + MaxPooling | 32 filters, 3 × 3 kernel, ReLU |
| Conv2D + MaxPooling | 64 filters, 3 × 3 kernel, ReLU |
| Conv2D + MaxPooling | 128 filters, 3 × 3 kernel, ReLU |
| Flatten | Converts feature maps into a vector |
| Dense | 256 neurons, ReLU |
| Dropout | Rate of 0.5 |
| Output | 5 neurons, Softmax |

The convolutional layers learn visual representations from the images, while the final softmax layer produces class probabilities for the five seed categories.

### Training configuration

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Loss function | Categorical Cross-Entropy |
| Batch size | 32 |
| Training epochs | 30 |
| Evaluation metric | Accuracy |

The model was trained using the training dataset and evaluated against a separate validation dataset.

## Results

The original coursework report records the following results:

| Metric | Reported result |
|---|---|
| Initial training accuracy | 26.15% |
| Later training accuracy | Approximately 71.88% |
| Best validation accuracy | **80.07%** |
| Reported validation loss | 0.5453 |
| Training duration | Approximately 2.5 hours on CPU |

The training history showed an overall improvement in accuracy, although fluctuations were observed during validation.

The reported results suggest that the CNN learned useful visual patterns for distinguishing the five categories. However, these figures are taken from the original project report and have not yet been independently reproduced.

An independent test-set evaluation would be needed to assess how well the model generalizes beyond the available validation data.

## Project Structure

```text
seed-classification/
├── main.py
├── README.md
├── dataset/
│   ├── train/
│   │   ├── Broken soybeans/
│   │   ├── Immature soybeans/
│   │   ├── Intact soybeans/
│   │   ├── Skin-damaged soybeans/
│   │   └── Spotted soybeans/
│   └── validation/
│       └── [same five class folders]
```

The dataset is stored locally. Its source and redistribution permissions should be confirmed before publishing the image files.

## Running the Project

Install TensorFlow and Matplotlib in a compatible Python environment.

```bash
pip install tensorflow matplotlib
```

Place the dataset in the directory structure described above, then run:

```bash
python main.py
```

The script trains the CNN for 30 epochs, evaluates validation performance, displays accuracy and loss curves, and saves the trained model as:

```text
soybean_classification_model.h5
```

## Limitations and Future Work

This project uses a straightforward CNN architecture rather than a pretrained or highly optimized image classification model.

Several extensions would help evaluate its performance more thoroughly:

- **Class-wise evaluation:** Generate a verified confusion matrix, precision, recall, and F1-score for each seed category.
- **Independent testing:** Evaluate the trained model on a separate test set.
- **Robustness testing:** Examine classification performance under changes in lighting, blur, and image quality.
- **Architecture comparison:** Compare the CNN against transfer-learning approaches such as ResNet or MobileNet.
- **Reproducibility:** Record dependency versions, dataset provenance, and consistent experimental settings.

## Project Context

This was an early academic project completed during the first semester of my MSc programme at MIT World Peace University, Pune, as part of the Research Methodology coursework.

The implementation uses established CNN techniques and served as an introduction to image classification, model training, and performance evaluation.

**Technologies:** Python · TensorFlow · Keras · CNN · Image Augmentation · Matplotlib