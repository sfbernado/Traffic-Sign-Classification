# Traffic-Sign-Classification

## Overview

The Traffic Sign Classification project aims to develop a robust deep-learning model capable of accurately identifying and classifying traffic signs from images. This is a crucial task for autonomous vehicle systems, where understanding and reacting to traffic signs correctly is essential for safe driving. The project is designed to tackle the challenges of recognizing traffic signs from diverse conditions and angles using convolutional neural networks (CNNs). It leverages the power of TensorFlow and Keras to build, train, and evaluate the model, ensuring it achieves high accuracy and performance.

## Features

- Data Visualization: Visualize various images from before augmentation and after augmentation, and visualize the loss and accuracy results of the training model.
- Data Preprocessing: Image pixel values are normalized to a range of 0 to 1 to aid in faster convergence during training and utilize `ImageDataGenerator` from Keras for augmenting the training set by applying random transformations like rotation, zoom, shear, and flips to the images. This helps the model generalize better and prevents overfitting.
- Model Architecture:
  * Convolutional Neural Network (CNN): The model uses several convolutional layers which are effective for feature extraction in image data.
  * Pooling Layers: Max pooling layers are used to reduce the spatial dimensions of the output volumes, thus reducing the number of parameters and computations in the network.
  * Batch Normalization: Normalize the activations of the previous layer, which helps to accelerate the training process and improve the stability of the neural network.
  * Dropout Layers: Included to prevent overfitting by randomly dropping units from the neural network during training.
  * Fully Connected Layers: Dense layers at the end of the network that perform classification based on the features extracted by the convolutional layers.

## Prerequisites

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- NLTK
- TensorFlow
- Keras
- Matplotlib
- Seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sfbernado/Traffic-Sign-Classification.git
```

2. Navigate to the project repository:
```bash
cd Traffic-Sign-Classification
```

3. Install the required packages:
```bash
pip install numpy pandas tensorflow keras matplotlib seaborn
```

or

```bash
pip install -r requirements.txt
```

4. Run the Jupyter Notebook:
```bash
jupyter notebook Traffic_Sign_Classification.ipynb
```

5. Run the notebook cells sequentially

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Pandas](https://pandas.pydata.org/) library
- [NumPy](https://numpy.org/) library
- [TensorFlow](https://www.tensorflow.org/) library
- [Keras](https://keras.io/) library
- [Matplotlib](https://matplotlib.org/) library
- [Seaborn](https://seaborn.pydata.org/) library

## Author

Stanislaus Frans Bernado

[![Gmail Badge](https://img.shields.io/badge/-stanislausfb@gmail.com-c14438?style=flat&logo=Gmail&logoColor=white)](mailto:stanislausfb@gmail.com "Connect via Email")
[![Linkedin Badge](https://img.shields.io/badge/-Stanislaus%20Frans%20Bernado-0072b1?style=flat&logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/stanislausfb/ "Connect on LinkedIn")
