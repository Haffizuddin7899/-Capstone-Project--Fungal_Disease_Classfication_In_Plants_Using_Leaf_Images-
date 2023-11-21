# -Capstone-Project--Fungal_Disease_Classfication_In_Plants_Using_Leaf_Images-
# Project Title: Plant Disease Classification using Leaf Images
# Project Overview
The "Plant Disease Classification using Leaf Images" project leverages artificial intelligence and computer vision technologies to address agricultural challenges faced by farmers. The primary objective is to develop a system for the early detection and diagnosis of crop diseases by analyzing images of plant leaves. Early disease detection is crucial for optimizing crop yields and ensuring healthier crops.

# Dataset
The project utilizes the "PlantVillage Dataset" sourced from Kaggle, containing a diverse collection of plant images, including various crops such as apple, cherry, corn, grape, potato, and tomato. The dataset encompasses 22 classes and a total of 22,387 images.

# Dataset Structure
APPLE (4 classes)
CHERRY (2 classes)
CORN (4 classes)
GRAPE (3 classes)
POTATO (3 classes)
TOMATO (6 classes)
# Model Architecture
The model is based on the VGG16 architecture, with the top (fully connected) layers replaced to suit the classification task. The key components of the model include:

Input layer: Accepts images of size 224x224x3.
Convolutional layers: 13 convolutional layers with ReLU activation functions.
Flatten layer: Flattens the output of the last max pooling layer.
Dense layers: Two fully connected layers with ReLU activation functions.
Dropout layer: Prevents overfitting by randomly dropping out neurons during training.
Output layer: Dense layer with 22 neurons, representing the 22 classes, and a softmax activation function.
# Model Training
The training process involves loading the pre-trained VGG16 model, freezing the pre-trained layers, and adding custom top layers. The model is compiled using the Adam optimizer and categorical crossentropy loss function. The training set undergoes augmentation using the ImageDataGenerator, enhancing robustness and generalization.

# Training Results
Training Accuracy: approximately 73.46%
Validation Accuracy: approximately 90.00%
Test Accuracy: 89.76%
F1-Score: 0.90
Training Loss: approximately 73.42%
Validation Loss: approximately 28.27%
# Data Preprocessing
The project employs the TensorFlow ImageDataGenerator for data preprocessing and augmentation. The training data is split into training and validation sets using a specified validation split of 20%. The test data undergoes normalization.

# Exploratory Data Analysis (EDA)
The EDA section includes visualizations of class distribution and a histogram of average pixel intensity, providing insights into the dataset.

# Class Weighting
Class weights are calculated manually to address class imbalance and normalize them for training.

# Model Evaluation
The trained model is evaluated on the test set, and metrics such as accuracy, precision, recall, F1-score, and a confusion matrix are presented. The precision-recall curve is plotted for each class.

# TensorBoard
TensorBoard is utilized for visualizing training and validation metrics, providing insights into the model's performance.

# Usage Example
A function named test_single_image is provided to test the model with individual images, demonstrating how to use the trained model for predictions.

# Future Improvements
Fine-tuning hyperparameters for further optimization.
Enhancing error handling in the testing function.
Continual improvement based on user feedback and new datasets.
Feel free to explore, contribute, and provide feedback to make this project even better!
