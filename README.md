# Malaria Classification

This repository contains the code and resources for a deep learning model that classifies malaria-infected cells. The model is trained on a dataset of microscopic images and deployed in a simple dashboard for easy and efficient malaria diagnosis.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Approach](#approach)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Introduction
Malaria is a life-threatening disease prevalent in many parts of the world. Early and accurate diagnosis is crucial for effective treatment and prevention of its spread. This project aims to provide a reliable and automated solution for malaria diagnosis using deep learning techniques.

## Dataset
The dataset used for training and evaluation consists of microscopic images of blood cells. Each image is labeled as either "Parasitized" (infected with malaria) or "Uninfected" (not infected). The dataset is divided into training, validation, and test sets to train the model and assess its performance.

## Approach
The following steps were taken to solve the malaria classification problem:

1. Data preprocessing: The dataset images were preprocessed by resizing them to a uniform size and normalizing the pixel values for better model performance.

2. Model selection: The MalariaClassifier model, a convolutional neural network (CNN), was chosen for its ability to effectively handle image classification tasks. The model architecture was implemented using the PyTorch deep learning framework.

3. Training: The model was trained using the training set with the incorporation of data augmentation techniques such as random cropping, rotation, and color jittering. The training process involved optimizing the model parameters using stochastic gradient descent (SGD) with momentum.

4. Model evaluation: The trained model was evaluated using the validation set to assess its performance metrics, including accuracy, precision, and recall. The evaluation results were used to fine-tune the model and improve its performance.

5. Explainability: Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps were employed to visualize the regions of the input image that contributed most significantly to the model's prediction. This provided insights into the decision-making process of the model.

6. Deployment: The trained model was deployed in a simple dashboard interface, allowing users to upload images and receive real-time predictions for malaria infection. The dashboard provides an intuitive and user-friendly interface for efficient malaria diagnosis.

## Installation
To run the code and deploy the dashboard, follow these installation steps:

1. Clone the repository:
   ```shell
   git clone https://github.com/khethan123/Malaria_IBQ.git
   cd Malaria_IBQ
   ```

2. Set up the Python environment:
   - Create a virtual environment (optional but recommended):
     ```shell
     python -m venv env
     source env/bin/activate  # On Windows: env\Scripts\activate
     ```
   - Install the required dependencies:
     ```shell
     pip install -r requirements.txt
     ```

3. Download the dataset:
   - The dataset can be obtained from [source-link](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).
   - Place the dataset in the appropriate directory (e.g., `data/`).

4. Run the code and launch the dashboard:
   ```shell
   streamlit run app.py  # Launch the dashboard
   ```

## Usage
Follow these steps to use the malaria classification dashboard:

1. Launch the dashboard by running the following command:
   ```shell
   streamlit run app.py
   ```

2. Upload an image containing blood cells that you want to classify.

3. Click the "Classify" button to obtain the prediction result, indicating whether the cells are infected with malaria or not.

## Results
Result images and detailed description of findings and model performance metrics are present in [source-link]()

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request. Let's collaborate to enhance the accuracy and usability of the malaria classification model.

