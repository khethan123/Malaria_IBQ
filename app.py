
import streamlit as st
import cv2
import torch
import numpy as np
import torch.nn as nn

from torchvision import transforms as T
from PIL import Image
from fpdf import FPDF


SZ = 128

trn_tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(SZ),
    T.CenterCrop(SZ),
    T.ColorJitter(brightness=(0.95, 1.05),
                  contrast=(0.95, 1.05),
                  saturation=(0.95, 1.05),
                  hue=0.05),
    T.RandomAffine(5, translate=(0.01, 0.1)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])

def convBlock(ni, no):
    """Create a convolutional block."""
    return nn.Sequential(
        nn.Dropout(0.2),
        nn.Conv2d(ni, no, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(2),
    )

class MalariaClassifier(nn.Module):
    """Malaria classifier model."""
    def __init__(self):
        """Initialize the MalariaClassifier model."""
        super().__init__()
        self.model = nn.Sequential(
            convBlock(3, 64),
            convBlock(64, 64),
            convBlock(64, 128),
            convBlock(128, 256),
            convBlock(256, 512),
            convBlock(512, 64),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, len(id2int))
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """Forward pass of the model."""
        return self.model(x)

    def compute_metrics(self, preds, targets):
        """Compute the loss and accuracy metrics."""
        loss = self.loss_fn(preds, targets)
        acc = (torch.max(preds, 1)[1] == targets).float().mean()
        return loss, acc

# Define the id2int variable here
id2int = ['Parasitized', 'Uninfected']

model = MalariaClassifier()
model_state_dict = torch.load("malaria_classifier_model.pth")
model.load_state_dict(model_state_dict)
model.eval()


im2fmap = nn.Sequential(*(list(model.model[:5].children()) + list(model.model[5][:2].children())))


def preprocess_image(image):
    """convert image to an ndarry"""
    image_array = np.array(image)
    tensor_image = trn_tfms(image_array).unsqueeze(0)
    return tensor_image


def im2gradCAM(x):
    logits = model(x)
    heatmaps = []
    activations = im2fmap(x)
    pred = logits.max(-1)[-1]
    model.zero_grad()
    logits[0, pred].backward(retain_graph=True)
    pooled_grads = model.model[-6][1].weight.grad.data.mean((1, 2, 3))
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grads[i]
    heatmap = torch.mean(activations, dim=1)[0].cpu().detach()
    return heatmap, 'Uninfected' if pred.item() else 'Parasitized'

def upsampleHeatmap(map, img):
    heatmap = cv2.resize(np.array(map), (img.shape[1], img.shape[0]))
    m, M = heatmap.min(), heatmap.max()
    heatmap = 255 * ((heatmap - m) / (M - m))
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(255 - heatmap, cv2.COLORMAP_JET)
    heatmap = np.uint8(heatmap)
    heatmap = np.uint8(heatmap * 0.7 + img * 0.3)
    return heatmap


def resize(image, size):
    return cv2.resize(image, (size, size))

    
def main():
    st.title("Malaria Classifier")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        resized_image = resize(np.array(image), image.size[0]+20)
        st.image(resized_image, caption="Uploaded Image", use_column_width=True)
        tensor_image = preprocess_image(image)
        heatmap, pred_label = im2gradCAM(tensor_image)
        heatmap_image = upsampleHeatmap(heatmap, np.array(image))

        col1, col2 = st.columns(2)
        col1.subheader("Original Image")
        col1.image(resized_image, caption="Uploaded Image", use_column_width=True)

        col2.subheader("Heat Map")
        col2.image(heatmap_image, caption="Heat Map", use_column_width=True)

        st.subheader("Prediction")
        probabilities = torch.softmax(model(tensor_image), dim=1)[0]
        parasitized_prob = probabilities[0].item() * 100
        uninfected_prob = (probabilities[1].item() * 1000)/10
        st.write(f"Predicted Label: {pred_label}")
        st.write(f"Parasitized Probability: {parasitized_prob:.2f}%")
        st.write(f"Uninfected Probability: {uninfected_prob:.2f}%")


if __name__ == '__main__':
    main()
