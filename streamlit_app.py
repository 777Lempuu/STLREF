import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import gdown
import os

# STL-10 class names
CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 
           'dog', 'horse', 'monkey', 'ship', 'truck']

# Define the model architecture (must match training)
class DeFixMatchSTLModel(nn.Module):
    def __init__(self, num_classes=10):
        super(DeFixMatchSTLModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@st.cache_resource
def download_model():
    """Download the model weights from Google Drive if not already present"""
    model_path = 'STL_ReFix.pt'
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1IteLug8tGJdmzHQMkveOYxRYaKFmt5Zz"
        gdown.download(url, model_path, quiet=False)
    return model_path

@st.cache_resource
def load_model():
    """Load the model with cached weights"""
    model_path = download_model()
    model = DeFixMatchSTLModel(len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    """Transform image for model input"""
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

def main():
    st.set_page_config(page_title="STL-10 Classifier", page_icon="🖼️")
    
    st.title("STL-10 Image Classifier")
    st.write("""
    This app classifies images into one of 10 STL-10 categories using a trained ReFixMatch model.
    Upload an image of an airplane, bird, car, cat, deer, dog, horse, monkey, ship, or truck.
    """)
    
    # Sidebar with info
    st.sidebar.header("About")
    st.sidebar.info("""
    - Model: ReFixMatch trained on STL-10 dataset
    - Classes: 10 common object categories
    - Input: 96x96 RGB images
    """)
    
    # Load model (cached)
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Select an image of one of the STL-10 categories"
    )
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Preprocess and predict
            with st.spinner('Analyzing image...'):
                input_tensor = preprocess_image(image)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidences = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
                    
            # Show top prediction
            predicted_class = max(confidences, key=confidences.get)
            st.success(f"**Prediction:** {predicted_class} (confidence: {confidences[predicted_class]:.1%})")
            
            # Show confidence bar chart
            st.subheader("Confidence Scores")
            st.bar_chart({k: v for k, v in sorted(confidences.items(), key=lambda item: item[1], reverse=True)})
            
            # Show detailed probabilities
            with st.expander("See detailed probabilities"):
                for class_name, prob in sorted(confidences.items(), key=lambda x: -x[1]):
                    st.write(f"▸ {class_name}: {prob:.2%}")
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
