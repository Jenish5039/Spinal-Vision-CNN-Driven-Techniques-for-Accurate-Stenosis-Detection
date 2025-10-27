import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define class labels
CLASS_LABELS = [
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
    "Spinal Canal Stenosis"
]

NUM_CLASSES = len(CLASS_LABELS)  # Number of classes in the model

# Load the model architecture and state dict
@st.cache_resource
def load_model():
    model = models.mnasnet1_0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    
    state_dict = torch.load("best_model_MNASNet-1.pt", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

st.title("Spinal Stenosis Classification")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    model = load_model()
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Convert to probabilities
        predicted_class = torch.argmax(probabilities).item()  # Get highest probability class
    
    st.write("### Prediction:")
    st.write(f"**{CLASS_LABELS[predicted_class]}**")
    
    st.write("### Raw Model Output (Probabilities):")
    for i, prob in enumerate(probabilities.numpy()):
        st.write(f"{CLASS_LABELS[i]}: {prob:.4f}")
