import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from torchvision import models

# Load the model architecture
model = models.resnet50(pretrained=False)

# Load the model state dictionary
MODEL_PATH = "best_resnet50_model_Sagittal T1.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as per model requirements
    transforms.ToTensor(),
])

st.title("Upload an Image for Model Inference")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and process the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
    
    st.write("Raw Model Output:")
    st.write(output)
