import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

from model.model import BaseModel
from milPart.model.multi_model import MultiModel
from milPart.model.large_multi_model import LargeMultiModel
from model.lenet import LeNet
from dataset.conv_img import generate_spectrogram

MODEL_OPTIONS = {
    "CNN + FCC": MultiModel,
    "CNN + FCC L": LargeMultiModel,
    # "LeNet": LeNet,
    "CNN": BaseModel,
}


def load_model(model_class):
    model = model_class(torch.Size([3, 64, 128]), 8, 93)
    print("Loading model: ", model_class.__name__)
    model.load_state_dict(torch.load(f"../data/models/{model_class.__name__}.pth"))
    model.eval()
    return model

def preprocess_spectrogram(spectrogram_image, mfcc=None):
    """
    Convert spectrogram bytes to a tensor suitable for model input.
    """
    from torchvision import transforms

    transform = transforms.Compose([
        # transforms.Resize((64, 128)),
        transforms.ToTensor(),
    ])

    spectrogram_tensor = transform(spectrogram_image)

    if mfcc is not None:
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        return spectrogram_tensor.unsqueeze(0), mfcc_tensor.unsqueeze(0)

    return transform(spectrogram_tensor).unsqueeze(0)


st.title("Multi-Model Audio Testing App")

selected_model_name = st.selectbox("Choose a model:", list(MODEL_OPTIONS.keys()))
model_class = MODEL_OPTIONS[selected_model_name]
model = load_model(model_class)

uploaded_audio = st.file_uploader("Upload an audio file (e.g., .wav, .mp3)", type=["wav", "mp3"])

prediction_result = st.empty()

if uploaded_audio is not None:
    try:
        spectrogram_buf, mfcc = generate_spectrogram(
            mp3_file=uploaded_audio,
            generate_mfcc=(selected_model_name == "CNN + FCC"),
        )

        spectrogram_image = Image.open(spectrogram_buf).convert("RGB")

        st.image(spectrogram_image, caption="Generated Spectrogram", use_container_width=True)

        # Form for prediction
        with st.form(key="prediction_form"):
            predict_button = st.form_submit_button(label="Make Prediction")

            if predict_button:
                # Preprocess inputs based on the selected model
                if selected_model_name == "LeNet" or selected_model_name == "CNN":
                    spectrogram_tensor = preprocess_spectrogram(spectrogram_image)
                    input_data = (spectrogram_tensor)
                elif selected_model_name == "CNN + FCC":
                    spectrogram_tensor, feature_tensor = preprocess_spectrogram(spectrogram_image, mfcc)
                    input_data = (spectrogram_tensor, feature_tensor)

                with torch.no_grad():
                    prediction = model(*input_data)
                    
                # Display the prediction result
                genres = ['Folk' 'Hip-Hop' 'Electronic' 'Rock' 'Classical' 'Old-Time']
                probabilities = torch.nn.functional.softmax(prediction, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                prediction_result.write(f"Predicted Class: {genres[predicted_class]}")

    except Exception as e:
        st.error(f"{e}")