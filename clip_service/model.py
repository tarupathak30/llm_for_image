import torch
import clip
from PIL import Image

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model, preprocess = clip.load("ViT-B/32", device=device)

# Labels your system can recognize
disaster_labels = [
    "tsunami",
    "flooding",
    "storm surge",
    "high waves",
    "coastal damage",
    "no hazard",
    "heavy rain"
]

# Tokenize labels once
text_tokens = clip.tokenize(disaster_labels).to(device)

def predict_disaster(image_path: str):
    """
    Classify an image into disaster categories using CLIP.
    Returns (predicted_label, confidence_score).
    """
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity → probabilities
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(1)
    return disaster_labels[indices[0]], values[0].item()
