from clip_service.model import predict_disaster
import os 

if __name__ == "__main__":
    image_path = "95f64c24ee004ce29dc6ba3121672686.jpeg"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} not found. Please add an image in sample_images/")

    label, confidence = predict_disaster(image_path)
    print(f"Predicted: {label} (confidence: {confidence:.2f})")
