import cv2
from transformers import CLIPProcessor, CLIPModel

class FeatureExtractor:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def extract_features(self, image_path):
        """
        Extracts features from an image using a pre-trained model.
        
        Parameters:
            image_path (str): Path to the image file.
            
        Returns:
            numpy.ndarray: Extracted features.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        features = outputs.detach().numpy()

        return features