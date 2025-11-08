from utils.utilities import Utilities
from preprocessing.signature_preprocessing import ImageProcessor

class SignatureFeatureExtraction:
        
    def preprocess_and_feature_extraction_radon_transform_features(location):
        processed_image = ImageProcessor.read_and_preprocess(location)
        if processed_image is not None:
            #print("Image preprocessing completed successfully.")
            # Extract features or proceed with training
            utils = Utilities()
            return utils.extract_features_discrete_radon_transform(processed_image)     
        else:
            print("Image preprocessing failed.")
            return None