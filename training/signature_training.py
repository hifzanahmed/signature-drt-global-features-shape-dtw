from utils.utilities import Utilities
from features.signature_feature_extraction import SignatureFeatureExtraction
import config
import numpy as np


class SignatureTraining:
    
    def training_genuine_with_shape_dtw(location, trainingSize):
        print("Training on Genuine Signatures...")
        featureList = []  # define an empty list to hold features 
        for i in range(1, trainingSize + 1):
            feature = SignatureFeatureExtraction.preprocess_and_feature_extraction_radon_transform_features(f'{location}{i}.png')
            if feature is not None:
                featureList.append(feature) # append each feature to the list
            else:
                print(f"Warning: Failed to process {location}")
        
        if len(featureList) == 0:
            print("No valid features extracted. Training aborted.")
            return None
    
        # Normalize all features collectively (optional)
        all_features = np.array(featureList, dtype=np.float32)
        min_val = np.min(all_features)
        max_val = np.max(all_features)
        normalized = [(f - min_val) / (max_val - min_val + 1e-8) for f in featureList]

        # Store for verification later
        config.global_features = normalized
        utils = Utilities()
        dtw_distance = utils.compute_training_score(normalized)
        #print("Computed DTW Distance:", dtw_distance)
        return dtw_distance
    
    




