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
            featureList.append(feature)  # append each feature to the list
        
        normalized = []
        for sig in featureList:
            sig = sig.astype(np.float32)
            sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-8)
            normalized.append(sig)
        config.global_features = normalized  # Store normalized features globally for training and verification phase    
        utils = Utilities()
        dtw_distance = utils.compute_training_score(normalized)
        #print("Computed DTW Distance:", dtw_distance)
        return dtw_distance
    
    




