import numpy as np
import cv2
from shapedtw import shape_dtw
from skimage.transform import radon
from shapedtw.shapeDescriptors import (
    RawSubsequenceDescriptor, PAADescriptor, DWTDescriptor)

class Utilities:
    @staticmethod
    def load_image(image_path):
        import cv2
        # Load/Read the image from the specified path
        signature = cv2.imread(image_path)
        
        # Check if image is loaded correctly
        if signature is None:
            print("Error: Signature Image not found or unable to load.")
            return None
        return signature
    
    @staticmethod
    def crop_and_resize_signature(image):
        """
        Crops the image by removing zero rows/columns and resizes to target size.
        Preserves aspect ratio by padding with zeros.
        Args:
            img (np.ndarray): 2D array representing the signature.
            size (tuple): Target size (width, height)
        Returns:
            np.ndarray: Cropped and resized image as float32 in [0,1]
        """
        assert image.ndim == 2, "Input must be a 2D array"
        
        # Identify non-zero rows and columns
        rows = np.any(image, axis=1)
        cols = np.any(image, axis=0)
        cropped_img = image[np.ix_(rows, cols)]
        
        # Convert to uint8 for OpenCV operations
        cropped_img_uint8 = (cropped_img * 255).astype(np.uint8) if image.max() <= 1.0 else cropped_img.astype(np.uint8)
        
        # Resize with aspect ratio preservation
        resized_img = Utilities.resize_image(cropped_img_uint8)
        
        # Normalize back to [0,1]
        return resized_img.astype(np.float32) / 255.0
    
    @staticmethod
    def resize_image(img, size=(300, 150)):
        h, w = img.shape[:2]
        scale = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
        # Pad to fit target size
        top = (size[1]-new_h)//2
        bottom = size[1]-new_h-top
        left = (size[0]-new_w)//2
        right = size[0]-new_w-left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return padded
    
    @staticmethod
    def extract_features_discrete_radon_transform(filtered_img, angles=None):
        """
        Extract global DRT features from a preprocessed image.
        Args:
            img (np.ndarray): Preprocessed grayscale image [0,1]
            angles (np.ndarray): Angles in degrees to compute projections
        Returns:
            np.ndarray: 1D feature vector concatenating projections
        """
        if angles is None:
            angles = np.linspace(0, 180, 12, endpoint=False) # 12 angles from 0 to 180 degrees
    
        features = []
        for angle in angles:
            projection = radon(filtered_img, theta=[angle], circle=False)
            projection = projection.flatten()  # 1D vector
            features.append(projection)
    
        features_vector = np.concatenate(features)
        return features_vector
    
    @staticmethod
    def compute_training_score(signatures, subsequence_width=15, descriptor_type="raw"):
        """
        Computes average ShapeDTW distance (S1) among genuine signatures.
        Args:
            signatures (list of np.ndarray): list of 1D DRT feature vectors
            window_size (int): local shape window length
        Returns:
            float: average ShapeDTW distance (S1)
        """
        if descriptor_type == "raw":
            shape_desc = RawSubsequenceDescriptor()
        else:
            raise ValueError(f"Unsupported shape descriptor type: {descriptor_type}")
    
        normalized_signatures = [sig / np.linalg.norm(sig) for sig in signatures]

        K = len(normalized_signatures)
        dist_matrix = np.zeros((K, K))

        for i in range(K):
            for j in range(i + 1, K):
                result = shape_dtw(
                    normalized_signatures[i].reshape(-1, 1),
                    normalized_signatures[j].reshape(-1, 1),
                    subsequence_width=subsequence_width,
                    shape_descriptor=shape_desc  
                )
                dist = result.distance
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        avg_distance = np.sum(np.triu(dist_matrix, 1)) / (K * (K - 1) / 2)
        return avg_distance
    
    @staticmethod
    def compute_verification_score(test_signature, genuine_signatures, subsequence_width=15, descriptor_type="raw"):
        """
        Computes average ShapeDTW distance (S2) between a test signature and all genuine ones.
        """
        if len(genuine_signatures) == 0:
            raise ValueError("No genuine signatures found for verification!")
    
        if descriptor_type == "raw":
            shape_desc = RawSubsequenceDescriptor()
        else:
            raise ValueError(f"Unsupported shape descriptor type: {descriptor_type}")
        
        # Normalize test and genuine signatures
        normalized_test = test_signature / np.linalg.norm(test_signature)
        normalized_genuine = [sig / np.linalg.norm(sig) for sig in genuine_signatures]
        
        total_dist = 0
        for ref in normalized_genuine:
            result = shape_dtw(
                normalized_test.reshape(-1, 1),
                ref.reshape(-1, 1),
                subsequence_width=subsequence_width, 
                shape_descriptor=shape_desc 
            )
            dist = result.distance
            total_dist += dist

        avg_dist = total_dist / len(normalized_genuine)
        return avg_dist